/* freq.c - probability -> integer frequency quantization (largest remainder).
 *
 * void freqs_from_probs(const float *probs, int n_rows, int n_sym, uint32_t *out)
 *
 * probs : n_rows * n_sym floats, row-major; each row should sum to ~1.
 * out   : n_rows * n_sym uint32; every row sums to exactly 65536.
 *
 * Per row:
 *   target = 65536 - n_sym
 *   base_j = floor(clamp(p_j) * target) + 1        (min freq 1)
 *   rem    = 65536 - sum(base)
 *   rem > 0 : +1 to the bins with the largest fractional parts
 *             (largest remainder; ties -> larger index first), cycling
 *             through every bin first if rem > n_sym.
 *   rem < 0 : -1 from bins with the smallest fractional parts (never
 *             below 1) - only reachable for badly unnormalized input.
 *
 * Selection uses a histogram over 256 uniform buckets of the fractional
 * part (frac * 65536, an exact power-of-two multiply, then >> 8): buckets
 * are uniform in frac, so the boundary bucket holds only ~n/256 bins and
 * needs an exact sort of those few. The common path never runs a full
 * sort. Signed int conversions are used for float->int casts: v is
 * provably in [0,65535] after clamping, and gcc then emits clean vector
 * code instead of unsigned-conversion fixup sequences (~4x faster pass).
 *
 * Compile: gcc -O3 -shared -fPIC -o libfreq.so freq.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FREQ_TOTAL 65536u
#define BKT_BITS 8
#define BKT_N (1u << BKT_BITS)      /* 256 histogram buckets */
#define FAST_MEMBERS 64             /* exact-sort fallback threshold */

/* Stable LSD radix sort of exact frac-bit keys carrying the bin index.
 * Ascending by frac; equal fracs keep increasing index order (stability).
 * Used only for pathological rows (negative remainder / huge boundary
 * bucket); the normal path uses histogram-select instead. */
static void sort_frac_idx(uint32_t *fk, uint32_t *fi,
                          uint32_t *tk, uint32_t *ti, int n)
{
    uint32_t cnt[256];
    uint32_t *sk = fk, *si = fi, *dk = tk, *di = ti;
    for (int shift = 0; shift < 32; shift += 8) {
        memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < n; i++) cnt[(sk[i] >> shift) & 0xffu]++;
        uint32_t acc = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t c = cnt[b];
            cnt[b] = acc;
            acc += c;
        }
        for (int i = 0; i < n; i++) {
            uint32_t k = sk[i];
            uint32_t p = cnt[(k >> shift) & 0xffu]++;
            dk[p] = k;
            di[p] = si[i];
        }
        uint32_t *tmp;
        tmp = sk; sk = dk; dk = tmp;
        tmp = si; si = di; di = tmp;
    }
    if (sk != fk) {
        memcpy(fk, sk, (size_t)n * sizeof(uint32_t));
        memcpy(fi, si, (size_t)n * sizeof(uint32_t));
    }
}

/* Recompute exact float32 fractional-part bit patterns for fallback sorts. */
static void fill_exact(const float *row, int n_sym, float ftarget,
                       uint32_t *fk, uint32_t *fi)
{
    for (int j = 0; j < n_sym; j++) {
        float v = row[j];
        if (!(v >= 0.0f)) v = 0.0f;
        v *= ftarget;
        if (v > ftarget) v = ftarget;
        /* signed conversions: v is provably in [0,65535] after the clamps,
         * and gcc emits plain cvttps2dq instead of unsigned fixup code */
        int32_t ui = (int32_t)v;
        float frac = v - (float)ui;
        uint32_t bits;
        memcpy(&bits, &frac, sizeof bits);
        fk[j] = bits;
        fi[j] = (uint32_t)j;
    }
}

void freqs_from_probs(const float *probs, int n_rows, int n_sym,
                      uint32_t *out)
{
    if (!probs || !out || n_rows <= 0 || n_sym <= 0) return;
    if ((unsigned)n_sym > FREQ_TOTAL) return; /* min freq 1 each -> impossible */

    const uint32_t target = FREQ_TOTAL - (uint32_t)n_sym;
    const float ftarget = (float)target;

    uint32_t *buf = (uint32_t *)malloc((size_t)n_sym * 5u * sizeof(uint32_t));
    if (!buf) return;
    uint32_t *bkts = buf;                       /* key buckets (frac>>8) */
    uint32_t *fk = buf + n_sym;                 /* exact frac bits (fallback) */
    uint32_t *fi = buf + 2 * (size_t)n_sym;
    uint32_t *tk = buf + 3 * (size_t)n_sym;
    uint32_t *ti = buf + 4 * (size_t)n_sym;
    uint32_t cnt[BKT_N];

    for (int i = 0; i < n_rows; i++) {
        const float *row = probs + (size_t)i * (size_t)n_sym;
        uint32_t *o = out + (size_t)i * (size_t)n_sym;

        /* pass 1: bases + key buckets + sum (vectorizable, no hist) */
        uint32_t sum = 0;
        for (int j = 0; j < n_sym; j++) {
            float v = row[j];
            if (!(v >= 0.0f)) v = 0.0f;   /* NaN / negative -> 0 */
            v *= ftarget;
            if (v > ftarget) v = ftarget; /* clamp unnormalized rows */
            /* v in [0,65535]: signed conversions give gcc clean vector code
             * (plain cvttps2dq/cvtdq2ps, no unsigned-fixup sequences) */
            int32_t ui = (int32_t)v;      /* trunc == floor for v >= 0 */
            uint32_t base = (uint32_t)ui + 1u; /* min freq 1 */
            o[j] = base;
            sum += base;
            /* frac * 65536 is exact (power-of-two multiply), so the bucket
             * is an exact, monotone quantization of the float32 frac */
            int32_t kb = (int32_t)((v - (float)ui) * 65536.0f);
            bkts[j] = (uint32_t)kb >> BKT_BITS;
        }

        int64_t rem = (int64_t)FREQ_TOTAL - (int64_t)sum;
        if (rem > 0) {
            int64_t cycles = rem / n_sym;
            if (cycles > 0) {
                uint32_t c = (uint32_t)cycles;
                for (int j = 0; j < n_sym; j++) o[j] += c;
                rem -= cycles * (int64_t)n_sym;
            }
            if (rem > 0) {
                /* pass 2: histogram over key buckets */
                memset(cnt, 0, sizeof(cnt));
                for (int j = 0; j < n_sym; j++) cnt[bkts[j]]++;

                /* walk down from the largest bucket to the boundary that
                 * absorbs the remainder */
                uint32_t acc = 0;
                int b = (int)BKT_N - 1;
                for (; b >= 0; b--) {
                    uint32_t cb = cnt[b];
                    if (acc + cb >= (uint32_t)rem) break;
                    acc += cb;
                }
                uint32_t need = (uint32_t)rem - acc; /* in [1, cnt[b]] */

                if (cnt[b] <= FAST_MEMBERS) {
                    /* fast path: vectorizable +1 for higher buckets, then a
                     * scalar pass collecting boundary members by exact frac */
                    uint32_t bb = (uint32_t)b;
                    for (int j = 0; j < n_sym; j++)
                        o[j] += (bkts[j] > bb);
                    uint32_t members[FAST_MEMBERS];
                    float mfrac[FAST_MEMBERS];
                    int m = 0;
                    for (int j = 0; j < n_sym; j++) {
                        if (bkts[j] == bb) {
                            float v = row[j];
                            if (!(v >= 0.0f)) v = 0.0f;
                            v *= ftarget;
                            if (v > ftarget) v = ftarget;
                            int32_t ui = (int32_t)v;
                            float fa = v - (float)ui;
                            int pos = m;
                            while (pos > 0 && mfrac[pos - 1] > fa) {
                                members[pos] = members[pos - 1];
                                mfrac[pos] = mfrac[pos - 1];
                                pos--;
                            }
                            members[pos] = (uint32_t)j;
                            mfrac[pos] = fa;
                            m++;
                        }
                    }
                    /* +1 to the `need` largest fracs in the boundary bucket
                     * (stable insertion keeps index order for equal fracs,
                     * so taking from the end matches stable argsort) */
                    for (int a = m - (int)need; a < m; a++)
                        o[members[a]] += 1u;
                } else {
                    /* rare: boundary bucket too large - full exact sort */
                    fill_exact(row, n_sym, ftarget, fk, fi);
                    sort_frac_idx(fk, fi, tk, ti, n_sym);
                    for (int64_t k = 0; k < rem; k++)
                        o[fi[n_sym - 1 - (int)k]] += 1u;
                }
            }
        } else if (rem < 0) {
            /* pathological unnormalized row: -1 from smallest fracs, >=1 */
            fill_exact(row, n_sym, ftarget, fk, fi);
            sort_frac_idx(fk, fi, tk, ti, n_sym);
            int64_t need = -rem;
            while (need > 0) {
                int moved = 0;
                for (int k = 0; k < n_sym && need > 0; k++) {
                    uint32_t idx = fi[k];
                    if (o[idx] > 1u) {
                        o[idx] -= 1u;
                        need--;
                        moved = 1;
                    }
                }
                if (!moved) break;
            }
        }
    }
    free(buf);
}
