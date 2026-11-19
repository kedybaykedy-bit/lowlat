/* rANS range coder - lossless, integer frequencies (sum = 1<<16) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RANS_L (1u << 23)
#define PRECISION 16
#define TOTAL (1u << PRECISION)

typedef struct {
    uint8_t *buf;
    size_t cap;
    size_t len;
    uint32_t state;
} Enc;

typedef struct {
    const uint8_t *buf;
    size_t len;
    size_t pos;
    uint32_t state;
} Dec;

static int enc_grow(Enc *e, size_t need) {
    if (e->len + need <= e->cap) return 0;
    size_t ncap = e->cap ? e->cap : 4096;
    while (ncap < e->len + need) ncap *= 2;
    uint8_t *n = (uint8_t *)realloc(e->buf, ncap);
    if (!n) return -1;
    e->buf = n;
    e->cap = ncap;
    return 0;
}

void *enc_new(size_t hint) {
    Enc *e = (Enc *)calloc(1, sizeof(Enc));
    if (!e) return NULL;
    e->cap = hint ? hint : 4096;
    e->buf = (uint8_t *)malloc(e->cap);
    if (!e->buf) { free(e); return NULL; }
    e->len = 0;
    e->state = RANS_L;
    return e;
}

/* freq[start..start+freq-1] must be >0; sum of all freqs over full alphabet = TOTAL */
int enc_put(void *h, uint32_t start, uint32_t freq) {
    Enc *e = (Enc *)h;
    if (freq == 0 || freq > TOTAL || start + freq > TOTAL) return -1;
    uint32_t x = e->state;
    uint32_t x_max = ((RANS_L >> PRECISION) << 8) * freq;
    if (x >= x_max) {
        if (enc_grow(e, 16) != 0) return -1;
        do {
            e->buf[e->len++] = (uint8_t)(x & 0xff);
            x >>= 8;
        } while (x >= x_max);
    }
    e->state = ((x / freq) << PRECISION) + (x % freq) + start;
    return 0;
}

/* Flush: write final state (4 bytes LE). Returns bytes written into out, or -1. */
int enc_flush(void *h, uint8_t *out, size_t out_cap, size_t *out_len) {
    Enc *e = (Enc *)h;
    if (out_cap < e->len + 4) return -1;
    /* output state in reverse byte order relative to stream: append state LE */
    uint32_t x = e->state;
    out[0] = (uint8_t)(x & 0xff);
    out[1] = (uint8_t)((x >> 8) & 0xff);
    out[2] = (uint8_t)((x >> 16) & 0xff);
    out[3] = (uint8_t)((x >> 24) & 0xff);
    /* stream body is written backwards during encode; reverse it into out after state?
       Simpler scheme: we wrote renorm bytes in order of emission; decoder reads them
       in reverse. Standard rANS: encoder emits bytes to a stack; final layout is
       [state][reversed renorm bytes] OR we reverse renorm into place here. */
    /* Body was appended in emission order; decoder will read from end. Copy reversed. */
    size_t body = e->len;
    if (out_cap < 4 + body) return -1;
    for (size_t i = 0; i < body; i++) {
        out[4 + i] = e->buf[body - 1 - i];
    }
    *out_len = 4 + body;
    return 0;
}

void enc_free(void *h) {
    Enc *e = (Enc *)h;
    if (!e) return;
    free(e->buf);
    free(e);
}

void *dec_new(const uint8_t *data, size_t len) {
    if (len < 4) return NULL;
    Dec *d = (Dec *)calloc(1, sizeof(Dec));
    if (!d) return NULL;
    d->buf = data;
    d->len = len;
    d->pos = 4; /* first body byte after state */
    d->state = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
               ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
    return d;
}

/* Decode symbol: find start such that cdf[start] <= slot < cdf[start+freq].
   cdf is array of TOTAL+1 ascending values? We pass freq table via lookup callback.
   For speed we pass flat freq array of length TOTAL (impractical) - instead caller
   provides start/freq search. We use: slot = state & (TOTAL-1); caller maps slot. */
uint32_t dec_slot(void *h) {
    Dec *d = (Dec *)h;
    return d->state & (TOTAL - 1);
}

int dec_advance(void *h, uint32_t start, uint32_t freq) {
    Dec *d = (Dec *)h;
    if (freq == 0 || start + freq > TOTAL) return -1;
    uint32_t x = d->state;
    uint32_t mask = TOTAL - 1;
    x = freq * (x >> PRECISION) + (x & mask) - start;
    if (x < RANS_L) {
        do {
            if (d->pos >= d->len) return -1;
            x = (x << 8) | d->buf[d->pos++];
        } while (x < RANS_L);
    }
    d->state = x;
    return 0;
}

void dec_free(void *h) { free(h); }

size_t enc_hint(void *h) { return ((Enc *)h)->len + 16; }

/* Batch API: encode symbols given per-symbol (start, freq). symbols in ORIGINAL order;
   encoder processes reverse internally. returns 0 on success. */
int enc_put_batch(void *h, const uint32_t *starts, const uint32_t *freqs, int n) {
    Enc *e = (Enc *)h;
    for (int i = n - 1; i >= 0; i--) {
        uint32_t start = starts[i], freq = freqs[i];
        if (freq == 0 || freq > TOTAL || start + freq > TOTAL) return -1;
        uint32_t x = e->state;
        uint32_t x_max = ((RANS_L >> PRECISION) << 8) * freq;
        if (x >= x_max) {
            if (enc_grow(e, 16) != 0) return -1;
            do {
                e->buf[e->len++] = (uint8_t)(x & 0xff);
                x >>= 8;
            } while (x >= x_max);
        }
        e->state = ((x / freq) << PRECISION) + (x % freq) + start;
    }
    return 0;
}

/* Decode n symbols. freqs rows are contiguous: row i at freqs + i*alphabet.
   csum optional: if non-NULL, cumulative freqs with row stride = alphabet.
   alphabet = A. Outputs symbols into out[]. */
int dec_symbol_batch(void *h, const uint32_t *freqs, int alphabet, int n, uint32_t *out) {
    Dec *d = (Dec *)h;
    /* build csum buffer once */
    uint32_t *csum = (uint32_t *)malloc((size_t)alphabet * n * sizeof(uint32_t));
    if (!csum) return -1;
    for (int i = 0; i < n; i++) {
        const uint32_t *row = freqs + (size_t)i * alphabet;
        uint32_t acc = 0;
        for (int j = 0; j < alphabet; j++) {
            acc += row[j];
            csum[(size_t)i * alphabet + j] = acc;
        }
    }
    int rc = 0;
    for (int i = 0; i < n; i++) {
        uint32_t slot = d->state & (TOTAL - 1);
        const uint32_t *cs = csum + (size_t)i * alphabet;
        /* binary search */
        int lo = 0, hi = alphabet - 1, idx = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (cs[mid] <= slot) { idx = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        /* cs[idx] <= slot < cs[idx+1] ; find first > slot */
        int l2 = 0, h2 = alphabet - 1;
        idx = 0;
        while (l2 <= h2) {
            int mid = (l2 + h2) >> 1;
            if (cs[mid] > slot) { idx = mid; h2 = mid - 1; }
            else l2 = mid + 1;
        }
        uint32_t start = idx > 0 ? cs[idx - 1] : 0;
        uint32_t freq = cs[idx] - start;
        uint32_t x = d->state;
        uint32_t mask = TOTAL - 1;
        x = freq * (x >> PRECISION) + (x & mask) - start;
        if (x < RANS_L) {
            int ok = 1;
            do {
                if (d->pos >= d->len) { ok = 0; break; }
                x = (x << 8) | d->buf[d->pos++];
            } while (x < RANS_L);
            if (!ok) { rc = -1; break; }
        }
        d->state = x;
        out[i] = (uint32_t)idx;
    }
    free(csum);
    return rc;
}
