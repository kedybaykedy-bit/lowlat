/* ccodec.c — fast lossless/adaptive screen-content codec core.
 *
 * FEATURES
 *   - dirty-tile inter coding, integer motion estimation (neighbour-MV hint)
 *   - per-tile hybrid lossless methods: varint residual / raw / changed-pixel submask
 *   - TILE CACHE (FIFO ring): a tile identical to a previously-sent tile costs ~7 bytes
 *     (a reference id) instead of the full payload. Great for repeated UI chrome,
 *     scrolled-back content, restored windows, duplicated widgets.
 *   - TEXT-PROTECTED ADAPTIVE QUALITY: when a per-frame byte budget is exceeded,
 *     only NON-TEXT tiles are degraded (skip / 2x downsample). Text-ish tiles
 *     (high edge density + bimodal luma) are ALWAYS coded losslessly.
 *
 * FRAME BITSTREAM
 *   u8  flags  (bit0=keyframe, bit1=has_lossy)
 *   keyframe: u16 cache_cap, u32 clen, deflate( PNG-adaptive-filtered RGB )
 *   inter:    u16 ndirty, u32 tlen,
 *             deflate( rle_bitmap | per-tile records )
 *   rle_bitmap: (u8 val, u16 run)*  (nt*nx values total)
 *   per-tile record (raster order of the '1' bits):
 *       i8 mvy, i8 mvx, u8 method, u32 rawlen, raw[rawlen]
 *       method 0 = zigzag-varint residual(pred)
 *       method 1 = raw tile
 *       method 2 = changed-pixel submask + zigzag-varint
 *       method 3 = CACHE REF, rawlen=2, raw=u16 ring slot
 *       method 4 = SKIP (no payload; keep previous reconstruction)
 *       method 5 = DOWN (rawlen=dh*dw*3 downsampled tile; decoder upsamples)
 *
 * Both endpoints keep identical recon + cache; insertion/eviction is deterministic
 * (raster order), so cache refs always resolve identically.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* ------------------------------------------------------------------ buffers */
typedef struct { uint8_t *buf; size_t len, cap; int err; } Buf;
static void buf_init(Buf*b){b->buf=NULL;b->len=b->cap=0;b->err=0;}
static void buf_free(Buf*b){free(b->buf);b->buf=NULL;b->len=b->cap=0;}
static void buf_need(Buf*b,size_t e){ if(b->err)return; if(b->len+e<=b->cap)return;
    size_t nc=b->cap?b->cap:4096; while(nc<b->len+e)nc*=2;
    uint8_t*n=realloc(b->buf,nc); if(!n){b->err=1;return;} b->buf=n;b->cap=nc;}
static void buf_u8(Buf*b,uint8_t v){buf_need(b,1);if(!b->err)b->buf[b->len++]=v;}
static void buf_bytes(Buf*b,const void*p,size_t n){buf_need(b,n);if(b->err)return;memcpy(b->buf+b->len,p,n);b->len+=n;}
static void buf_u16(Buf*b,uint16_t v){uint8_t t[2]={(uint8_t)v,(uint8_t)(v>>8)};buf_bytes(b,t,2);}
static void buf_u32(Buf*b,uint32_t v){uint8_t t[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)};buf_bytes(b,t,4);}
static int deflate_into(const uint8_t*in,size_t n,Buf*out,int level){ if(out->err)return -1;
    uLongf bound=compressBound((uLong)n); buf_need(out,bound); if(out->err)return -1;
    uLongf dl=bound; if(compress2(out->buf+out->len,&dl,in,(uLong)n,level)!=Z_OK)return -1; out->len+=dl; return 0; }

static int iabs(int v){return v<0?-v:v;}

/* ------------------------------------------------------------------ cache */
typedef struct { uint8_t *pix; int h, w, used; uint64_t hash; } TEnt;
typedef struct { TEnt *e; int cap, n, head; } TCache;

static uint64_t tile_hash(const uint8_t*p,int h,int w){
    uint64_t s=1469598103934665603ULL; /* FNV-ish */
    size_t n=(size_t)h*w*3;
    for(size_t i=0;i<n;i++){ s^=p[i]; s*=1099511628211ULL; }
    s^=(uint64_t)((unsigned)h<<20)^(unsigned)w;
    s^=s>>33; s*=0xff51afd7ed558ccdULL; s^=s>>29;
    return s;
}
static void cache_init(TCache*c,int cap,int tile){
    c->cap=cap; c->n=0; c->head=0; c->e=NULL;
    if(cap<=0)return;
    c->e=calloc((size_t)cap,sizeof(TEnt));
    if(!c->e){c->cap=0;return;}
    for(int i=0;i<cap;i++){ c->e[i].pix=malloc((size_t)tile*tile*3); if(!c->e[i].pix){c->cap=0;break;} }
}
static void cache_free(TCache*c){
    if(!c->e)return;
    for(int i=0;i<c->cap;i++)free(c->e[i].pix);
    free(c->e); c->e=NULL; c->n=0; c->head=0; c->cap=0;
}
static void cache_clear(TCache*c){ c->n=0; c->head=0; for(int i=0;i<c->cap;i++)c->e[i].used=0; }
static int cache_find(TCache*c,const uint8_t*p,int h,int w){
    if(!c->e||c->cap<=0||c->n<=0)return -1;
    uint64_t hh=tile_hash(p,h,w);
    size_t n=(size_t)h*w*3;
    for(int i=0;i<c->n;i++){
        int idx=(c->head+i)%c->cap;
        TEnt*e=&c->e[idx];
        if(e->used&&e->hash==hh&&e->h==h&&e->w==w&&memcmp(e->pix,p,n)==0)return idx;
    }
    return -1;
}
static void cache_insert(TCache*c,const uint8_t*p,int h,int w){
    if(!c->e||c->cap<=0)return;
    int idx;
    if(c->n<c->cap){ idx=(c->head+c->n)%c->cap; c->n++; }
    else { idx=c->head; c->head=(c->head+1)%c->cap; }
    TEnt*e=&c->e[idx];
    size_t n=(size_t)h*w*3;
    memcpy(e->pix,p,n); e->h=h; e->w=w; e->used=1; e->hash=tile_hash(p,h,w);
}

/* ------------------------------------------------------------- text detect */
/* A tile is "text-like" if it has many strong local edges AND a bimodal luma
 * histogram (mass at both the dark and light ends). Photos/noise/gradients are
 * not text-like and may be degraded under bandwidth pressure. Conservative:
 * anything borderline is treated as text (never degraded). */
static int is_text_tile(const uint8_t*p,int h,int w){
    if(h<4||w<4)return 1;
    long strong=0, total=0;
    int lmin=255,lmax=0;
    long histo[16]={0};
    for(int y=0;y<h;y++){
        const uint8_t*row=p+(size_t)y*w*3;
        int pl=-1;
        for(int x=0;x<w;x++){
            const uint8_t*q=row+(size_t)x*3;
            int l=(q[0]*77+q[1]*150+q[2]*29)>>8;
            if(l<lmin)lmin=l; if(l>lmax)lmax=l;
            histo[(l*16)>>8]++;
            if(pl>=0){ int d=iabs(l-pl); if(d>24)strong++; total++; }
            pl=l;
        }
    }
    for(int y=1;y<h;y++){
        const uint8_t*row=p+(size_t)y*w*3, *up=row-(size_t)w*3;
        for(int x=0;x<w;x++){
            int l=(row[x*3]*77+row[x*3+1]*150+row[x*3+2]*29)>>8;
            int lu=(up[x*3]*77+up[x*3+1]*150+up[x*3+2]*29)>>8;
            if(iabs(l-lu)>24)strong++; total++;
        }
    }
    if(total<=0)return 0;
    double edge=(double)strong/(double)total;
    /* bimodality: pixels near the tile's luma extremes */
    int lo=lmin+16, hi=lmax-16; if(hi<lo)hi=lo;
    long ext=0, tot2=0;
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){
        const uint8_t*q=p+((size_t)y*w+x)*3;
        int l=(q[0]*77+q[1]*150+q[2]*29)>>8;
        if(l<=lo||l>=hi)ext++; tot2++;
    }
    double bim = tot2 ? (double)ext/(double)tot2 : 0.0;
    if(edge>0.12) return 1;                 /* dense strokes -> protect */
    if(edge>0.045 && bim>0.55) return 1;    /* text on flat bg -> protect */
    return 0;
}

/* --------------------------------------------------------- small helpers */
static uint8_t *fp(uint8_t*f,int cw,int y,int x){return f+((size_t)y*cw+x)*3;}

static long tile_sad(const uint8_t*cur,const uint8_t*ref,int ww,int hh,int ry,int rx,int cw){
    long s=0;
    for(int r=0;r<hh;r++){
        const uint8_t*a=cur+(size_t)r*ww*3;
        const uint8_t*b=ref+((size_t)(ry+r)*cw+rx)*3;
        for(int k=0;k<ww*3;k++)s+=iabs((int)a[k]-(int)b[k]);
    }
    return s;
}

static size_t enc_down4(uint8_t*dst,const uint8_t*src,int hh,int ww){
    int dw=(ww+3)/4, dh=(hh+3)/4; size_t p=0;
    for(int y=0;y<dh;y++)for(int x=0;x<dw;x++){
        int x0=x*4,y0=y*4,x1=x0+3<ww?x0+3:ww-1,y1=y0+3<hh?y0+3:hh-1;
        for(int c=0;c<3;c++){int sm=0,nn=0;
            for(int yy=y0;yy<=y1;yy++)for(int xx=x0;xx<=x1;xx++){sm+=src[((size_t)yy*ww+xx)*3+c];nn++;}
            dst[p++]=(uint8_t)(sm/nn);}
    }
    return p;
}
static void dec_up4(uint8_t*dst,const uint8_t*src,int hh,int ww){
    int dw=(ww+3)/4, dh=(hh+3)/4;
    for(int y=0;y<hh;y++){int sy=(y*dh)/hh;
        for(int x=0;x<ww;x++){int sx=(x*dw)/ww;
            const uint8_t*s=src+(size_t)(sy*dw+sx)*3; uint8_t*d=dst+((size_t)y*ww+x)*3;
            d[0]=s[0];d[1]=s[1];d[2]=s[2];}}
}

/* global scroll dy estimate (sampled luma SAD) */
static int global_dy(int H,int W,const uint8_t*frame,const uint8_t*ref,int hint){
    long best=1L<<60; int bdy=0;
    int stepy=H>400?8:4, stepx=W>600?8:4;
    int lo=hint-4, hi=hint+4;
    if(lo<-64)lo=-64; if(hi>64)hi=64;
    for(int dy=lo;dy<=hi;dy++){
        long sad=0; int cnt=0;
        for(int y=0;y<H;y+=stepy){
            int yy=y+dy; if(yy<0||yy>=H)continue;
            const uint8_t*a=frame+((size_t)y*W)*3, *b=ref+((size_t)yy*W)*3;
            for(int x=0;x<W;x+=stepx){
                sad += iabs((int)a[x*3]-(int)b[x*3]); cnt++;
            }
        }
        if(cnt>0 && sad<best){best=sad;bdy=dy;}
    }
    return bdy;
}

/* --- tile payload decoders --- */
static int dec_varint(const uint8_t*raw,size_t rl,size_t*po,const uint8_t*pred,int n,uint8_t*out){
    size_t p=*po;
    for(int i=0;i<n;i++){
        uint32_t zz=0;int sh=0;
        while(1){ if(p>=rl)return -1; uint8_t b=raw[p++]; zz|=(uint32_t)(b&0x7F)<<sh; if(!(b&0x80))break; sh+=7; }
        int v=(int)((zz>>1)^-(int)(zz&1));
        int val=(int)pred[i]+v; if(val<0)val=0; else if(val>255)val=255; out[i]=(uint8_t)val;
    }
    *po=p; return 0;
}
static int dec_submask(const uint8_t*raw,size_t rl,size_t*po,const uint8_t*pred,int hh,int ww,uint8_t*out){
    int np=hh*ww; size_t mb=(np+7)/8; if(*po+mb>rl)return -1;
    const uint8_t*mask=raw+*po; size_t p=*po+mb;
    memcpy(out,pred,(size_t)np*3);
    for(int i=0;i<np;i++){
        if(!(mask[i>>3]&(1u<<(i&7))))continue;
        for(int k=0;k<3;k++){
            uint32_t zz=0;int sh=0;
            while(1){ if(p>=rl)return -1; uint8_t b=raw[p++]; zz|=(uint32_t)(b&0x7F)<<sh; if(!(b&0x80))break; sh+=7; }
            int v=(int)((zz>>1)^-(int)(zz&1));
            int val=(int)pred[i*3+k]+v; if(val<0)val=0; else if(val>255)val=255; out[i*3+k]=(uint8_t)val;
        }
    }
    *po=p; return 0;
}

/* --------------------------------------------------------------- codec */
typedef struct {
    int h,w,tile,nt,nx,enable_me,cache_cap;
    int me_search,tile_level;   /* tunables: LOWLAT_ME_SEARCH / LOWLAT_DEFLATE_TILE */
    uint8_t *recon;
    uint8_t *sent_src;   /* last SOURCE content we transmitted per pixel (basis for dirty) */
    TCache cache;
    /* last-op stats */
    int st_bytes,st_ndirty,st_cachehits,st_lossy,st_text,st_haslossy;
    int last_gdy;
    int dbg_m[8];
    uint8_t *lossy_mask;   /* nt*nx, filled by decode+encode */
} CCodec;

CCodec *ccodec_new(int h,int w,int tile,int enable_me,int cache_cap){
    if(tile<=0)tile=32;
    CCodec*c=calloc(1,sizeof(CCodec)); if(!c)return NULL;
    c->h=h;c->w=w;c->tile=tile;c->nt=(h+tile-1)/tile;c->nx=(w+tile-1)/tile;
    c->enable_me=enable_me; c->cache_cap=cache_cap<0?0:cache_cap;
    { const char*e=getenv("LOWLAT_ME_SEARCH"); c->me_search=e?atoi(e):24; if(c->me_search<0)c->me_search=0; }
    { const char*e=getenv("LOWLAT_DEFLATE_TILE"); c->tile_level=e?atoi(e):9; if(c->tile_level<1||c->tile_level>9)c->tile_level=9; }
    c->recon=calloc((size_t)h*w*3,1);
    c->sent_src=calloc((size_t)h*w*3,1);
    c->lossy_mask=calloc((size_t)c->nt*c->nx,1);
    if(!c->recon||!c->sent_src||!c->lossy_mask){free(c->recon);free(c->sent_src);free(c->lossy_mask);free(c);return NULL;}
    cache_init(&c->cache,c->cache_cap,tile);
    return c;
}
void ccodec_free(CCodec*c){ if(!c)return; cache_free(&c->cache); free(c->recon); free(c->sent_src); free(c->lossy_mask); free(c); }
void ccodec_reset(CCodec*c){ memset(c->recon,0,(size_t)c->h*c->w*3); memset(c->sent_src,0,(size_t)c->h*c->w*3); cache_clear(&c->cache); }

void ccodec_get_stats(CCodec*c,int*bytes,int*ndirty,int*hits,int*lossy,int*text,int*haslossy){
    if(bytes)*bytes=c->st_bytes; if(ndirty)*ndirty=c->st_ndirty; if(hits)*hits=c->st_cachehits;
    if(lossy)*lossy=c->st_lossy; if(text)*text=c->st_text; if(haslossy)*haslossy=c->st_haslossy;
}
const uint8_t *ccodec_lossy_mask(CCodec*c){ return c->lossy_mask; }

/* motion search with neighbour-MV hint */
static void motion_search(int H,int W,const uint8_t*cur,const uint8_t*ref,int y0,int x0,int hh,int ww,
                          int search,int hy,int hx,int*bdy,int*bdx){
    long best=-1;int by=0,bx=0;
    if((hy||hx)&&y0+hy>=0&&x0+hx>=0&&y0+hy+hh<=H&&x0+hx+ww<=W){
        best=tile_sad(cur,ref,ww,hh,y0+hy,x0+hx,W);by=hy;bx=hx;
    }
    for(int dy=-search;dy<=search;dy+=2)
        for(int dx=-search;dx<=search;dx+=2){
            int yy=y0+dy,xx=x0+dx;
            if(yy<0||xx<0||yy+hh>H||xx+ww>W)continue;
            long s=tile_sad(cur,ref,ww,hh,yy,xx,W);
            if(best<0||s<best){best=s;by=dy;bx=dx;}
        }
    int rby=by,rbx=bx;long rb=best<0?0:best;
    for(int dy=by-2;dy<=by+2;dy++)
        for(int dx=bx-2;dx<=bx+2;dx++){
            int yy=y0+dy,xx=x0+dx;
            if(yy<0||xx<0||yy+hh>H||xx+ww>W)continue;
            long s=tile_sad(cur,ref,ww,hh,yy,xx,W);
            if(s<rb){rb=s;rby=dy;rbx=dx;}
        }
    *bdy=rby;*bdx=rbx;
}

/* --- tile payload encoders --- */
static size_t enc_varint(uint8_t*dst,const uint8_t*cur,const uint8_t*pred,int n){
    size_t p=0;
    for(int i=0;i<n;i++){
        int v=(int)cur[i]-(int)pred[i];
        uint32_t zz=(uint32_t)((v<<1)^(v>>31));
        while(zz>=0x80){dst[p++]=(uint8_t)((zz&0x7F)|0x80);zz>>=7;}
        dst[p++]=(uint8_t)zz;
    }
    return p;
}
static size_t enc_submask(uint8_t*dst,const uint8_t*cur,const uint8_t*pred,int hh,int ww){
    if(hh<=0||ww<=0)return 0;
    int np=hh*ww; size_t mb=(np+7)/8; memset(dst,0,mb); size_t p=mb;
    for(int i=0;i<np;i++){
        const uint8_t*cc=cur+i*3,*pp=pred+i*3;
        if(cc[0]==pp[0]&&cc[1]==pp[1]&&cc[2]==pp[2])continue;
        dst[i>>3]|=(uint8_t)(1u<<(i&7));
        for(int k=0;k<3;k++){int v=(int)cc[k]-(int)pp[k];uint32_t zz=(uint32_t)((v<<1)^(v>>31));
            while(zz>=0x80){dst[p++]=(uint8_t)((zz&0x7F)|0x80);zz>>=7;}dst[p++]=(uint8_t)zz;}
    }
    return p;
}
/* 2x downsample (box) -> dh*dw*3 bytes */
static size_t enc_down(uint8_t*dst,const uint8_t*src,int hh,int ww){
    int dw=(ww+1)/2, dh=(hh+1)/2; size_t p=0;
    for(int y=0;y<dh;y++)for(int x=0;x<dw;x++){
        int x0=x*2,y0=y*2,x1=x0+1<ww?x0+1:x0,y1=y0+1<hh?y0+1:y0;
        for(int c=0;c<3;c++){
            int s=0,n=0;
            for(int yy=y0;yy<=y1;yy++)for(int xx=x0;xx<=x1;xx++){s+=src[((size_t)yy*ww+xx)*3+c];n++;}
            dst[p++]=(uint8_t)(s/n);
        }
    }
    return p;
}
/* nearest upsample dh x dw -> hh x ww (deterministic) */
static void dec_up(uint8_t*dst,const uint8_t*src,int hh,int ww){
    int dw=(ww+1)/2, dh=(hh+1)/2;
    for(int y=0;y<hh;y++){
        int sy=(y*dh)/hh;
        for(int x=0;x<ww;x++){
            int sx=(x*dw)/ww;
            const uint8_t*s=src+(size_t)(sy*dw+sx)*3;
            uint8_t*d=dst+((size_t)y*ww+x)*3;
            d[0]=s[0];d[1]=s[1];d[2]=s[2];
        }
    }
}

/* ---- band (scrolling strip) Paeth coder ---- */
static size_t band_enc(uint8_t*dst,const uint8_t*frame,int W,int y0,int nrows,const uint8_t*uprow){
    size_t p=0; const uint8_t*up=uprow;
    for(int r=0;r<nrows;r++){
        const uint8_t*cur=frame+((size_t)(y0+r)*W)*3;
        for(int x=0;x<W*3;x++){
            int a=x>=3?cur[x-3]:0;
            int b=up?up[x]:0;
            int cc=(up&&x>=3)?up[x-3]:0;
            int pp=a+b-cc; int pa=iabs(pp-a),pb=iabs(pp-b),pc=iabs(pp-cc);
            int pr=(pa<=pb&&pa<=pc)?a:(pb<=pc?b:cc);
            dst[p++]=(uint8_t)((cur[x]-pr)&0xff);
        }
        up=cur;
    }
    return p;
}
static int band_dec(uint8_t*out,int W,int y0,int nrows,const uint8_t*raw,size_t rl,const uint8_t*uprow){
    size_t p=0; const uint8_t*up=uprow;
    for(int r=0;r<nrows;r++){
        uint8_t*cur=out+((size_t)(y0+r)*W)*3;
        for(int x=0;x<W*3;x++){
            int a=x>=3?cur[x-3]:0;
            int b=up?up[x]:0;
            int cc=(up&&x>=3)?up[x-3]:0;
            int pp=a+b-cc; int pa=iabs(pp-a),pb=iabs(pp-b),pc=iabs(pp-cc);
            int pr=(pa<=pb&&pa<=pc)?a:(pb<=pc?b:cc);
            if(p>=rl)return -1;
            cur[x]=(uint8_t)((pr+raw[p++])&0xff);
        }
        up=cur;
    }
    return 0;
}

/* ============================ ENCODE ============================ */
uint8_t *ccodec_encode(CCodec*c,const uint8_t*frame,int keyframe,int budget,int*outlen,int*ndirty_out){
    Buf out; buf_init(&out);
    buf_u8(&out,0); /* flags placeholder */

    if(keyframe){
        buf_bytes(&out,NULL,0);
        int stride=c->w*3; size_t flen=(size_t)c->h*(stride+1);
        uint8_t*filt=malloc(flen); uint8_t*prev=calloc(stride,1); uint8_t*scr=malloc(stride*2);
        if(!filt||!prev||!scr){free(filt);free(prev);free(scr);buf_free(&out);return NULL;}
        uint8_t*best=scr+stride; size_t fo=0;
        for(int y=0;y<c->h;y++){
            const uint8_t*row=frame+(size_t)y*stride; long bsd=-1;int bf=0;
            for(int f=0;f<=4;f++){
                long sad=0;
                for(int x=0;x<stride;x++){
                    int a=x>=3?row[x-3]:0,b=prev[x],cc=x>=3?prev[x-3]:0,pr;
                    switch(f){case 0:pr=0;break;case 1:pr=a;break;case 2:pr=b;break;case 3:pr=(a+b)>>1;break;
                        default:{int pp=a+b-cc;int pa=iabs(pp-a),pb=iabs(pp-b),pc=iabs(pp-cc);
                            pr=(pa<=pb&&pa<=pc)?a:(pb<=pc?b:cc);}}
                    int v=(uint8_t)((row[x]-pr)&0xff); scr[x]=(uint8_t)v; sad+=v<128?v:256-v;
                }
                if(bsd<0||sad<bsd){bsd=sad;bf=f;memcpy(best,scr,stride);}
            }
            filt[fo++]=(uint8_t)bf; memcpy(filt+fo,best,stride); fo+=stride;
            memcpy(prev,row,stride);
        }
        free(prev);free(scr);
        Buf body; buf_init(&body); int rc=deflate_into(filt,flen,&body,6); free(filt);
        if(rc){buf_free(&out);buf_free(&body);return NULL;}
        /* flags = keyframe */
        out.buf[0]=1;
        buf_u16(&out,(uint16_t)c->cache_cap);
        buf_u32(&out,(uint32_t)body.len); buf_bytes(&out,body.buf,body.len); buf_free(&body);
        memcpy(c->recon,frame,(size_t)c->h*c->w*3);
        memcpy(c->sent_src,frame,(size_t)c->h*c->w*3);
        c->last_gdy=0;
        /* prefill cache with all tiles (deterministic raster) */
        for(int ty=0;ty<c->nt;ty++)for(int tx=0;tx<c->nx;tx++){
            int y0=ty*c->tile,x0=tx*c->tile,y1=y0+c->tile>c->h?c->h:y0+c->tile,x1=x0+c->tile>c->w?c->w:x0+c->tile;
            int hh=y1-y0,ww=x1-x0;
            uint8_t*t=malloc((size_t)hh*ww*3);
            if(t){ for(int r=0;r<hh;r++)memcpy(t+(size_t)r*ww*3,frame+((size_t)(y0+r)*c->w+x0)*3,(size_t)ww*3);
                   cache_insert(&c->cache,t,hh,ww); free(t); }
        }
        c->st_bytes=(int)out.len; c->st_ndirty=c->nt*c->nx; c->st_cachehits=0;
        c->st_lossy=0; c->st_text=0; c->st_haslossy=0;
        memset(c->lossy_mask,0,(size_t)c->nt*c->nx);
        *outlen=(int)out.len; if(ndirty_out)*ndirty_out=c->nt*c->nx;
        if(out.err){buf_free(&out);return NULL;} return out.buf;
    }

    for(int z=0;z<8;z++)c->dbg_m[z]=0;
    int nt=c->nt,nx=c->nx,tile=c->tile, total=nt*nx;
    int gdy = c->enable_me ? global_dy(c->h,c->w,frame,c->recon,c->last_gdy) : 0;
    c->last_gdy = gdy;
    int use_global = (iabs(gdy)>=1);
    /* ---- BAND MODE: pure global scroll = shift + code only the new strip ---- */
    if(c->enable_me && iabs(gdy)>=1 && iabs(gdy)<=128){
        int nrows=iabs(gdy);
        int by0 = gdy>0 ? (c->h-gdy) : 0;
        size_t S=(size_t)c->h*c->w*3;
        uint8_t*sh=malloc(S);
        uint8_t*br=malloc((size_t)nrows*c->w*3+16);
        if(sh&&br){
            for(int y=0;y<c->h;y++){ int sy=y+gdy;
                if(sy>=0&&sy<c->h) memcpy(sh+(size_t)y*c->w*3,c->recon+(size_t)sy*c->w*3,(size_t)c->w*3);
                else memset(sh+(size_t)y*c->w*3,0,(size_t)c->w*3); }
            const uint8_t*up=(by0>0)? sh+(size_t)(by0-1)*c->w*3 : NULL;
            size_t bn=band_enc(br,frame,c->w,by0,nrows,up);
            band_dec(sh,c->w,by0,nrows,br,bn,up);   /* reconstruct into candidate */
            if(memcmp(sh,frame,S)==0){
                Buf body; buf_init(&body);
                if(deflate_into(br,bn,&body,c->tile_level)==0){
                    out.buf[0]=(uint8_t)(4|8);              /* global + band */
                    buf_u8(&out,(uint8_t)(int8_t)gdy);
                    buf_u32(&out,(uint32_t)body.len);
                    buf_bytes(&out,body.buf,body.len);
                    buf_free(&body);
                    memcpy(c->recon,frame,S); memcpy(c->sent_src,frame,S);
                    free(sh);free(br);
                    c->st_bytes=(int)out.len;c->st_ndirty=nrows;c->st_cachehits=0;
                    c->st_lossy=0;c->st_text=0;c->st_haslossy=0;
                    memset(c->lossy_mask,0,(size_t)total);
                    *outlen=(int)out.len; if(ndirty_out)*ndirty_out=nrows;
                    if(out.err){buf_free(&out);return NULL;} return out.buf;
                }
                buf_free(&body);
            }
        }
        free(sh);free(br);
        /* else fall through to normal tile path */
    }
    uint8_t*shadow=malloc((size_t)c->h*c->w*3);
    if(!shadow){buf_free(&out);return NULL;}
    if(use_global){
        /* shadow = recon shifted by gdy: dec predicts next frame from this */
        for(int y=0;y<c->h;y++){
            int sy=y+gdy;
            if(sy>=0&&sy<c->h) memcpy(shadow+(size_t)y*c->w*3, c->recon+(size_t)sy*c->w*3,(size_t)c->w*3);
            else memset(shadow+(size_t)y*c->w*3,0,(size_t)c->w*3);
        }
        memcpy(c->recon,shadow,(size_t)c->h*c->w*3);
    } else {
        memcpy(shadow,c->recon,(size_t)c->h*c->w*3);
    }
    uint8_t*dirty=calloc((size_t)total,1); if(!dirty){free(shadow);buf_free(&out);return NULL;}
    int nd=0;
    for(int ty=0;ty<nt;ty++)for(int tx=0;tx<nx;tx++){
        int y0=ty*tile,x0=tx*tile,y1=y0+tile>c->h?c->h:y0+tile,x1=x0+tile>c->w?c->w:x0+tile;
        int hh=y1-y0,ww=x1-x0,diff=0;
        const uint8_t*refb = use_global ? shadow : c->sent_src;
        for(int r=0;r<hh&&!diff;r++)
            if(memcmp(frame+((size_t)(y0+r)*c->w+x0)*3,refb+((size_t)(y0+r)*c->w+x0)*3,(size_t)ww*3))diff=1;
        if(diff){dirty[ty*nx+tx]=1;nd++;}
    }

    /* Single deterministic pass. budget<=0 -> fully lossless (no degradation). */
    Buf tsec; buf_init(&tsec);
    /* RLE bitmap into tsec first */
    { int i=0;
      while(i<total){int v=dirty[i],j=i+1;while(j<total&&dirty[j]==v)j++;int run=j-i;
        while(run>0){int r=run>65535?65535:run;buf_u8(&tsec,(uint8_t)v);buf_u16(&tsec,(uint16_t)r);run-=r;} i=j;} }

    int last_mvy=0,last_mvx=0;
    if(c->enable_me){ last_mvy=global_dy(c->h,c->w,frame,c->recon,last_mvy); }
    size_t maxn=(size_t)tile*tile*3;
    uint8_t*rA=malloc(maxn*2+16), *rC=malloc(maxn*2+(maxn/8)+32);
    uint8_t*pred=malloc(maxn);
    size_t down2n=(size_t)((tile+1)/2)*((tile+1)/2)*3;
    size_t down4n=(size_t)((tile+3)/4)*((tile+3)/4)*3;
    uint8_t*down=malloc(down2n+down4n+32);
    if(!rA||!rC||!pred||!down){
        free(dirty);free(shadow);free(rA);free(rC);free(pred);free(down);
        buf_free(&out);buf_free(&tsec);return NULL;
    }

    long used=(long)tsec.len + 8;      /* running raw-bytes estimate */
    long raw_budget = budget>0 ? (long)((double)budget*1.6) : -1; /* <=0: unlimited */
    int haslossy=0, nlossy=0, ntext=0, hits=0;
    memset(c->lossy_mask,0,(size_t)total);

    for(int ty=0;ty<nt;ty++)for(int tx=0;tx<nx;tx++){
        if(!dirty[ty*nx+tx])continue;
        int y0=ty*tile,x0=tx*tile,y1=y0+tile>c->h?c->h:y0+tile,x1=x0+tile>c->w?c->w:x0+tile;
        int hh=y1-y0,ww=x1-x0,n=hh*ww*3;
        uint8_t*ct=malloc(n);
        if(!ct){free(dirty);free(rA);free(rC);free(pred);free(shadow);free(down);buf_free(&out);buf_free(&tsec);return NULL;}
        for(int r=0;r<hh;r++)memcpy(ct+(size_t)r*ww*3,frame+((size_t)(y0+r)*c->w+x0)*3,(size_t)ww*3);

        int slot=cache_find(&c->cache,ct,hh,ww);
        if(slot>=0){
            hits++;
            buf_u8(&tsec,0);buf_u8(&tsec,0);buf_u8(&tsec,3);buf_u32(&tsec,2);buf_u16(&tsec,(uint16_t)slot);
            used+=9;
            /* recon = source (exact, matches cached pixels) */
            for(int r=0;r<hh;r++){
                memcpy(c->recon+((size_t)(y0+r)*c->w+x0)*3,ct+(size_t)r*ww*3,(size_t)ww*3);
                memcpy(shadow+((size_t)(y0+r)*c->w+x0)*3,ct+(size_t)r*ww*3,(size_t)ww*3);
                memcpy(c->sent_src+((size_t)(y0+r)*c->w+x0)*3,ct+(size_t)r*ww*3,(size_t)ww*3);
            }
            free(ct); continue;
        }

        int text=is_text_tile(ct,hh,ww); if(text)ntext++;

        /* prediction vs shadow (raster-updated) */
        int mvy=0,mvx=0;
        if(c->enable_me){ motion_search(c->h,c->w,ct,shadow,y0,x0,hh,ww,c->me_search,last_mvy,last_mvx,&mvy,&mvx);
                          last_mvy=mvy;last_mvx=mvx; }
        int py=y0+mvy,px=x0+mvx;
        if(py<0)py=0; if(px<0)px=0; if(py+hh>c->h)py=c->h-hh; if(px+ww>c->w)px=c->w-ww;
        for(int r=0;r<hh;r++)memcpy(pred+(size_t)r*ww*3,shadow+((size_t)(py+r)*c->w+px)*3,(size_t)ww*3);

        int exact = (memcmp(ct,pred,(size_t)n)==0);
        size_t la,lc,lb=n;
        const uint8_t*sel=NULL; size_t seln=0; int method=6; /* PRED default */
        if(exact){ method=6; seln=0; }
        else {
            la=enc_varint(rA,ct,pred,n);
            lc=enc_submask(rC,ct,pred,hh,ww);
            method=0;sel=rA;seln=la;
            if(lb<seln){method=1;sel=ct;seln=lb;}
            if(lc<seln){method=2;sel=rC;seln=lc;}
        }
        size_t lossless_bytes=seln;

        /* decide: keep lossless, or degrade (only non-text, only when over budget) */
        int chosen_lossy=0; int ch_method=method; size_t ch_len=seln; const uint8_t*ch_sel=sel;
        if(raw_budget>0 && used+(long)lossless_bytes>raw_budget && !text){
            /* delta magnitude (source vs decoder recon) for skip decision */
            long dsum=0;
            for(int r=0;r<hh;r++)for(int x=0;x<ww;x++)for(int k=0;k<3;k++){
                int a=ct[((size_t)r*ww+x)*3+k];
                int b=c->recon[((size_t)(y0+r)*c->w+x0+x)*3+k];
                dsum+=iabs(a-b);
            }
            double dmean=n?(double)dsum/n:0.0;
            if(dmean<=4.0){ ch_method=4; ch_len=0; chosen_lossy=2; }      /* SKIP: imperceptible */
            else {
                size_t dlen=enc_down(down,ct,hh,ww);         /* 2x */
                if(used > raw_budget*2){
                    size_t dlen4=enc_down4(down+ (size_t)((tile+1)/2)*((tile+1)/2)*3,ct,hh,ww); /* 4x into spare buf */
                    if(dlen4<dlen){ ch_method=7; ch_len=dlen4; chosen_lossy=1; }
                    else if(dlen<lossless_bytes){ ch_method=5; ch_len=dlen; chosen_lossy=1; }
                    else { ch_method=4; ch_len=0; chosen_lossy=2; }
                } else if(dlen<lossless_bytes){ ch_method=5; ch_len=dlen; chosen_lossy=1; }
            }
        }

        c->dbg_m[ch_method&7]++;
        buf_u8(&tsec,(uint8_t)(int8_t)mvy); buf_u8(&tsec,(uint8_t)(int8_t)mvx);
        buf_u8(&tsec,(uint8_t)ch_method); buf_u32(&tsec,(uint32_t)ch_len);
        if(ch_method==0||ch_method==1||ch_method==2) buf_bytes(&tsec,ch_sel,ch_len);
        else if(ch_method==5) buf_bytes(&tsec,down,ch_len);
        else if(ch_method==7) buf_bytes(&tsec,down+down2n,ch_len);
        used += 7 + (long)ch_len;

        if(chosen_lossy){ haslossy=1; nlossy++; c->lossy_mask[ty*nx+tx]=1; }
        /* remember what source we just represented, so lossy tiles aren't re-sent until they change */
        for(int r=0;r<hh;r++) memcpy(c->sent_src+((size_t)(y0+r)*c->w+x0)*3, ct+(size_t)r*ww*3,(size_t)ww*3);

        /* commit recon/shadow+cache per method */
        if(ch_method==4){ /* skip: recon unchanged */
        } else if(ch_method==5||ch_method==7){
            uint8_t*tmp=malloc(n);
            if(tmp){
                if(ch_method==5) dec_up(tmp,down,hh,ww); else dec_up4(tmp,down+down2n,hh,ww);
                for(int r=0;r<hh;r++){
                    memcpy(c->recon+((size_t)(y0+r)*c->w+x0)*3,tmp+(size_t)r*ww*3,(size_t)ww*3);
                    memcpy(shadow+((size_t)(y0+r)*c->w+x0)*3,tmp+(size_t)r*ww*3,(size_t)ww*3);
                }
                cache_insert(&c->cache,tmp,hh,ww); free(tmp);
            }
        } else { /* lossless 0/1/2/6 */
            for(int r=0;r<hh;r++){
                memcpy(c->recon+((size_t)(y0+r)*c->w+x0)*3,ct+(size_t)r*ww*3,(size_t)ww*3);
                memcpy(shadow+((size_t)(y0+r)*c->w+x0)*3,ct+(size_t)r*ww*3,(size_t)ww*3);
            }
            cache_insert(&c->cache,ct,hh,ww);
        }
        free(ct);
    }
    free(dirty);free(rA);free(rC);free(pred);free(shadow);free(down);

    Buf body; buf_init(&body);
    if(deflate_into(tsec.buf,tsec.len,&body,6)){ buf_free(&tsec);buf_free(&out);buf_free(&body);return NULL; }
    buf_free(&tsec);

    out.buf[0]=(uint8_t)((haslossy?2:0) | (use_global?4:0)); /* inter, bit1=lossy, bit2=global */
    if(use_global) buf_u8(&out,(uint8_t)(int8_t)gdy);
    buf_u16(&out,(uint16_t)nd);
    buf_u32(&out,(uint32_t)body.len);
    buf_bytes(&out,body.buf,body.len); buf_free(&body);

    /* persistent per-tile approximation mask: recon != source */
    { int plossy=0;
      for(int ty=0;ty<nt;ty++)for(int tx=0;tx<nx;tx++){
        int y0=ty*tile,x0=tx*tile,y1=y0+tile>c->h?c->h:y0+tile,x1=x0+tile>c->w?c->w:x0+tile;
        int hh=y1-y0,ww=x1-x0,diff=0;
        for(int r=0;r<hh&&!diff;r++)
            if(memcmp(frame+((size_t)(y0+r)*c->w+x0)*3,c->recon+((size_t)(y0+r)*c->w+x0)*3,(size_t)ww*3))diff=1;
        c->lossy_mask[ty*nx+tx]=(uint8_t)diff;
        if(diff)plossy++;
      }
      nlossy=plossy;
    }
    c->st_bytes=(int)out.len; c->st_ndirty=nd; c->st_cachehits=hits;
    c->st_lossy=nlossy; c->st_text=ntext; c->st_haslossy=haslossy;
    *outlen=(int)out.len; if(ndirty_out)*ndirty_out=nd;
    if(out.err){buf_free(&out);return NULL;} return out.buf;
}

/* ============================ DECODE ============================ */
uint8_t *ccodec_decode(CCodec*c,const uint8_t*data,int datalen,int*outlen){
    size_t p=0; if(datalen<1)return NULL;
    int flags=data[p++];
    int keyframe=flags&1, lossy=flags&2;
    uint8_t*frame=malloc((size_t)c->h*c->w*3); if(!frame)return NULL;
    memset(c->lossy_mask,0,(size_t)c->nt*c->nx);

    if(keyframe){
        if(p+2>(size_t)datalen){free(frame);return NULL;}
        int cap=data[p]|(data[p+1]<<8); p+=2;
        if(p+4>(size_t)datalen){free(frame);return NULL;}
        uint32_t blen=data[p]|(data[p+1]<<8)|(data[p+2]<<16)|((uint32_t)data[p+3]<<24);p+=4;
        if(p+blen>(size_t)datalen){free(frame);return NULL;}
        int stride=c->w*3; size_t flen=(size_t)c->h*(stride+1);
        uint8_t*filt=malloc(flen); uLongf dl=flen;
        if(!filt||uncompress(filt,&dl,data+p,blen)!=Z_OK){free(frame);free(filt);return NULL;}
        uint8_t*prev=calloc(stride,1); size_t fo=0;
        for(int y=0;y<c->h;y++){
            uint8_t*row=frame+(size_t)y*stride; int f=filt[fo++]; const uint8_t*in=filt+fo; fo+=stride;
            for(int x=0;x<stride;x++){
                int a=x>=3?row[x-3]:0,b=prev[x],cc=x>=3?prev[x-3]:0,pr;
                switch(f){case 0:pr=0;break;case 1:pr=a;break;case 2:pr=b;break;case 3:pr=(a+b)>>1;break;
                    default:{int pp=a+b-cc;int pa=iabs(pp-a),pb=iabs(pp-b),pc=iabs(pp-cc);pr=(pa<=pb&&pa<=pc)?a:(pb<=pc?b:cc);}}
                row[x]=(uint8_t)((pr+in[x])&0xff);
            }
            memcpy(prev,row,stride);
        }
        free(prev);free(filt);memcpy(c->recon,frame,(size_t)c->h*c->w*3);
        /* reset + prefill cache (matches encoder) */
        cache_clear(&c->cache);
        for(int ty=0;ty<c->nt;ty++)for(int tx=0;tx<c->nx;tx++){
            int y0=ty*c->tile,x0=tx*c->tile,y1=y0+c->tile>c->h?c->h:y0+c->tile,x1=x0+c->tile>c->w?c->w:x0+c->tile;
            int hh=y1-y0,ww=x1-x0; uint8_t*t=malloc((size_t)hh*ww*3);
            if(t){ for(int r=0;r<hh;r++)memcpy(t+(size_t)r*ww*3,frame+((size_t)(y0+r)*c->w+x0)*3,(size_t)ww*3);
                   cache_insert(&c->cache,t,hh,ww); free(t); }
        }
        (void)cap;
        c->st_bytes=datalen; c->st_lossy=0; c->st_haslossy=0;
        *outlen=c->h*c->w*3; return frame;
    }

    if(flags&8){ /* BAND mode */
        if(p+1>(size_t)datalen){free(frame);return NULL;}
        int bgdy=(int8_t)data[p++];
        if(p+4>(size_t)datalen){free(frame);return NULL;}
        uint32_t blen=data[p]|(data[p+1]<<8)|(data[p+2]<<16)|((uint32_t)data[p+3]<<24);p+=4;
        if(p+blen>(size_t)datalen){free(frame);return NULL;}
        int nrows=iabs(bgdy), by0=bgdy>0?(c->h-bgdy):0;
        size_t rawb=(size_t)nrows*c->w*3+16;
        uint8_t*br=malloc(rawb); if(!br){free(frame);return NULL;}
        uLongf rl=rawb;
        if(uncompress(br,&rl,data+p,blen)!=Z_OK){free(br);free(frame);return NULL;}
        for(int y=0;y<c->h;y++){ int sy=y+bgdy;
            if(sy>=0&&sy<c->h) memcpy(frame+(size_t)y*c->w*3,c->recon+(size_t)sy*c->w*3,(size_t)c->w*3);
            else memset(frame+(size_t)y*c->w*3,0,(size_t)c->w*3); }
        const uint8_t*up=(by0>0)? frame+(size_t)(by0-1)*c->w*3 : NULL;
        if(band_dec(frame,c->w,by0,nrows,br,rl,up)!=0){free(br);free(frame);return NULL;}
        free(br);
        memcpy(c->recon,frame,(size_t)c->h*c->w*3);
        memset(c->lossy_mask,0,(size_t)c->nt*c->nx);
        c->st_bytes=datalen;c->st_lossy=0;c->st_haslossy=0;
        *outlen=c->h*c->w*3; return frame;
    }
    int gdy=0, use_global=(flags&4)?1:0;
    if(use_global){ if(p+1>(size_t)datalen){free(frame);return NULL;} gdy=(int8_t)data[p++]; }
    if(p+2>(size_t)datalen){free(frame);return NULL;}
    uint16_t nd=data[p]|(data[p+1]<<8);p+=2;
    int total=c->nt*c->nx;
    if(p+4>(size_t)datalen){free(frame);return NULL;}
    uint32_t tlen=data[p]|(data[p+1]<<8)|(data[p+2]<<16)|((uint32_t)data[p+3]<<24);p+=4;
    if(p+tlen>(size_t)datalen){free(frame);return NULL;}
    size_t rawcap=(size_t)total*(size_t)c->tile*c->tile*3/8 + (size_t)nd*((size_t)c->tile*c->tile*3*2+32) + (size_t)total*16 + 64;
    uint8_t*raw=malloc(rawcap);
    if(!raw){free(frame);return NULL;}
    uLongf rl=rawcap;
    if(uncompress(raw,&rl,data+p,tlen)!=Z_OK){free(raw);free(frame);return NULL;}
    size_t rp=0;

    uint8_t*dirty=calloc(total,1); if(!dirty){free(raw);free(frame);return NULL;}
    { int idx=0;
      while(idx<total){ if(rp+3>rl){free(dirty);free(raw);free(frame);return NULL;}
        int v=raw[rp],run=raw[rp+1]|(raw[rp+2]<<8);rp+=3;int end=idx+run;if(end>total)end=total;
        if(v)memset(dirty+idx,1,end-idx); idx=end; } }
    if(use_global){
        for(int y=0;y<c->h;y++){
            int sy=y+gdy;
            if(sy>=0&&sy<c->h) memcpy(frame+(size_t)y*c->w*3,c->recon+(size_t)sy*c->w*3,(size_t)c->w*3);
            else memset(frame+(size_t)y*c->w*3,0,(size_t)c->w*3);
        }
        memcpy(c->recon,frame,(size_t)c->h*c->w*3);
    } else {
        memcpy(frame,c->recon,(size_t)c->h*c->w*3);
    }
    uint8_t*pred=malloc((size_t)c->tile*c->tile*3);
    uint8_t*tile_out=malloc((size_t)c->tile*c->tile*3);
    uint8_t*dnbuf=malloc((size_t)((c->tile+1)/2)*((c->tile+1)/2)*3+16);
    if(!pred||!tile_out||!dnbuf){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}

    int nlossy=0;
    for(int ty=0;ty<c->nt;ty++)for(int tx=0;tx<c->nx;tx++){
        if(!dirty[ty*c->nx+tx])continue;
        int y0=ty*c->tile,x0=tx*c->tile,y1=y0+c->tile>c->h?c->h:y0+c->tile,x1=x0+c->tile>c->w?c->w:x0+c->tile;
        int hh=y1-y0,ww=x1-x0,n=hh*ww*3;
        if(rp+7>rl){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
        int mvy=(int8_t)raw[rp++],mvx=(int8_t)raw[rp++];
        int method=raw[rp++];
        uint32_t plen=raw[rp]|(raw[rp+1]<<8)|(raw[rp+2]<<16)|((uint32_t)raw[rp+3]<<24);rp+=4;
        if(rp+plen>rl){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
        const uint8_t*payload=raw+rp;

        if(method==3){
            if(plen!=2){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
            int slot=payload[0]|(payload[1]<<8);
            if(!c->cache.e||slot<0||slot>=c->cache.cap||!c->cache.e[slot].used||
               c->cache.e[slot].h!=hh||c->cache.e[slot].w!=ww){
                free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
            memcpy(tile_out,c->cache.e[slot].pix,(size_t)n);
            rp+=plen;
            for(int r=0;r<hh;r++){
                memcpy(frame+((size_t)(y0+r)*c->w+x0)*3,tile_out+(size_t)r*ww*3,(size_t)ww*3);
                memcpy(c->recon+((size_t)(y0+r)*c->w+x0)*3,tile_out+(size_t)r*ww*3,(size_t)ww*3);
            }
            continue;
        }

        int py=y0+mvy,px=x0+mvx;
        if(py<0)py=0; if(px<0)px=0; if(py+hh>c->h)py=c->h-hh; if(px+ww>c->w)px=c->w-ww;
        for(int r=0;r<hh;r++)memcpy(pred+(size_t)r*ww*3,fp(frame,c->w,py+r,px),(size_t)ww*3);

        int rc=0;
        if(method==4){ /* skip */
            rp+=plen; c->lossy_mask[ty*c->nx+tx]=1; nlossy++;
            continue;
        } else if(method==5){ /* down 2x */
            if(plen!=(uint32_t)(((ww+1)/2)*((hh+1)/2)*3)){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
            dec_up(tile_out,payload,hh,ww); rc=0; nlossy++;
        } else if(method==7){ /* down 4x */
            if(plen!=(uint32_t)(((ww+3)/4)*((hh+3)/4)*3)){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}
            dec_up4(tile_out,payload,hh,ww); rc=0; nlossy++;
        } else if(method==6){ memcpy(tile_out,pred,(size_t)n); rc=0; }
        else if(method==0){ size_t o=0; rc=dec_varint(payload,plen,&o,pred,n,tile_out); }
        else if(method==2){ size_t o=0; rc=dec_submask(payload,plen,&o,pred,hh,ww,tile_out); }
        else { memcpy(tile_out,payload,n<plen?n:plen); }
        rp+=plen;
        if(rc){free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);free(frame);return NULL;}

        for(int r=0;r<hh;r++){
            memcpy(frame+((size_t)(y0+r)*c->w+x0)*3,tile_out+(size_t)r*ww*3,(size_t)ww*3);
            memcpy(c->recon+((size_t)(y0+r)*c->w+x0)*3,tile_out+(size_t)r*ww*3,(size_t)ww*3);
        }
        cache_insert(&c->cache,tile_out,hh,ww);
    }
    free(pred);free(tile_out);free(dnbuf);free(dirty);free(raw);
    c->st_bytes=datalen; c->st_lossy=nlossy; c->st_haslossy=lossy?1:0;
    *outlen=c->h*c->w*3; return frame;
}

void ccodec_dbg_methods(CCodec*c,int*out8){ if(!c||!out8)return; for(int i=0;i<8;i++)out8[i]=c->dbg_m[i]; }
