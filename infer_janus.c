/*
 * infer_janus.c — Janus RRPRAM inference on notorch
 *
 * Loads janus_char_leo_d12.bin (26.2M params, char-level)
 * Architecture: V=256, E=384, H=4, D=96, 12 blocks, M=768
 * 3-way gated attention: QKV + RRPRAM + Janus echo
 *
 * Build:
 *   Mac:   cc -O2 -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *          -framework Accelerate infer_janus.c notorch.c -lm -o infer_janus_nt
 *   Linux: cc -O2 -DUSE_BLAS infer_janus.c notorch.c -lm -lopenblas -o infer_janus_nt
 *
 * Run:
 *   ./infer_janus_nt ~/Downloads/janus-weights/janus_char_leo_d12.bin
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define V    256
#define E    384
#define H    4
#define D    96
#define BLK  12
#define FFN  768
#define MT   256

// ── Weight layout ────────────────────────────────────────────────────────────

typedef struct {
    float *tok_emb, *pos_emb;
    struct {
        float *rms1, *wq, *wk, *wv, *wr, *wvr, *wj, *gate, *wo;
        float *rms2, *wg, *wu, *wd;
    } b[BLK];
    float *rms_f, *head;
} Weights;

static int param_count(void) {
    int s = V*E + MT*E;
    for (int i = 0; i < BLK; i++)
        s += E + H*E*MT + H*3 + E*E*6 + E + FFN*E + FFN*E + E*FFN;
    s += E + V*E;
    return s;
}

static void assign_weights(Weights *w, float *p) {
    w->tok_emb = p; p += V*E;
    w->pos_emb = p; p += MT*E;
    for (int i = 0; i < BLK; i++) {
        w->b[i].rms1 = p; p += E;
        w->b[i].wr = p;   p += H*E*MT;
        w->b[i].gate = p; p += H*3;
        w->b[i].wq = p;   p += E*E;
        w->b[i].wk = p;   p += E*E;
        w->b[i].wv = p;   p += E*E;
        w->b[i].wvr = p;  p += E*E;
        w->b[i].wj = p;   p += E*E;
        w->b[i].wo = p;   p += E*E;
        w->b[i].rms2 = p; p += E;
        w->b[i].wg = p;   p += FFN*E;
        w->b[i].wu = p;   p += FFN*E;
        w->b[i].wd = p;   p += E*FFN;
    }
    w->rms_f = p; p += E;
    w->head = p;
}

// ── Matmul helpers ───────────────────────────────────────────────────────────
// PyTorch Linear stores W as [out, in]. F.linear(x, W) = x @ W^T

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

// C[m,n] = A[m,k] @ B^T[n,k] where B stored as [n,k]
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

// C[m,n] = A[m,k] @ B[k,n]
static void mm(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
#endif
}

static void rmsnorm(float *out, const float *x, const float *w, int T, int dim) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) ss += x[t*dim+i] * x[t*dim+i];
        float inv = 1.0f / sqrtf(ss/dim + 1e-5f);
        for (int i = 0; i < dim; i++) out[t*dim+i] = w[i] * x[t*dim+i] * inv;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x/(1+expf(-x)) : 0; }

// ── Forward pass ─────────────���───────────────────────────────────────────────

static void forward(Weights *w, int *tok, int T, float *logits) {
    float *x = (float*)calloc(T*E, sizeof(float));
    float *rn = (float*)calloc(T*E, sizeof(float));
    float sc = 1.0f / sqrtf((float)D);

    // Embed
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            x[t*E+e] = w->tok_emb[tok[t]*E+e] + w->pos_emb[t*E+e];

    float *cat = (float*)calloc(T*E, sizeof(float));
    float *ao = (float*)calloc(T*E, sizeof(float));
    float *r1 = (float*)calloc(T*E, sizeof(float));
    float *mg = (float*)calloc(T*FFN, sizeof(float));
    float *mu = (float*)calloc(T*FFN, sizeof(float));
    float *mo = (float*)calloc(T*E, sizeof(float));

    for (int bl = 0; bl < BLK; bl++) {
        rmsnorm(rn, x, w->b[bl].rms1, T, E);

        float *qa = (float*)calloc(T*E, sizeof(float));
        float *ka = (float*)calloc(T*E, sizeof(float));
        float *va = (float*)calloc(T*E, sizeof(float));
        float *vra = (float*)calloc(T*E, sizeof(float));
        mm_t(qa, rn, w->b[bl].wq, T, E, E);
        mm_t(ka, rn, w->b[bl].wk, T, E, E);
        mm_t(va, rn, w->b[bl].wv, T, E, E);
        mm_t(vra, rn, w->b[bl].wvr, T, E, E);

        // Janus echo
        float *echo = (float*)calloc(T*E, sizeof(float));
        mm_t(echo, rn, w->b[bl].wj, T, E, E);
        float *eback = (float*)calloc(T*E, sizeof(float));
        mm(eback, echo, w->b[bl].wj, T, E, E);

        // Janus scores
        float *jsc = (float*)calloc(T, sizeof(float));
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int e = 0; e < E; e++) s += rn[t*E+e] * eback[t*E+e];
            jsc[t] = s / sqrtf((float)E);
        }
        float *jat = (float*)calloc(T*T, sizeof(float));
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++)
                jat[i*T+j] = (j > i) ? -1e9f : jsc[i] * jsc[j];
            softmax(jat + i*T, T);
        }

        // Gates: [H, 3]
        float gs[H][3];
        for (int h = 0; h < H; h++) {
            gs[h][0] = w->b[bl].gate[h*3+0];
            gs[h][1] = w->b[bl].gate[h*3+1];
            gs[h][2] = w->b[bl].gate[h*3+2];
            softmax(gs[h], 3);
        }

        memset(cat, 0, T*E*sizeof(float));
        float *at = (float*)calloc(T*T, sizeof(float));
        float *ho = (float*)calloc(T*D, sizeof(float));

        for (int h = 0; h < H; h++) {
            float *q = (float*)calloc(T*D, sizeof(float));
            float *k = (float*)calloc(T*D, sizeof(float));
            float *v = (float*)calloc(T*D, sizeof(float));
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    q[t*D+d] = qa[t*E + h*D + d];
                    k[t*D+d] = ka[t*E + h*D + d];
                    v[t*D+d] = va[t*E + h*D + d];
                }

            // QKV attention (causal)
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) { at[i*T+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < D; d++) s += q[i*D+d] * k[j*D+d];
                    at[i*T+j] = s * sc;
                }
                softmax(at + i*T, T);
            }
            mm(ho, at, v, T, T, D);

            // RRPRAM
            float *wr_h = w->b[bl].wr + h*E*MT;
            float rrp_sc[MT];
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[j*E+e] * wr_h[e*MT+j];
                rrp_sc[j] = s * sc;
            }
            float *ra = (float*)calloc(T*T, sizeof(float));
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++)
                    ra[i*T+j] = (j > i) ? -1e9f : rrp_sc[j];
                softmax(ra + i*T, T);
            }
            float *rv = (float*)calloc(T*D, sizeof(float));
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    rv[t*D+d] = vra[t*E + h*D + d];
            float *ro = (float*)calloc(T*D, sizeof(float));
            mm(ro, ra, rv, T, T, D);

            // Janus values per head
            float *jv = (float*)calloc(T*D, sizeof(float));
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    jv[t*D+d] = echo[t*E + h*D + d];
            float *jo = (float*)calloc(T*D, sizeof(float));
            mm(jo, jat, jv, T, T, D);

            // Blend
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = gs[h][0]*ho[t*D+d]
                                        + gs[h][1]*ro[t*D+d]
                                        + gs[h][2]*jo[t*D+d];
            free(q); free(k); free(v); free(ra); free(rv); free(ro); free(jv); free(jo);
        }

        mm_t(ao, cat, w->b[bl].wo, T, E, E);
        for (int i = 0; i < T*E; i++) r1[i] = x[i] + ao[i];

        // MLP: SiLU-gated
        rmsnorm(rn, r1, w->b[bl].rms2, T, E);
        mm_t(mg, rn, w->b[bl].wg, T, E, FFN);
        mm_t(mu, rn, w->b[bl].wu, T, E, FFN);
        for (int i = 0; i < T*FFN; i++) mg[i] = siluf(mg[i]) * mu[i];
        mm_t(mo, mg, w->b[bl].wd, T, FFN, E);
        for (int i = 0; i < T*E; i++) x[i] = r1[i] + mo[i];

        free(qa); free(ka); free(va); free(vra);
        free(echo); free(eback); free(jsc); free(jat);
        free(at); free(ho);
    }

    rmsnorm(rn, x, w->rms_f, T, E);
    mm_t(logits, rn, w->head, T, E, V);

    free(x); free(rn); free(cat); free(ao); free(r1);
    free(mg); free(mu); free(mo);
}

// ── Sampling ─────────────────────────────────────────────────────────────────

static int sample_top_p(float *logits, int n, float temp, float top_p) {
    // Temperature
    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);

    // Top-p (nucleus) sampling
    // Simple: sort by probability, accumulate until top_p
    int indices[V];
    for (int i = 0; i < n; i++) indices[i] = i;
    // Selection sort (V=256 is small)
    for (int i = 0; i < n - 1; i++) {
        int mx = i;
        for (int j = i + 1; j < n; j++)
            if (logits[indices[j]] > logits[indices[mx]]) mx = j;
        if (mx != i) { int tmp = indices[i]; indices[i] = indices[mx]; indices[mx] = tmp; }
    }

    float cum = 0;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cum += logits[indices[i]];
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    // Renormalize and sample
    float sum = 0;
    for (int i = 0; i < cutoff; i++) sum += logits[indices[i]];
    float r = (float)rand() / (float)RAND_MAX * sum;
    cum = 0;
    for (int i = 0; i < cutoff; i++) {
        cum += logits[indices[i]];
        if (cum >= r) return indices[i];
    }
    return indices[0];
}

// ── Timer ────────────────────────────────────��───────────────────────────────

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Main ───────────────────────────────────────────────────────────���─────────

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("notorch inference — Janus RRPRAM (char-level)\n");
        printf("usage: %s weights.bin [prompt] [max_tokens] [temp]\n", argv[0]);
        return 1;
    }

    // Load weights
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 1; }
    int np;
    fread(&np, 4, 1, f);
    int expected = param_count();
    if (np != expected) {
        printf("param count mismatch: file=%d expected=%d\n", np, expected);
        fclose(f);
        return 1;
    }
    float *data = (float*)malloc(np * sizeof(float));
    fread(data, sizeof(float), np, f);
    fclose(f);

    Weights w;
    assign_weights(&w, data);
    printf("notorch: loaded %d params (%.1f MB)\n", np, np * 4.0f / 1048576.0f);

    // Verify loss on training data
    FILE *df = fopen("/Users/ataeff/Downloads/janus-weights/leo_train.txt", "rb");
    if (!df) df = fopen("leo_train.txt", "rb");
    if (df) {
        fseek(df, 0, SEEK_END);
        long dsz = ftell(df);
        fseek(df, 0, SEEK_SET);
        unsigned char *dt = (unsigned char*)malloc(dsz);
        fread(dt, 1, dsz, df);
        fclose(df);

        int T = 64;
        int tok[64], tgt[64];
        float loss_sum = 0;
        int n_windows = 20;
        double t0 = now_ms();

        for (int wi = 0; wi < n_windows; wi++) {
            int off = (int)((dsz - T - 1) * wi / n_windows);
            for (int t = 0; t < T; t++) { tok[t] = dt[off+t]; tgt[t] = dt[off+t+1]; }
            float *lg = (float*)calloc(T * V, sizeof(float));
            forward(&w, tok, T, lg);
            float wloss = 0;
            for (int t = 0; t < T; t++) {
                softmax(lg + t*V, V);
                float p = lg[t*V + tgt[t]];
                if (p < 1e-10f) p = 1e-10f;
                wloss -= logf(p);
            }
            wloss /= T;
            loss_sum += wloss;
            if (wi < 3) printf("  window %d: loss=%.4f\n", wi, wloss);
            free(lg);
        }
        double elapsed = now_ms() - t0;
        printf("avg loss: %.4f (%d windows, %.0f ms)\n",
               loss_sum / n_windows, n_windows, elapsed);
        free(dt);
    }

    // Generate
    const char *prompt = argc > 2 ? argv[2] : "Q: who are you?\nA: ";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 200;
    float temp = argc > 4 ? atof(argv[4]) : 0.8f;

    int ctx[MT * 2];
    int len = 0;
    for (int i = 0; prompt[i] && len < MT; i++)
        ctx[len++] = (unsigned char)prompt[i];

    printf("\n── generation (temp=%.2f, top_p=0.9) ──\n%s", temp, prompt);

    double gen_start = now_ms();
    for (int step = 0; step < max_tokens; step++) {
        int T = len < 64 ? len : 64;
        int *tok = ctx + (len > 64 ? len - 64 : 0);
        float *lg = (float*)calloc(T * V, sizeof(float));
        forward(&w, tok, T, lg);
        float *last = lg + (T - 1) * V;
        int next = sample_top_p(last, V, temp, 0.9f);
        printf("%c", (char)next);
        fflush(stdout);
        if (len < MT * 2) ctx[len++] = next;
        free(lg);
    }
    double gen_elapsed = now_ms() - gen_start;
    printf("\n── %.0f ms, %.1f tok/s ──\n", gen_elapsed, max_tokens * 1000.0 / gen_elapsed);

    free(data);
    return 0;
}
