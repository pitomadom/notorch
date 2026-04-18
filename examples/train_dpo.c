/*
 * train_dpo.c — Direct Preference Optimization on notorch
 *
 * DPO loss: L = -log σ(β · (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
 *
 * Two forward passes per step: chosen and rejected.
 * Reference model is frozen copy of initial weights.
 * Cross-entropy ∝ -log π, so DPO reuses the same backward path.
 *
 * Dataset: JSONL with {"chosen": [...], "rejected": [...]} token ID arrays.
 *
 * Build: make train_dpo
 * Run:   ./train_dpo <data.jsonl> <base_weights.bin> [steps] [lr] [beta]
 *
 * By Arianna Method. DOI: 10.5281/zenodo.19638451
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* ── Model config (MiniMind-64M defaults, override via #define) ── */
#ifndef DIM
#define DIM       512
#endif
#ifndef NLAYERS
#define NLAYERS   8
#endif
#ifndef NHEADS
#define NHEADS    8
#endif
#ifndef NKV_HEADS
#define NKV_HEADS 4
#endif
#define HEAD_DIM  (DIM / NHEADS)
#define KV_DIM    (NKV_HEADS * HEAD_DIM)
#ifndef HIDDEN
#define HIDDEN    (DIM * 4)
#endif
#ifndef CTX
#define CTX       256
#endif
#ifndef VOCAB
#define VOCAB     6400
#endif

#define LOG_EVERY   10
#define CKPT_EVERY  100

/* ── Model ── */

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *qnorm, *knorm, *wo;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static int model_n_tensors(void) { return 1 + NLAYERS * 11 + 2; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1; p[i++]=m->L[l].wq; p[i++]=m->L[l].wk;
        p[i++]=m->L[l].wv; p[i++]=m->L[l].qnorm; p[i++]=m->L[l].knorm;
        p[i++]=m->L[l].wo; p[i++]=m->L[l].rms2;
        p[i++]=m->L[l].w_gate; p[i++]=m->L[l].w_up; p[i++]=m->L[l].w_down;
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static Model* model_new(void) {
    Model* m = calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->wte, VOCAB, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq, DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(KV_DIM, DIM); nt_tensor_xavier(m->L[l].wk, DIM, KV_DIM);
        m->L[l].wv = nt_tensor_new2d(KV_DIM, DIM); nt_tensor_xavier(m->L[l].wv, DIM, KV_DIM);
        m->L[l].qnorm = nt_tensor_new(HEAD_DIM); nt_tensor_fill(m->L[l].qnorm, 1.0f);
        m->L[l].knorm = nt_tensor_new(HEAD_DIM); nt_tensor_fill(m->L[l].knorm, 1.0f);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo, DIM, DIM);
        m->L[l].rms2 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_gate, DIM, HIDDEN);
        m->L[l].w_up = nt_tensor_new2d(HIDDEN, DIM); nt_tensor_xavier(m->L[l].w_up, DIM, HIDDEN);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN); nt_tensor_xavier(m->L[l].w_down, HIDDEN, DIM);
    }
    m->rms_f = nt_tensor_new(DIM); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(VOCAB, DIM); nt_tensor_xavier(m->head, DIM, VOCAB);
    return m;
}

static Model* model_clone(Model* src) {
    /* Deep copy for frozen reference model */
    Model* m = calloc(1, sizeof(Model));
    nt_tensor** sp = model_param_array(src);
    int n = model_n_tensors();
    nt_tensor** dp = malloc(n * sizeof(nt_tensor*));
    for (int i = 0; i < n; i++) {
        dp[i] = nt_tensor_new(sp[i]->len);
        dp[i]->ndim = sp[i]->ndim;
        memcpy(dp[i]->shape, sp[i]->shape, sizeof(sp[i]->shape));
        memcpy(dp[i]->stride, sp[i]->stride, sizeof(sp[i]->stride));
        memcpy(dp[i]->data, sp[i]->data, sp[i]->len * sizeof(float));
    }
    /* Assign to struct */
    int j = 0;
    m->wte = dp[j++];
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = dp[j++]; m->L[l].wq = dp[j++]; m->L[l].wk = dp[j++];
        m->L[l].wv = dp[j++]; m->L[l].qnorm = dp[j++]; m->L[l].knorm = dp[j++];
        m->L[l].wo = dp[j++]; m->L[l].rms2 = dp[j++];
        m->L[l].w_gate = dp[j++]; m->L[l].w_up = dp[j++]; m->L[l].w_down = dp[j++];
    }
    m->rms_f = dp[j++]; m->head = dp[j++];
    free(sp); free(dp);
    return m;
}

static void model_free(Model* m) {
    nt_tensor_free(m->wte);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].rms1); nt_tensor_free(m->L[l].rms2);
        nt_tensor_free(m->L[l].wq); nt_tensor_free(m->L[l].wk);
        nt_tensor_free(m->L[l].wv); nt_tensor_free(m->L[l].qnorm);
        nt_tensor_free(m->L[l].knorm); nt_tensor_free(m->L[l].wo);
        nt_tensor_free(m->L[l].w_gate); nt_tensor_free(m->L[l].w_up);
        nt_tensor_free(m->L[l].w_down);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head); free(m);
}

/* ── Tape-based forward (returns cross-entropy loss index on tape) ── */

static int forward_on_tape(Model* m, int* tok, int* tgt, int T) {
    nt_tensor** params = model_param_array(m);
    int n = model_n_tensors();
    int* pi_ids = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        pi_ids[i] = nt_tape_param(params[i]);
    nt_tape_no_decay(pi_ids[0]); /* embedding — no weight decay */

    nt_tensor* tok_t = nt_tensor_new(T);
    nt_tensor* tgt_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) { tok_t->data[t] = (float)tok[t]; tgt_t->data[t] = (float)tgt[t]; }
    int tok_idx = nt_tape_record(tok_t, 0, -1, -1, 0);
    int tgt_idx = nt_tape_record(tgt_t, 0, -1, -1, 0);

    int pi = 0;
    int h = nt_seq_embedding(pi_ids[pi++], -1, tok_idx, T, DIM);

    for (int l = 0; l < NLAYERS; l++) {
        int rms1=pi_ids[pi++], wq=pi_ids[pi++], wk=pi_ids[pi++], wv=pi_ids[pi++];
        int qn=pi_ids[pi++], kn=pi_ids[pi++], wo=pi_ids[pi++], rms2=pi_ids[pi++];
        int wg=pi_ids[pi++], wu=pi_ids[pi++], wd=pi_ids[pi++];

        int xn = nt_seq_rmsnorm(h, rms1, T, DIM);
        int q = nt_seq_linear(wq, xn, T);
        int k = nt_seq_linear(wk, xn, T);
        int v = nt_seq_linear(wv, xn, T);
        q = nt_seq_rmsnorm(q, qn, T * NHEADS, HEAD_DIM);
        k = nt_seq_rmsnorm(k, kn, T * NKV_HEADS, HEAD_DIM);
        q = nt_rope(q, T, HEAD_DIM);
        k = nt_rope(k, T, HEAD_DIM);
        int attn = nt_gqa_causal_attention(q, k, v, T, HEAD_DIM, NHEADS, NKV_HEADS);
        h = nt_add(h, nt_seq_linear(wo, attn, T));

        xn = nt_seq_rmsnorm(h, rms2, T, DIM);
        int gate = nt_silu(nt_seq_linear(wg, xn, T));
        int up = nt_seq_linear(wu, xn, T);
        h = nt_add(h, nt_seq_linear(wd, nt_mul(gate, up), T));
    }

    int rmsf = pi_ids[pi++], head_p = pi_ids[pi++];
    int hf = nt_seq_rmsnorm(h, rmsf, T, DIM);
    int logits = nt_seq_linear(head_p, hf, T);
    int loss_idx = nt_seq_cross_entropy(logits, tgt_idx, T, VOCAB);

    free(params); free(pi_ids);
    return loss_idx;
}

/* ── Read cross-entropy loss value from tape ── */

static float read_tape_loss(int loss_idx) {
    nt_tape* tape = nt_tape_get();
    nt_tensor* lt = tape->entries[loss_idx].output;
    return lt->data[0];
}

/* ── DPO training step ──
 *
 * The DPO gradient through cross-entropy is:
 *   ∂L_dpo/∂θ = -β·σ(-β·Δ) · (∂CE_chosen/∂θ - ∂CE_rejected/∂θ)
 *
 * where Δ = T·(CE_rejected_pi - CE_chosen_pi) - T·(CE_rejected_ref - CE_chosen_ref)
 *
 * We compute this by:
 *   1. Ref forward (no tape) on chosen & rejected → get ref losses
 *   2. Policy forward (tape) on chosen → backward → save grads
 *   3. Policy forward (tape) on rejected → backward → accumulate scaled grads
 *   4. Apply DPO-weighted update
 */

static void dpo_step(Model* policy, Model* ref, int* chosen, int* rejected,
                     int* chosen_tgt, int* rejected_tgt, int T_c, int T_r,
                     float beta, float lr) {
    int n = model_n_tensors();
    nt_tensor** pp = model_param_array(policy);

    /* 1. Reference model forward (no tape, no grad) */
    nt_train_mode(0);

    nt_tape_start();
    int ref_c_idx = forward_on_tape(ref, chosen, chosen_tgt, T_c);
    float ref_ce_chosen = read_tape_loss(ref_c_idx);
    nt_tape_clear();

    nt_tape_start();
    int ref_r_idx = forward_on_tape(ref, rejected, rejected_tgt, T_r);
    float ref_ce_rejected = read_tape_loss(ref_r_idx);
    nt_tape_clear();

    /* 2. Policy forward on chosen (with tape + grad) */
    nt_train_mode(1);

    nt_tape_start();
    int pi_c_idx = forward_on_tape(policy, chosen, chosen_tgt, T_c);
    float pi_ce_chosen = read_tape_loss(pi_c_idx);
    nt_tape_backward(pi_c_idx);

    /* Save chosen gradients */
    float** grad_chosen = malloc(n * sizeof(float*));
    nt_tape* tape = nt_tape_get();
    for (int i = 0; i < n; i++) {
        grad_chosen[i] = malloc(pp[i]->len * sizeof(float));
        /* Gradients are accumulated in tape's adam states; read from tape entries */
        /* Actually, after backward the grads are in the tape param entries */
        int pi_tape = i; /* tape param index */
        if (tape->entries[pi_tape].grad)
            memcpy(grad_chosen[i], tape->entries[pi_tape].grad->data, pp[i]->len * sizeof(float));
        else
            memset(grad_chosen[i], 0, pp[i]->len * sizeof(float));
    }
    nt_tape_clear();

    /* 3. Policy forward on rejected (with tape + grad) */
    nt_tape_start();
    int pi_r_idx = forward_on_tape(policy, rejected, rejected_tgt, T_r);
    float pi_ce_rejected = read_tape_loss(pi_r_idx);
    nt_tape_backward(pi_r_idx);

    /* 4. Compute DPO coefficient and apply weighted gradients */
    /* Δ = T_c·(CE_rejected_pi - CE_chosen_pi) - T_r is wrong, use log probs directly:
     * log π(y|x) = -T * CE, so:
     * log_ratio_pi = -T_c*pi_ce_chosen + T_r*pi_ce_rejected (higher prob chosen = lower CE)
     * Actually: log π(chosen) = -T_c * CE_chosen, log π(rejected) = -T_r * CE_rejected
     * pi_logratios = log π(chosen) - log π(rejected) = -T_c*CE_chosen + T_r*CE_rejected
     * ref_logratios = -T_c*CE_ref_chosen + T_r*CE_ref_rejected
     */
    float pi_logratios = -T_c * pi_ce_chosen + T_r * pi_ce_rejected;
    float ref_logratios = -T_c * ref_ce_chosen + T_r * ref_ce_rejected;
    float delta = pi_logratios - ref_logratios;
    float sigmoid_neg = 1.0f / (1.0f + expf(beta * delta)); /* σ(-β·Δ) */
    float dpo_coeff = beta * sigmoid_neg;

    /* Combined gradient: dpo_coeff * (grad_chosen - grad_rejected) */
    /* grad_chosen = ∂CE_chosen/∂θ, we want -∂log_pi_chosen/∂θ direction */
    /* DPO wants to increase log_pi_chosen and decrease log_pi_rejected */
    /* ∂L/∂θ = dpo_coeff * (T_c * ∂CE_chosen/∂θ - T_r * ∂CE_rejected/∂θ) */
    /* But CE backward already divides by T, so ∂CE/∂θ is normalized */
    tape = nt_tape_get();
    for (int i = 0; i < n; i++) {
        if (tape->entries[i].grad) {
            float* g_rej = tape->entries[i].grad->data;
            for (int j = 0; j < pp[i]->len; j++) {
                /* Final grad = dpo_coeff * (grad_chosen - grad_rejected) */
                g_rej[j] = dpo_coeff * (grad_chosen[i][j] - g_rej[j]);
            }
        }
        free(grad_chosen[i]);
    }
    free(grad_chosen);

    /* Update with Chuck */
    float dpo_loss = -logf(1.0f / (1.0f + expf(-beta * delta)) + 1e-10f);
    nt_tape_clip_grads(1.0f);
    nt_tape_chuck_step(lr, dpo_loss);
    nt_tape_clear();

    free(pp);
}

/* ── Main ── */

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data.jsonl> <base_weights.bin> [steps] [lr] [beta]\n", argv[0]);
        return 1;
    }
    const char* data_path = argv[1];
    const char* weights_path = argv[2];
    int max_steps = argc > 3 ? atoi(argv[3]) : 1000;
    float lr = argc > 4 ? atof(argv[4]) : 4e-5f;
    float beta = argc > 5 ? atof(argv[5]) : 0.15f;

    printf("═══════════════════════════════════════════════════\n");
    printf("  notorch — DPO training\n");
    printf("  Direct Preference Optimization (Rafailov 2023)\n");
    printf("  DIM=%d LAYERS=%d HEADS=%d VOCAB=%d CTX=%d\n", DIM, NLAYERS, NHEADS, VOCAB, CTX);
    printf("═══════════════════════════════════════════════════\n");

    /* Load base model */
    Model* policy = model_new();
    int n = model_n_tensors();
    nt_tensor** pp = model_param_array(policy);
    int loaded_n = 0;
    nt_tensor** loaded = nt_load(weights_path, &loaded_n);
    if (!loaded || loaded_n != n) {
        fprintf(stderr, "Failed to load weights from %s (got %d, need %d)\n", weights_path, loaded_n, n);
        return 1;
    }
    for (int i = 0; i < n; i++)
        memcpy(pp[i]->data, loaded[i]->data, pp[i]->len * sizeof(float));
    printf("  loaded policy from %s\n", weights_path);
    free(pp);

    /* Clone for frozen reference */
    Model* ref = model_clone(policy);
    printf("  cloned reference model (frozen)\n");
    printf("  β=%.2f  lr=%.1e  steps=%d\n", beta, lr, max_steps);

    /* TODO: load DPO pairs from data.jsonl */
    /* For now, this is the training infrastructure. */
    /* Dataset loading to be implemented per dataset format. */

    printf("\n  DPO training infrastructure ready.\n");
    printf("  Awaiting dataset integration.\n");

    model_free(policy);
    model_free(ref);
    return 0;
}
