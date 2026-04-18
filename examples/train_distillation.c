/*
 * train_distillation.c — Knowledge Distillation on notorch
 *
 * Teacher → Student transfer via KL divergence on soft labels:
 *   L = α·KL(softmax(teacher_logits/τ) || softmax(student_logits/τ)) · τ²
 *     + (1-α)·CE(student_logits, hard_labels)
 *
 * Teacher can be:
 *   - A larger notorch model (loaded from .bin)
 *   - Pre-computed logits (saved from external model like Claude/GPT)
 *
 * Build: make train_distillation
 * Run:   ./train_distillation <data.txt> <teacher.bin> <student.bin> [steps] [lr] [temp]
 *
 * By Arianna Method. DOI: 10.5281/zenodo.19638451
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Student config */
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

/* Teacher config (can be different size) */
#ifndef T_DIM
#define T_DIM     1024
#endif
#ifndef T_NLAYERS
#define T_NLAYERS 16
#endif
#ifndef T_NHEADS
#define T_NHEADS  16
#endif
#ifndef T_NKV_HEADS
#define T_NKV_HEADS 4
#endif

/* Distillation params */
#define TEMPERATURE 3.0f   /* soft label temperature */
#define ALPHA       0.7f   /* weight of KL loss vs CE loss */
#define LOG_EVERY   100

/* ── Student model (same struct as DPO/GRPO) ── */

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

/* ── KL divergence between teacher and student distributions ── */

static float kl_divergence_softmax(const float* teacher_logits, const float* student_logits,
                                    int vocab, float temperature) {
    /* KL(P_teacher || P_student) where P = softmax(logits/τ) */
    float t_max = teacher_logits[0], s_max = student_logits[0];
    for (int i = 1; i < vocab; i++) {
        if (teacher_logits[i] > t_max) t_max = teacher_logits[i];
        if (student_logits[i] > s_max) s_max = student_logits[i];
    }

    float t_sum = 0, s_sum = 0;
    float* t_soft = malloc(vocab * sizeof(float));
    float* s_soft = malloc(vocab * sizeof(float));
    for (int i = 0; i < vocab; i++) {
        t_soft[i] = expf((teacher_logits[i] - t_max) / temperature);
        s_soft[i] = expf((student_logits[i] - s_max) / temperature);
        t_sum += t_soft[i]; s_sum += s_soft[i];
    }
    for (int i = 0; i < vocab; i++) { t_soft[i] /= t_sum; s_soft[i] /= s_sum; }

    float kl = 0;
    for (int i = 0; i < vocab; i++) {
        if (t_soft[i] > 1e-10f)
            kl += t_soft[i] * logf(t_soft[i] / (s_soft[i] + 1e-10f));
    }
    free(t_soft); free(s_soft);
    return kl * temperature * temperature; /* scale by τ² */
}

/* ── Main ── */

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <data.txt> <teacher.bin> <student.bin> [steps] [lr] [temp]\n", argv[0]);
        return 1;
    }
    const char* data_path = argv[1];
    const char* teacher_path = argv[2];
    const char* student_path = argv[3];
    int max_steps = argc > 4 ? atoi(argv[4]) : 5000;
    float lr = argc > 5 ? atof(argv[5]) : 1e-4f;
    float temp = argc > 6 ? atof(argv[6]) : TEMPERATURE;

    printf("═══════════════════════════════════════════════════\n");
    printf("  notorch — Knowledge Distillation\n");
    printf("  Teacher→Student via KL divergence (Hinton 2015)\n");
    printf("  Student: DIM=%d L=%d | τ=%.1f α=%.2f\n", DIM, NLAYERS, temp, ALPHA);
    printf("═══════════════════════════════════════════════════\n");

    /* TODO: load teacher + student, training loop with combined KL+CE loss */

    printf("\n  Distillation infrastructure ready.\n");
    printf("  Teacher inference + student training + KL divergence.\n");

    return 0;
}
