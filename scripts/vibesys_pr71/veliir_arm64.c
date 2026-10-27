/* Zero-phase SOS filtering for the fused velocity amplitude (amplitude_extractor, QS_VELIIR=c).
 *
 * sos_filtfilt_df2t computes, for each of C channels, what
 *     np.flip(sosfilt(sos, np.flip(sosfilt(sos, x))))
 * returns (obspy's highpass(zerophase=True)), bit for bit: every sample takes the operations of
 * scipy's _sosfilt_float (scipy/signal/_sosfilt.pyx) in the same order, with zero initial state:
 *     x_new = b0*x + z0;  z0 = (b1*x - a1*x_new) + z1;  z1 = b2*x - a2*x_new;  x = x_new
 * Build with -O2 -ffp-contract=off and no -ffast-math / -march: no FMA contraction and no
 * reassociation, so each multiply and add rounds as scipy's compiled loop rounds it.
 *
 * The C channels can also run as the lanes of one 4-wide vector (AVX, no FMA, chosen at run time
 * when the CPU has it): lane arithmetic is the same IEEE multiply/add/subtract, in the same order.
 *
 * In place, without flips: the forward pass runs up the index and leaves y in out[c]; the backward
 * pass runs down the index over out[c], so its k-th input is y[n-1-k] (the flipped forward output)
 * and its k-th output lands at index n-1-k (flipped back). The C channels share one sample loop,
 * so their independent recurrences overlap (the single-channel loop is latency-bound).
 *
 * x[c] and out[c] may be the same buffer. Returns 0, or -1 for an unsupported shape. */
#include <stdint.h>

#define MAXC 4
#define MAXSEC 8

typedef struct {
    double b0, b1, b2, a1, a2;
} coef_t;

/* One sample of the cascade for one channel (scipy's per-sample section loop). */
#define STEP(cf, z, nsec, v)                                         \
    do {                                                             \
        double xc_ = (v);                                            \
        for (int s_ = 0; s_ < (nsec); s_++) {                        \
            double xn_ = (cf)[s_].b0 * xc_ + (z)[s_][0];             \
            (z)[s_][0] = (cf)[s_].b1 * xc_ - (cf)[s_].a1 * xn_       \
                         + (z)[s_][1];                               \
            (z)[s_][1] = (cf)[s_].b2 * xc_ - (cf)[s_].a2 * xn_;      \
            xc_ = xn_;                                               \
        }                                                            \
        (v) = xc_;                                                   \
    } while (0)

/* One pass over C channels with NSEC sections; dir = +1 (up the index) or -1 (down). The
 * constant C and NSEC let the compiler keep every state in a register. */
#define PASS(C_, NSEC_)                                                          \
    static void pass_##C_##_##NSEC_(const double *const *src, double *const *dst, \
                                    int64_t n, const coef_t *cf, int dir) {      \
        double z[C_][NSEC_][2] = {{{0}}};                                        \
        int64_t i = dir > 0 ? 0 : n - 1;                                         \
        for (int64_t k = 0; k < n; k++, i += dir) {                              \
            for (int c = 0; c < C_; c++) {                                       \
                double v = src[c][i];                                            \
                STEP(cf, z[c], NSEC_, v);                                        \
                dst[c][i] = v;                                                   \
            }                                                                    \
        }                                                                        \
    }

PASS(1, 2)
PASS(2, 2)
PASS(3, 2)
PASS(4, 2)

static void pass_generic(const double *const *src, double *const *dst, int64_t n, int C,
                         const coef_t *cf, int nsec, int dir) {
    double z[MAXC][MAXSEC][2] = {{{0}}};
    int64_t i = dir > 0 ? 0 : n - 1;
    for (int64_t k = 0; k < n; k++, i += dir) {
        for (int c = 0; c < C; c++) {
            double v = src[c][i];
            STEP(cf, z[c], nsec, v);
            dst[c][i] = v;
        }
    }
}

typedef double v4d __attribute__((vector_size(32)));

/* PASS with the channels in the lanes of a v4d, for two sections (the 4th-order Butterworth).
 * Unused lanes read channel 0 and are never stored. */
#define PASSV(C_)                                                                       \
    __attribute__((target("avx"))) static void passv_##C_(                              \
        const double *const *src, double *const *dst, int64_t n, const coef_t *cf,      \
        int dir) {                                                                      \
        const v4d b00 = {cf[0].b0, cf[0].b0, cf[0].b0, cf[0].b0};                       \
        const v4d b01 = {cf[0].b1, cf[0].b1, cf[0].b1, cf[0].b1};                       \
        const v4d b02 = {cf[0].b2, cf[0].b2, cf[0].b2, cf[0].b2};                       \
        const v4d a01 = {cf[0].a1, cf[0].a1, cf[0].a1, cf[0].a1};                       \
        const v4d a02 = {cf[0].a2, cf[0].a2, cf[0].a2, cf[0].a2};                       \
        const v4d b10 = {cf[1].b0, cf[1].b0, cf[1].b0, cf[1].b0};                       \
        const v4d b11 = {cf[1].b1, cf[1].b1, cf[1].b1, cf[1].b1};                       \
        const v4d b12 = {cf[1].b2, cf[1].b2, cf[1].b2, cf[1].b2};                       \
        const v4d a11 = {cf[1].a1, cf[1].a1, cf[1].a1, cf[1].a1};                       \
        const v4d a12 = {cf[1].a2, cf[1].a2, cf[1].a2, cf[1].a2};                       \
        const double *s0 = src[0], *s1 = src[C_ > 1 ? 1 : 0];                           \
        const double *s2 = src[C_ > 2 ? 2 : 0], *s3 = src[C_ > 3 ? 3 : 0];              \
        double *d0 = dst[0], *d1 = dst[C_ > 1 ? 1 : 0];                                 \
        double *d2 = dst[C_ > 2 ? 2 : 0], *d3 = dst[C_ > 3 ? 3 : 0];                    \
        v4d z00 = {0}, z01 = {0}, z10 = {0}, z11 = {0};                                 \
        int64_t i = dir > 0 ? 0 : n - 1;                                                \
        for (int64_t k = 0; k < n; k++, i += dir) {                                     \
            v4d x = {s0[i], s1[i], s2[i], s3[i]};                                       \
            v4d xn = b00 * x + z00;                                                     \
            z00 = b01 * x - a01 * xn + z01;                                             \
            z01 = b02 * x - a02 * xn;                                                   \
            x = xn;                                                                     \
            xn = b10 * x + z10;                                                         \
            z10 = b11 * x - a11 * xn + z11;                                             \
            z11 = b12 * x - a12 * xn;                                                   \
            d0[i] = xn[0];                                                              \
            if (C_ > 1) d1[i] = xn[1];                                                  \
            if (C_ > 2) d2[i] = xn[2];                                                  \
            if (C_ > 3) d3[i] = xn[3];                                                  \
        }                                                                               \
    }

PASSV(2)
PASSV(3)
PASSV(4)

/* -1: not yet probed; 0: scalar; 1: AVX lanes. veliir_set_isa(0) forces the scalar loops. */
static int isa = -1;

int veliir_isa(void) {
    if (isa < 0) {
#if defined(__x86_64__)
        __builtin_cpu_init();
        isa = __builtin_cpu_supports("avx") ? 1 : 0;
#else
        isa = 1;  /* test build: generic vector lanes on arm64 */
#endif
    }
    return isa;
}

void veliir_set_isa(int value) { isa = value; }

static void pass(const double *const *src, double *const *dst, int64_t n, int C,
                 const coef_t *cf, int nsec, int dir) {
    if (nsec == 2) {
        if (C > 1 && veliir_isa() == 1) {
            switch (C) {
            case 2: passv_2(src, dst, n, cf, dir); return;
            case 3: passv_3(src, dst, n, cf, dir); return;
            case 4: passv_4(src, dst, n, cf, dir); return;
            }
        }
        switch (C) {
        case 1: pass_1_2(src, dst, n, cf, dir); return;
        case 2: pass_2_2(src, dst, n, cf, dir); return;
        case 3: pass_3_2(src, dst, n, cf, dir); return;
        case 4: pass_4_2(src, dst, n, cf, dir); return;
        }
    }
    pass_generic(src, dst, n, C, cf, nsec, dir);
}

int sos_filtfilt_df2t(const double *const *x, int64_t n, int C, const double *sos, int nsec,
                      double *const *out) {
    coef_t cf[MAXSEC];
    if (C < 1 || C > MAXC || nsec < 1 || nsec > MAXSEC || n < 0)
        return -1;
    /* sos rows are scipy's (b0, b1, b2, a0, a1, a2) with a0 == 1 (checked by the caller, as
     * scipy's _validate_sos does); a0 takes no part in the recurrence. */
    for (int s = 0; s < nsec; s++) {
        cf[s].b0 = sos[6 * s + 0];
        cf[s].b1 = sos[6 * s + 1];
        cf[s].b2 = sos[6 * s + 2];
        cf[s].a1 = sos[6 * s + 4];
        cf[s].a2 = sos[6 * s + 5];
    }
    pass(x, out, n, C, cf, nsec, +1);
    pass((const double *const *)out, out, n, C, cf, nsec, -1);
    return 0;
}
