#include <math.h>

static inline double interp64(
    const double *arr,
    int n,
    double q,
    double q_min,
    double q_max,
    double inv_dq
) {
    if (q <= q_min) {
        return arr[0];
    }
    if (q >= q_max) {
        return arr[n - 1];
    }
    double pos = (q - q_min) * inv_dq;
    int idx = (int)pos;
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= n - 1) {
        idx = n - 2;
    }
    double frac = pos - (double)idx;
    return (1.0 - frac) * arr[idx] + frac * arr[idx + 1];
}

static inline float interp32(
    const float *arr,
    int n,
    float q,
    float q_min,
    float q_max,
    float inv_dq
) {
    if (q <= q_min) {
        return arr[0];
    }
    if (q >= q_max) {
        return arr[n - 1];
    }
    float pos = (q - q_min) * inv_dq;
    int idx = (int)pos;
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= n - 1) {
        idx = n - 2;
    }
    float frac = pos - (float)idx;
    return (1.0f - frac) * arr[idx] + frac * arr[idx + 1];
}

static inline double clip_grad64(double grad) {
    if (!isfinite(grad)) {
        return 0.0;
    }
    if (grad > 1000.0) {
        return 1000.0;
    }
    if (grad < -1000.0) {
        return -1000.0;
    }
    return grad;
}

static inline float clip_grad32(float grad) {
    if (!isfinite(grad)) {
        return 0.0f;
    }
    if (grad > 1000.0f) {
        return 1000.0f;
    }
    if (grad < -1000.0f) {
        return -1000.0f;
    }
    return grad;
}

static inline double clamp_q64(double q, double q_min, double q_max) {
    if (!isfinite(q)) {
        return q_min;
    }
    if (q < q_min) {
        return q_min;
    }
    if (q > q_max) {
        return q_max;
    }
    return q;
}

static inline float clamp_q32(float q, float q_min, float q_max) {
    if (!isfinite(q)) {
        return q_min;
    }
    if (q < q_min) {
        return q_min;
    }
    if (q > q_max) {
        return q_max;
    }
    return q;
}

static inline void full_loss_grad64(
    const double *b_scaled,
    const double *c_scaled,
    const double *bq_scaled,
    const double *cq_scaled,
    int n_profile,
    int order,
    double q_min,
    double q_max,
    double inv_dq,
    double target_norm_sq,
    double q,
    double *rel_error,
    double *grad_a,
    double *grad_q,
    double *h,
    double amplitude
) {
    *h = exp(q);
    double h_order = exp((double)order * q);
    double h_2order = h_order * h_order;
    double b = interp64(b_scaled, n_profile, q, q_min, q_max, inv_dq);
    double c = interp64(c_scaled, n_profile, q, q_min, q_max, inv_dq);
    double bq = interp64(bq_scaled, n_profile, q, q_min, q_max, inv_dq);
    double cq = interp64(cq_scaled, n_profile, q, q_min, q_max, inv_dq);

    if (b <= 0.0 || !isfinite(b) || !isfinite(c)) {
        *rel_error = 1.0;
        *grad_a = 0.0;
        *grad_q = 0.0;
        return;
    }

    double feature_norm_sq = h_2order * b;
    double feature_target_ip = h_order * c;
    double d_feature_norm_sq_q = h_2order * bq;
    double d_feature_target_ip_q = h_order * cq;
    double loss = 0.5 * (
        amplitude * amplitude * feature_norm_sq
        - 2.0 * amplitude * feature_target_ip
        + target_norm_sq
    );
    *grad_a = amplitude * feature_norm_sq - feature_target_ip;
    *grad_q = 0.5 * amplitude * amplitude * d_feature_norm_sq_q
        - amplitude * d_feature_target_ip_q;
    double rel_arg = 2.0 * loss / target_norm_sq;
    *rel_error = rel_arg > 0.0 ? sqrt(rel_arg) : 0.0;
}

static inline void full_loss_grad32(
    const float *b_scaled,
    const float *c_scaled,
    const float *bq_scaled,
    const float *cq_scaled,
    int n_profile,
    int order,
    float q_min,
    float q_max,
    float inv_dq,
    float target_norm_sq,
    float q,
    float *rel_error,
    float *grad_a,
    float *grad_q,
    float *h,
    float amplitude
) {
    *h = expf(q);
    float h_order = expf((float)order * q);
    float h_2order = h_order * h_order;
    float b = interp32(b_scaled, n_profile, q, q_min, q_max, inv_dq);
    float c = interp32(c_scaled, n_profile, q, q_min, q_max, inv_dq);
    float bq = interp32(bq_scaled, n_profile, q, q_min, q_max, inv_dq);
    float cq = interp32(cq_scaled, n_profile, q, q_min, q_max, inv_dq);

    if (b <= 0.0f || !isfinite(b) || !isfinite(c)) {
        *rel_error = 1.0f;
        *grad_a = 0.0f;
        *grad_q = 0.0f;
        return;
    }

    float feature_norm_sq = h_2order * b;
    float feature_target_ip = h_order * c;
    float d_feature_norm_sq_q = h_2order * bq;
    float d_feature_target_ip_q = h_order * cq;
    float loss = 0.5f * (
        amplitude * amplitude * feature_norm_sq
        - 2.0f * amplitude * feature_target_ip
        + target_norm_sq
    );
    *grad_a = amplitude * feature_norm_sq - feature_target_ip;
    *grad_q = 0.5f * amplitude * amplitude * d_feature_norm_sq_q
        - amplitude * d_feature_target_ip_q;
    float rel_arg = 2.0f * loss / target_norm_sq;
    *rel_error = rel_arg > 0.0f ? sqrtf(rel_arg) : 0.0f;
}

int run_diff_optimizer64(
    const double *b_scaled,
    const double *c_scaled,
    const double *bq_scaled,
    const double *cq_scaled,
    int n_profile,
    int order,
    double coeff_norm_sq,
    double offset_sq_sum,
    double q_init,
    double q_min,
    double q_max,
    double inv_dq,
    double target_norm_sq,
    long long max_iter,
    double lr,
    double amplitude_init_scale,
    const long long *records,
    int n_records,
    double *iteration,
    double *lambda_hist,
    double *profiled_relative_error,
    double *separation,
    double *amplitude_hist,
    double *weight_norm,
    double *theta_norm,
    double *grad_norm,
    double *q_hist
) {
    double q = q_init;
    double b0 = interp64(b_scaled, n_profile, q, q_min, q_max, inv_dq);
    double c0 = interp64(c_scaled, n_profile, q, q_min, q_max, inv_dq);
    double amplitude = (b0 > 0.0 && isfinite(b0) && isfinite(c0))
        ? amplitude_init_scale * exp(-(double)order * q) * c0 / b0
        : 0.0;
    double m_a = 0.0;
    double v_a = 0.0;
    double m_q = 0.0;
    double v_q = 0.0;
    double beta1_pow = 1.0;
    double beta2_pow = 1.0;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    int rec_idx = 0;

    for (long long it = 0; it <= max_iter; ++it) {
        double rel_error = 0.0;
        double grad_a = 0.0;
        double grad_q = 0.0;
        double h = 0.0;
        full_loss_grad64(
            b_scaled, c_scaled, bq_scaled, cq_scaled, n_profile, order,
            q_min, q_max, inv_dq, target_norm_sq, q,
            &rel_error, &grad_a, &grad_q, &h, amplitude
        );
        grad_a = clip_grad64(grad_a);
        grad_q = clip_grad64(grad_q);
        if (rec_idx < n_records && it == records[rec_idx]) {
            double wn = sqrt(coeff_norm_sq) * fabs(amplitude);
            double tn = sqrt(coeff_norm_sq * amplitude * amplitude + (double)(order + 1) + offset_sq_sum * h * h);
            iteration[rec_idx] = (double)it;
            lambda_hist[rec_idx] = 0.0;
            profiled_relative_error[rec_idx] = rel_error;
            separation[rec_idx] = h;
            amplitude_hist[rec_idx] = amplitude;
            weight_norm[rec_idx] = wn;
            theta_norm[rec_idx] = tn;
            grad_norm[rec_idx] = sqrt(grad_a * grad_a + grad_q * grad_q);
            q_hist[rec_idx] = q;
            rec_idx += 1;
        }

        if (it == max_iter) {
            break;
        }

        beta1_pow *= beta1;
        beta2_pow *= beta2;
        m_a = beta1 * m_a + (1.0 - beta1) * grad_a;
        v_a = beta2 * v_a + (1.0 - beta2) * grad_a * grad_a;
        m_q = beta1 * m_q + (1.0 - beta1) * grad_q;
        v_q = beta2 * v_q + (1.0 - beta2) * grad_q * grad_q;

        double mhat_a = m_a / (1.0 - beta1_pow);
        double vhat_a = v_a / (1.0 - beta2_pow);
        double mhat_q = m_q / (1.0 - beta1_pow);
        double vhat_q = v_q / (1.0 - beta2_pow);
        double step_a = lr * mhat_a / (sqrt(vhat_a) + eps);
        double step_q = lr * mhat_q / (sqrt(vhat_q) + eps);
        if (!isfinite(step_a)) {
            step_a = 0.0;
        }
        if (!isfinite(step_q)) {
            step_q = 0.0;
        }
        amplitude -= step_a;
        if (!isfinite(amplitude)) {
            amplitude = 0.0;
        }
        q = clamp_q64(q - step_q, q_min, q_max);
    }

    return rec_idx;
}

int run_diff_optimizer32(
    const float *b_scaled,
    const float *c_scaled,
    const float *bq_scaled,
    const float *cq_scaled,
    int n_profile,
    int order,
    float coeff_norm_sq,
    float offset_sq_sum,
    float q_init,
    float q_min,
    float q_max,
    float inv_dq,
    float target_norm_sq,
    long long max_iter,
    float lr,
    float amplitude_init_scale,
    const long long *records,
    int n_records,
    double *iteration,
    double *lambda_hist,
    double *profiled_relative_error,
    double *separation,
    double *amplitude_hist,
    double *weight_norm,
    double *theta_norm,
    double *grad_norm,
    double *q_hist
) {
    float q = q_init;
    float b0 = interp32(b_scaled, n_profile, q, q_min, q_max, inv_dq);
    float c0 = interp32(c_scaled, n_profile, q, q_min, q_max, inv_dq);
    float amplitude = (b0 > 0.0f && isfinite(b0) && isfinite(c0))
        ? amplitude_init_scale * expf(-(float)order * q) * c0 / b0
        : 0.0f;
    float m_a = 0.0f;
    float v_a = 0.0f;
    float m_q = 0.0f;
    float v_q = 0.0f;
    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    int rec_idx = 0;

    for (long long it = 0; it <= max_iter; ++it) {
        float rel_error = 0.0f;
        float grad_a = 0.0f;
        float grad_q = 0.0f;
        float h = 0.0f;
        full_loss_grad32(
            b_scaled, c_scaled, bq_scaled, cq_scaled, n_profile, order,
            q_min, q_max, inv_dq, target_norm_sq, q,
            &rel_error, &grad_a, &grad_q, &h, amplitude
        );
        grad_a = clip_grad32(grad_a);
        grad_q = clip_grad32(grad_q);
        if (rec_idx < n_records && it == records[rec_idx]) {
            double amplitude_d = (double)amplitude;
            double h_d = (double)h;
            double wn = sqrt((double)coeff_norm_sq) * fabs(amplitude_d);
            double tn = sqrt(
                (double)coeff_norm_sq * amplitude_d * amplitude_d
                + (double)(order + 1)
                + (double)offset_sq_sum * h_d * h_d
            );
            iteration[rec_idx] = (double)it;
            lambda_hist[rec_idx] = 0.0;
            profiled_relative_error[rec_idx] = (double)rel_error;
            separation[rec_idx] = (double)h;
            amplitude_hist[rec_idx] = amplitude_d;
            weight_norm[rec_idx] = wn;
            theta_norm[rec_idx] = tn;
            grad_norm[rec_idx] = (double)sqrtf(grad_a * grad_a + grad_q * grad_q);
            q_hist[rec_idx] = (double)q;
            rec_idx += 1;
        }

        if (it == max_iter) {
            break;
        }

        beta1_pow *= beta1;
        beta2_pow *= beta2;
        m_a = beta1 * m_a + (1.0f - beta1) * grad_a;
        v_a = beta2 * v_a + (1.0f - beta2) * grad_a * grad_a;
        m_q = beta1 * m_q + (1.0f - beta1) * grad_q;
        v_q = beta2 * v_q + (1.0f - beta2) * grad_q * grad_q;

        float mhat_a = m_a / (1.0f - beta1_pow);
        float vhat_a = v_a / (1.0f - beta2_pow);
        float mhat_q = m_q / (1.0f - beta1_pow);
        float vhat_q = v_q / (1.0f - beta2_pow);
        float step_a = lr * mhat_a / (sqrtf(vhat_a) + eps);
        float step_q = lr * mhat_q / (sqrtf(vhat_q) + eps);
        if (!isfinite(step_a)) {
            step_a = 0.0f;
        }
        if (!isfinite(step_q)) {
            step_q = 0.0f;
        }
        amplitude -= step_a;
        if (!isfinite(amplitude)) {
            amplitude = 0.0f;
        }
        q = clamp_q32(q - step_q, q_min, q_max);
    }

    return rec_idx;
}
