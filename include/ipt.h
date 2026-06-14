/*
 * libipt — C API for real-time Instrumental Playing Technique (IPT) classification.
 *
 * Thin, host-agnostic C wrapper over the ipt_tilde inference core (IptClassifier).
 * Link against libipt and include this single header — no C++, torch, or buffer
 * internals leak to the consumer.
 *
 * Typical lifecycle (all calls on ONE thread — the audio/processing thread):
 *
 *     ipt_classifier* clf = ipt_create("model.ts", IPT_DEVICE_CPU, -60.0, 20);
 *     ipt_initialize_model(clf);            // loads TorchScript, parses metadata
 *     ipt_init_buffers(clf, 48000, 512);    // sample rate + audio block size
 *     ...
 *     float dist[256]; double latency;
 *     int n = ipt_process(clf, samples, num_samples, dist, 256, &latency);
 *     if (n > 0) { ... dist[] holds the class probability distribution ... }
 *     ...
 *     ipt_destroy(clf);
 */
#ifndef IPT_H
#define IPT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. */
typedef struct ipt_classifier ipt_classifier;

typedef enum {
    IPT_DEVICE_CPU = 0,
    IPT_DEVICE_MPS = 1   /* Apple Metal (GPU) */
} ipt_device;

typedef enum {
    IPT_OK              =  0,
    IPT_ERR_NULL        = -1,  /* null handle */
    IPT_ERR_NOT_INIT    = -2,  /* model/buffers not initialized */
    IPT_ERR_TORCH       = -3,  /* TorchScript load/inference failed (see ipt_last_error) */
    IPT_ERR_INVALID_ARG = -4
} ipt_status;

/*
 * Create a classifier for a TorchScript (.ts) model. Returns NULL on failure
 * (see ipt_last_error). Does NOT load the model yet — call ipt_initialize_model()
 * on the thread that will subsequently call ipt_process().
 *
 *   energy_threshold_db : RMS gate below which audio is treated as silence
 *                         (e.g. -60.0; <= -80.0 disables the gate)
 *   threshold_window_ms : window length used for the silence/onset gate (e.g. 20)
 */
ipt_classifier* ipt_create(const char* model_path,
                           ipt_device device,
                           double energy_threshold_db,
                           int threshold_window_ms);

/* Free a classifier. Safe to call with NULL. */
void ipt_destroy(ipt_classifier* h);

/*
 * Load the TorchScript model and parse its embedded metadata (sample rate,
 * segment length, class names). Must run on the same thread as ipt_process().
 * Returns IPT_OK or IPT_ERR_TORCH (see ipt_last_error).
 */
int ipt_initialize_model(ipt_classifier* h);

/*
 * (Re)allocate the internal buffers. Call after ipt_initialize_model(), and
 * again whenever the sample rate or block size changes.
 *   sample_rate         : sample rate of the audio you will feed
 *   input_vector_length : audio block size per ipt_process() call (an upper
 *                         bound is fine; larger blocks are handled internally)
 */
int ipt_init_buffers(ipt_classifier* h, int sample_rate, int input_vector_length);

/*
 * Feed one block of mono audio (double samples) and, when a classification is
 * produced, receive the class probability distribution.
 *
 *   out_dist  : caller-provided buffer of out_cap floats; receives the softmax
 *               distribution (one probability per class)
 *   out_latency_ms : optional (may be NULL); set to inference latency in ms when
 *               a classification is produced
 *
 * Returns:
 *    > 0  number of classes in the distribution (a classification was produced).
 *         Up to min(return, out_cap) values were written to out_dist; if the
 *         return value exceeds out_cap your buffer was too small.
 *      0  no classification this call (still buffering, or below energy threshold)
 *    < 0  error (an ipt_status value; see ipt_last_error)
 */
int ipt_process(ipt_classifier* h,
                const double* samples, int num_samples,
                float* out_dist, int out_cap,
                double* out_latency_ms);

/*
 * Optional temporal smoothing of the output distribution (leaky integrator).
 * tau_ms is the time constant in milliseconds; 0 disables smoothing (default).
 * In the Max object: tau_ms = (1 - sensitivity) * range_ms where sensitivity [0,1] and range_ms [0,2000].
 */
void ipt_set_smoothing_tau(ipt_classifier* h, double tau_ms);

void ipt_set_energy_threshold(ipt_classifier* h, double threshold_db);
void ipt_set_threshold_window(ipt_classifier* h, int duration_ms);

/* Number of classes the loaded model predicts, or -1 if the model isn't loaded. */
int ipt_num_classes(ipt_classifier* h);

/*
 * Copy class name [index] into out as a NUL-terminated string (truncated to
 * out_cap). Returns the number of characters written, or -1 on error.
 */
int ipt_get_class_name(ipt_classifier* h, int index, char* out, int out_cap);

/* Human-readable description of the last error on the calling thread. Never NULL. */
const char* ipt_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IPT_H */
