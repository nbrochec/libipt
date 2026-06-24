/*
 * Minimal C consumer of libipt — no C++, no torch, just ipt.h + libipt.
 *
 * Build: see ../README.md
 * Run:   ./example /path/to/model.ts
 *
 * Feeds synthetic audio blocks (replace with real audio device / file frames)
 * and prints the top class whenever the model produces a classification.
 */
#include "ipt.h"

#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.ts>\n", argv[0]);
        return 1;
    }

    /* 1. create + load the model (on this thread) */
    ipt_classifier* clf = ipt_create(argv[1], IPT_DEVICE_CPU,
                                     /*energy_threshold_db=*/-60.0,
                                     /*threshold_window_ms=*/20);
    if (!clf) {
        fprintf(stderr, "ipt_create failed: %s\n", ipt_last_error());
        return 1;
    }
    if (ipt_initialize_model(clf) != IPT_OK) {
        fprintf(stderr, "model load failed: %s\n", ipt_last_error());
        ipt_destroy(clf);
        return 1;
    }

    /* 2. configure audio settings */
    const int sr    = 48000;
    const int block = 512;
    ipt_init_buffers(clf, sr, block);
    ipt_set_smoothing_tau(clf, 100.0);  /* optional temporal smoothing */
    ipt_set_period(clf, 30.0); /* Set period */

    const int nclasses = ipt_num_classes(clf);
    printf("model loaded: %d classes\n", nclasses);

    /* 3. processing loop — feed mono audio blocks, read classifications */
    double buf[512];
    float  out_dist[256];
    float  smoothed[256];
    double latency;
    double phase = 0.0;

    for (int frame = 0; frame < 4000; ++frame) {
        for (int i = 0; i < block; ++i) {
            buf[i] = 0.2 * sin(phase);                 /* fake 440 Hz tone */
            phase += 2.0 * M_PI * 440.0 / (double) sr;
        }

        int n = ipt_process(clf, buf, block, out_dist, 256, &latency);
        if (n > 0) {
            /* smooth the raw distribution; -1 = use the live wall clock */
            ipt_smooth(clf, out_dist, n, -1.0, smoothed, 256);
            int best = 0;
            for (int i = 1; i < n; ++i) {
                if (smoothed[i] > smoothed[best]) best = i;
            }
            char name[128];
            ipt_get_class_name(clf, best, name, sizeof(name));
            printf("frame %4d -> %-20s p=%.3f  (%.2f ms)\n",
                   frame, name, smoothed[best], latency);
        } else if (n < 0) {
            fprintf(stderr, "process error: %s\n", ipt_last_error());
            break;
        }
    }

    /* 4. Acquire windows and classify in batch */
    printf("\nAcquiring windows and classifying in batch...\n");
    #define MAX_BATCH 128
    float* windows[MAX_BATCH] = {NULL};
    int window_length = 0;
    int batch_count = 0;
    phase = 0.0;
    
    for (int frame = 0; frame < 2000 && batch_count < MAX_BATCH; ++frame) {
        for (int i = 0; i < block; ++i) {
            buf[i] = 0.2 * sin(phase);
            phase += 2.0 * M_PI * 440.0 / (double) sr;
        }
        
        float* window = NULL;
        int len = ipt_acquire_window(clf, buf, block, &window);
        if (len > 0) {
            window_length = len;
            windows[batch_count++] = window;
        }
    }
    
    if (batch_count > 0) {
        printf("Acquired %d windows\n", batch_count);
        
        float dist[MAX_BATCH * 256];
        double latencies[MAX_BATCH];
        int n = ipt_classify_batch(clf, (const float* const*)windows, batch_count, 
                                   window_length, dist, MAX_BATCH * 256, latencies);
        if (n > 0) {
            printf("ipt_classify_batch processed %d windows in one forward pass\n", n);
        } else {
            fprintf(stderr, "classify_batch error: %s\n", ipt_last_error());
        }
        
        for (int i = 0; i < batch_count; i++) {
            ipt_free_window(windows[i]);
        }
    }

    ipt_destroy(clf);
    return 0;
}
