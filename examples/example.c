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

    const int nclasses = ipt_num_classes(clf);
    printf("model loaded: %d classes\n", nclasses);

    /* 3. processing loop — feed mono audio blocks, read classifications */
    double buf[512];
    float  dist[256];
    double latency;
    double phase = 0.0;

    for (int frame = 0; frame < 4000; ++frame) {
        for (int i = 0; i < block; ++i) {
            buf[i] = 0.2 * sin(phase);                 /* fake 440 Hz tone */
            phase += 2.0 * M_PI * 440.0 / (double) sr;
        }

        int n = ipt_process(clf, buf, block, dist, 256, &latency);
        if (n > 0) {
            int best = 0;
            for (int i = 1; i < n; ++i) {
                if (dist[i] > dist[best]) best = i;
            }
            char name[128];
            ipt_get_class_name(clf, best, name, sizeof(name));
            printf("frame %4d -> %-20s p=%.3f  (%.2f ms)\n",
                   frame, name, dist[best], latency);
        } else if (n < 0) {
            fprintf(stderr, "process error: %s\n", ipt_last_error());
            break;
        }
    }

    ipt_destroy(clf);
    return 0;
}
