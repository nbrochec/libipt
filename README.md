# libipt

A C API for real-time **Instrumental Playing Technique (IPT)** classification.

**Disclaimer: Part of this code was written with Claude Opus 4.8.**

`libipt` is a thin C ABI wrapper around the inference core of
[`ipt_tilde`](https://github.com/DYCI2/ipt_tilde) (the `IptClassifier` engine:
TorchScript model loading → resampling buffer → energy gating → softmax →
optional temporal smoothing). It lets **any** native project — a Pd external, a
SuperCollider UGen, a VST/AU plugin, a VAMP plugin, a CLI tool, or another C/C++
app — reuse the exact same classifier without reimplementing the pipeline or
touching C++/torch.

The core sources are **reused, not copied**, from a sibling `ipt_tilde` checkout
— there is a single source of truth.

## Layout

```
libipt/
├── include/ipt.h     # the entire public API (one header)
├── src/ipt.cpp       # C ABI implementation over IptClassifier
├── examples/example.c# minimal C consumer
└── CMakeLists.txt
```

## Requirements

- macOS, Apple Silicon (arm64) — same constraint as `ipt_tilde`
- CMake ≥ 3.19
- A checkout of `ipt_tilde` next to this folder (`../ipt_tilde`), or pointed to via `-DIPT_TILDE_DIR=...`.
  libtorch is reused from `ipt_tilde/libs/libtorch` (downloaded there automatically if absent).

## Build

```bash
cd libipt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # add -DIPT_TILDE_DIR=/path/to/ipt_tilde if not a sibling
cmake --build build -j 8
```

Produces `build/libipt.dylib` and the `build/example` demo.

```bash
./build/example /path/to/model.ts
```

## Using it in another project

Include the one header and link the library:

```c
#include "ipt.h"

ipt_classifier* clf = ipt_create("model.ts", IPT_DEVICE_CPU, -60.0, 20);
ipt_initialize_model(clf);          // load TorchScript (same thread as process)
ipt_init_buffers(clf, 48000, 512);  // sample rate + audio block size

float dist[256]; double latency;
int n = ipt_process(clf, samples, num_samples, dist, 256, &latency);
// n > 0  -> dist[] holds the per-class probability distribution
// n == 0 -> no classification this block (buffering / silence)
// n < 0  -> error, see ipt_last_error()

ipt_destroy(clf);
```

`gcc myapp.c -I/path/to/libipt/include -L/path/to/libipt/build -lipt`

### Runtime note

`libipt.dylib` depends on the libtorch dylibs. The build bakes an rpath to
`ipt_tilde`'s libtorch so the example runs in place. To **ship** libipt in
another app, copy the torch dylibs (`libtorch_cpu.dylib`, `libc10.dylib`, …)
next to your binary and set the rpath accordingly — the same approach
`ipt_tilde`'s `pipo.ipt` uses when it bundles torch into the `.mxo`.

## API reference

See [`include/ipt.h`](include/ipt.h) — every function is documented there.
