# libipt

[![CI](https://github.com/nbrochec/libipt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nbrochec/libipt/actions/workflows/ci.yml)

A C API for real-time **Instrumental Playing Technique (IPT)** classification.

**Disclaimer: Part of this code was written with Claude Opus 4.8 and Claude Fable 5.**
Claude code was utilized for faster implementation of the API. The code has been reviewed and validated by a human before making it available online.

`libipt` is a self-contained C library for IPT classification. It **owns** the
inference core (the `IptClassifier` routine, originally developed in
[`ipt_tilde`](https://github.com/DYCI2/ipt_tilde): TorchScript model loading →
resampling buffer → energy gating → softmax → temporal smoothing) and exposes it
through a single C ABI (`ipt.h`). It lets **any** native project, a Pd external, a
SuperCollider UGen, a VST/AU plugin, a VAMP plugin, a CLI tool, `ipt_tilde`'s own
Max/PiPo externals, or another C/C++ app, reuse the exact same classifier without
reimplementing the pipeline or touching C++/torch.

## Layout

```
libipt/
├── include/ipt.h            # the entire public API (one header)
├── src/ipt.cpp              # C ABI implementation over IptClassifier
├── core/                    # the C++ inference core (header-only, private to libipt)
├── libs/r8brain/            # vendored r8brain resampler
├── examples/example.c       # minimal C consumer
├── tools/make_dummy_model.py# generates the dummy TorchScript test model
├── tests/dummy.ts           # tiny TorchScript model used by CI (4 classes)
└── CMakeLists.txt
```

## Requirements

- macOS Apple Silicon (arm64), or Windows x64 (Visual Studio 2022+)
- CMake ≥ 3.19
- No external checkout needed — libipt is self-contained. r8brain is vendored;
  libtorch (2.4.1, CPU — the same version on every platform, so TorchScript
  models are interchangeable) is downloaded into `libs/libtorch` automatically
  on first configure if absent.

## Build

```bash
cd libipt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

Produces `build/libipt.dylib` and the `build/example` demo.

```bash
./build/example /path/to/model.ts
```

On Windows (multi-config MSVC generator):

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

Produces `build/Release/ipt.dll` (with its `ipt.lib` import library) and
`build/Release/example.exe`; the torch DLLs it needs are copied beside the
example automatically.

## Using it in another project

Include the one header and link the library:

```c
#include "ipt.h"

ipt_classifier* clf = ipt_create("model.ts", IPT_DEVICE_CPU, -60.0, 20); //path to model, device, threshold in dB, window in ms
ipt_initialize_model(clf);          // load TorchScript (same thread as process)
ipt_init_buffers(clf, 48000, 512);  // sample rate + audio block size

float out_dist[256]; double latency;
int n = ipt_process(clf, samples, num_samples, out_dist, 256, &latency); // classifier, samples block, sample count, out prob distribution buffer, its capacity, latency pointer
// n > 0  -> out_dist[] holds the per-class probability distribution
// n == 0 -> no classification this block (buffering / silence)
// n < 0  -> error, see ipt_last_error()

ipt_destroy(clf);
```

`gcc myapp.c -I/path/to/libipt/include -L/path/to/libipt/build -lipt`

### Threads

torch runs the forward pass on an intra-op thread pool sized to the machine's
cores by default. `ipt_set_num_threads(n)` lets a host pick a smaller pool
(process-global; call it **before** the first `ipt_initialize_model()` in the
process, torch ignores later changes). The trade-off is latency against cores
left free for the host, e.g. flute model on an M4 Pro: 1 thread 4.3 ms / 100 %
CPU, 4 threads 3.0 ms / 190 %, default 10 threads 3.1 ms / 267 %. There is no
single right value, so the default is left to torch and hosts expose the knob.

Models are also frozen (`torch::jit::freeze`) at load and run under
`c10::InferenceMode`, which is 10-22 % faster than the plain eval module.

### Runtime note

`libipt` depends on the libtorch runtime libraries. On macOS the build bakes an
rpath to libipt's own `libs/libtorch` so the example runs in place — but that
path is specific to your machine. On Windows there is no rpath: the DLLs simply
have to sit next to the executable (the build copies them there for the
example). To **ship** libipt elsewhere, use the install step below.

## Distributing (self-contained bundle)

`cmake --install` gathers a relocatable bundle (the header, `libipt.dylib`, and
the libtorch dylibs it needs) into one folder you can zip and hand off:

```bash
cmake --install build            # -> ./dist  (or: --prefix /some/where)
```

```
dist/
├── include/ipt.h
└── lib/
    ├── libipt.dylib                 # rpath = @loader_path
    ├── libtorch.dylib
    ├── libtorch_cpu.dylib
    └── libc10.dylib
```

On Windows the same step gathers `lib/ipt.dll`, its `ipt.lib` import library,
and the torch DLL closure (`torch.dll`, `torch_cpu.dll`, `c10.dll`,
`fbgemm.dll`, `asmjit.dll`, `libiomp5md.dll`, `uv.dll`, …) into `lib/`.

`libipt.dylib` finds the torch dylibs sitting next to it (`@loader_path`), so the
whole `lib/` folder works wherever it's copied; on Windows, DLLs resolve each
other by co-location the same way. A consumer builds against it and, on macOS,
adds an rpath to the bundle's `lib/`:

```bash
gcc myapp.c -Idist/include -Ldist/lib -lipt -Wl,-rpath,/path/to/dist/lib
```

(Or just place the binary inside `dist/lib/` — on Windows that co-location IS
the mechanism.) Note: the default `dist` prefix
only applies on a build tree already configured with
another prefix, pass `--prefix` explicitly.

## Testing

[`tests/dummy.ts`](tests/dummy.ts) is a tiny TorchScript model committed for
CI and local smoke tests; `sr=24000`, `segment_length=12000`, 4 classes
(`pizzicato`, `arco`, `col_legno`, `sul_ponticello`). It implements the model
contract libipt expects (`forward`, `get_sr`, `get_seglen`, `get_classnames`)
but its weights are random, it only proves the C ABI / torch chain loads and
classifies end to end.

```bash
./build/example tests/dummy.ts      # should print "model loaded: 4 classes" + per-frame classes
bash .github/scripts/smoke.sh       # the exact check CI runs
```

Regenerate it (needs a torch matching the linked libtorch, 2.4.x):

```bash
python tools/make_dummy_model.py tests/dummy.ts
```

## API reference

See [`include/ipt.h`](include/ipt.h), every function is documented there.

## Licence

> libipt is distributed under CC BY-NC 4.0, inherited from ipt_tilde, which it incorporates. Commercial use is not permitted. (The wrapper code in include/, src/, examples/ is additionally available under MIT, but the built library as a whole remains CC BY-NC 4.0.)
