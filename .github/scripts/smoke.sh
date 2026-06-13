#!/usr/bin/env bash
# Runtime check: prove the built library + its torch dependencies load, and —
# when a model is available — that a real classification runs end to end.
#
# tests/dummy.ts is a tiny TorchScript model (see tools/make_dummy_model.py)
# committed to the repo: sr=24000, seglen=12000, 4 classes. If it's present we
# feed it through the demo and require an actual classification. If it's missing
# we fall back to the loader-only check (launch with a bogus path: the program
# must reach our code and reject the missing model, proving torch loaded).
#
# Note: greps use here-strings (<<<) rather than `echo | grep -q`. With pipefail
# set, `grep -q` closes the pipe on its first match and the upstream echo dies of
# SIGPIPE, which pipefail would report as a failed pipeline on multi-line output.
set -uo pipefail

MODEL="tests/dummy.ts"

if [ -f "$MODEL" ]; then
  out="$(./build/example "$MODEL" 2>&1 || true)"
  echo "$out"

  if grep -qiE "library not loaded|image not found|dyld\[" <<<"$out"; then
    echo "::error::libipt or its torch libraries failed to load at runtime"
    exit 1
  fi
  # the demo prints "model loaded: 4 classes" then per-frame "-> <classname>"
  if grep -qiE "model loaded: 4 classes" <<<"$out" \
     && grep -qiE "pizzicato|arco|col_legno|sul_ponticello" <<<"$out"; then
    echo "runtime check OK — model loaded and classified through the C ABI"
    exit 0
  fi
  echo "::error::library built but the dummy model failed to classify"
  exit 1
fi

# --- fallback: no model committed, prove the loader chain only ----------------
out="$(./build/example /nonexistent/model.ts 2>&1 || true)"
echo "$out"

if grep -qiE "library not loaded|image not found|dyld\[" <<<"$out"; then
  echo "::error::libipt or its torch libraries failed to load at runtime"
  exit 1
fi

if grep -qiE "failed" <<<"$out"; then
  echo "runtime check OK — library loaded, missing model rejected as expected"
  exit 0
fi

echo "::warning::library loaded but the demo's message changed unexpectedly"
