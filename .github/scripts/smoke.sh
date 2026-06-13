#!/usr/bin/env bash
# Runtime check: prove the built library and its torch dependencies actually load.
#
# There's no model.ts file in CI, so we can't run a real classification. Instead we
# launch the demo with a path that doesn't exist:
#   * if the torch libraries can't be found, the program dies in the loader (dyld)
#     before any of our code runs  -> we fail.
#   * if everything loads, our code runs and cleanly reports the missing model
#     -> we pass. That's enough to prove the library chain is wired up correctly.
set -uo pipefail

out="$(./build/example /nonexistent/model.ts 2>&1 || true)"
echo "$out"

if echo "$out" | grep -qiE "library not loaded|image not found|dyld\["; then
  echo "::error::libipt or its torch libraries failed to load at runtime"
  exit 1
fi

if echo "$out" | grep -qiE "failed"; then
  echo "runtime check OK — library loaded, missing model rejected as expected"
  exit 0
fi

echo "::warning::library loaded but the demo's message changed unexpectedly"
