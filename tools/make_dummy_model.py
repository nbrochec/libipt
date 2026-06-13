#!/usr/bin/env python3
"""
Generate a dummy TorchScript model for libipt — for tests / CI smoke runs.

The model implements the exact contract libipt's Model loader expects
(see ipt_tilde/src/model.h):

    forward(x)         x: float32 tensor [1, 1, segment_length]  ->  logits [1, n_classes]
                       (libipt applies softmax over the last dim itself)
    get_sr()       ->  int   sample rate the model expects
    get_seglen()   ->  int   analysis window length, in samples
    get_classnames -> List[str]  one name per class

It does NOT classify anything meaningfully — the linear layer is randomly
initialised. Its only job is to load and run so the C ABI / torch link chain
can be exercised end to end.

Usage:
    python tools/make_dummy_model.py [output_path]      # default: tests/dummy.ts

Requires a torch matching the libtorch version libipt links (2.4.x). The
`clap` conda env on this machine (py3.12 + torch 2.4.1) works:
    conda run -n clap python tools/make_dummy_model.py
"""
import sys
from typing import List

import torch
import torch.nn as nn

# --- model parameters (the ones requested) ----------------------------------
SAMPLE_RATE = 24000
SEGMENT_LENGTH = 12000
CLASS_NAMES = ["pizzicato", "arco", "col_legno", "sul_ponticello"]


class DummyIPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # one linear layer: flattened window -> class logits
        self.fc = nn.Linear(SEGMENT_LENGTH, len(CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as [1, 1, SEGMENT_LENGTH]; flatten to [1, SEGMENT_LENGTH]
        x = x.reshape(1, -1)
        return self.fc(x)  # [1, n_classes] logits; libipt softmaxes these

    # literals are inlined below: TorchScript methods can't close over
    # Python module-level globals, so the values are baked in at script time.
    @torch.jit.export
    def get_sr(self) -> int:
        return 24000

    @torch.jit.export
    def get_seglen(self) -> int:
        return 12000

    @torch.jit.export
    def get_classnames(self) -> List[str]:
        return ["pizzicato", "arco", "col_legno", "sul_ponticello"]


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tests/dummy.ts"

    torch.manual_seed(0)  # reproducible weights -> reproducible output
    model = DummyIPT().eval()
    scripted = torch.jit.script(model)

    # sanity-check the contract before saving
    with torch.no_grad():
        y = scripted(torch.zeros(1, 1, SEGMENT_LENGTH))
    assert y.shape == (1, len(CLASS_NAMES)), y.shape
    assert scripted.get_sr() == SAMPLE_RATE
    assert scripted.get_seglen() == SEGMENT_LENGTH
    assert scripted.get_classnames() == CLASS_NAMES

    scripted.save(out_path)
    print(f"wrote {out_path}: torch {torch.__version__}, "
          f"sr={SAMPLE_RATE}, seglen={SEGMENT_LENGTH}, "
          f"classes={CLASS_NAMES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
