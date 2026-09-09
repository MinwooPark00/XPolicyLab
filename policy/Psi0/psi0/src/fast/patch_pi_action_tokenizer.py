#!/usr/bin/env python3
"""
Patches src/fast/pi/processing_action_tokenizer.py to add padding/truncation
in decode() before reshape, matching the egodex-rel-50w-1x48-v2048-s100 version.
"""

import sys
from pathlib import Path

TARGET = Path(__file__).parent / "pi" / "processing_action_tokenizer.py"

OLD = "                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)\n"

NEW = """\
                if decoded_dct_coeff.size < self.time_horizon * self.action_dim:
                    decoded_dct_coeff = np.pad(
                        decoded_dct_coeff,
                        (0, self.time_horizon * self.action_dim - decoded_dct_coeff.size),
                    )
                elif decoded_dct_coeff.size > self.time_horizon * self.action_dim:
                    decoded_dct_coeff = decoded_dct_coeff[: self.time_horizon * self.action_dim]

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
"""

text = TARGET.read_text()

if NEW.strip() in text:
    print("Already patched, nothing to do.")
    sys.exit(0)

if OLD not in text:
    print(f"ERROR: expected string not found in {TARGET}", file=sys.stderr)
    sys.exit(1)

TARGET.write_text(text.replace(OLD, NEW, 1))
print(f"Patched {TARGET}")
