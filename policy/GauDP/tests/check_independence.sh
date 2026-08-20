#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_name="Policy""-Lightning"
path_variable="PYTHON""PATH"
forbidden="/home/usd/${external_name}|\\.\\./${external_name}|${path_variable}|^(from|import) (model|policy|common|src)\\b"

if find "${POLICY_DIR}" \( -type l -o -name .git -o -name .gitmodules \) -print -quit | grep -q .; then
    echo "GauDP contains a symlink or nested repository" >&2
    exit 1
fi
if rg -n "${forbidden}" "${POLICY_DIR}" -g '*.py' -g '*.sh'; then
    echo "GauDP contains a forbidden runtime reference" >&2
    exit 1
fi

cd /tmp
env -u "${path_variable}" python "${POLICY_DIR}/process_data.py" --help >/dev/null
env -u "${path_variable}" python "${POLICY_DIR}/train_gaussian.py" --help >/dev/null
env -u "${path_variable}" python "${POLICY_DIR}/train_policy.py" --help >/dev/null
env -u "${path_variable}" python - <<'PY'
import sys
import XPolicyLab.policy.GauDP.model

for name, module in sys.modules.items():
    path = getattr(module, "__file__", None)
    if "GauDP" in name and path and "/policy/GauDP/" not in path and "site-packages" not in path:
        raise AssertionError(f"{name} resolved outside GauDP/site-packages: {path}")
PY

echo "[GauDP] independence checks passed"
