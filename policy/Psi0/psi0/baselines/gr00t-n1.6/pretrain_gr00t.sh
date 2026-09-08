#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: pretrain_gr00t.sh [--preset NAME] [additional finetune_gr00t.py args...]

Presets:
  pretrain_g1_ee
  pretrain_h1_ee
  pretrain_g1_manip
  pretrain_h1_manip
  pretrain_he_mixed_scratch

Examples:
  bash baselines/gr00t-n1.6/pretrain_gr00t.sh --preset pretrain_g1_ee
  bash baselines/gr00t-n1.6/pretrain_gr00t.sh --preset pretrain_he_mixed_scratch --dry-run
EOF
}

PRESET="${PRESET:-pretrain_g1_ee}"

if [[ $# -ge 1 ]]; then
  case "$1" in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

exec python3 "$SCRIPT_DIR/finetune_gr00t.py" \
  --preset "$PRESET" \
  "${@}"
