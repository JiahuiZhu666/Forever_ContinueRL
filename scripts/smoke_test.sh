#!/usr/bin/env bash
# Short end-to-end check of the full pipeline (C + KL + es + release) on the first 3 CW10 tasks with tiny budgets:
# 2 candidates, 3000 steps each, start training after 1000 steps. Exercises mask drawing, KL anchors, the es
# evaluation/confirmation path, neuron release before tasks 1 and 2 (with released-first masks), and the ledger.
# Not a result. usage: bash scripts/smoke_test.sh [out_dir]   (default /tmp/ckl_smoke_<pid>)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$HERE/scripts/env.sh"
OUT=${1:-/tmp/ckl_smoke_$$}
cd "$HERE/code"
python run_maskdecay.py --exp-dir "$OUT" --env-name cw10 --tasks 3 --candidates 2 --parallel 2 \
  --lam 0.01 --steps 3000 --start-training 1000 --final-eval-episodes 2 --eval-interval 1000 --eval-episodes 2 \
  --hidden-layers 4 --critic-width 256 --actor-width 256 --seed 110 --mask-seed 7 \
  --forward-masking 1 --include-biases 1 --mask-rule random --sample-ratio 0.25 \
  --anchor-coef 30 --anchor-states-per-task 512 --anchor-batch 64 \
  --early-stop-success 0.9 --confirm-episodes 2 \
  --release 1 --release-kl-step 0.1 --release-kl-cum 0.3 --release-frac 0.05 --release-max-tries 4 \
  --mask-priority released
rc=$?
echo "driver exit code $rc"
python - "$OUT" <<'EOF'
import json, os, sys
d = sys.argv[1]
L = json.load(open(os.path.join(d, "ledger.json")))
print("tasks committed:", len(L["tasks"]), "| releases logged:", len(L.get("releases", [])))
for t in L["tasks"]:
    print(" ", t["task_index"], t["task_name"], "winner", t["winner"], "confirmed", t["winner_confirmed"],
          "row", [round(x["success"], 2) for x in t["winner_final_eval"]])
ok = len(L["tasks"]) == 3 and len(L.get("releases", [])) == 2
print("SMOKE_TEST", "PASS" if ok else "FAIL")
EOF
