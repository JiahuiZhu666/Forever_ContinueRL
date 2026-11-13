#!/usr/bin/env bash
# Scheme C + KL anchor + success-based task switching (es) + KL-checked neuron release, CW10, actor 4x256.
#
#   scheme C : every task draws K=5 fresh random neuron masks (25 % per layer), forward pass masked (Nevo structure),
#              weights in blocks of earlier winners are multiplied by lam^h AFTER Adam (lam = 0.01)
#   + KL     : 4096 stored states + old (mu, sigma) per finished task; actor loss += 30 * KL(pi_old || pi_theta),
#              each anchor state evaluated under its own task's mask
#   + es     : every 20k steps a 10-episode eval; at success >= THR a 50-episode confirmation with fresh seeds;
#              the first candidate that confirms wins the task and the other 4 stop. Cap 1M steps per candidate;
#              without a confirmation all train to the cap and the winner is chosen by return
#   + release: before each task t > 0, unimportant neurons are removed from old tasks' masks one at a time, each
#              accepted only if step KL <= 0.1 and cumulative KL <= 0.3 on that task's anchor states; neurons no old
#              task references any more are re-initialised and returned to the pool (neuron_release.py)
#
# usage: bash scripts/launch_ckl_es_release.sh <exp_dir> [seed] [mask_seed] [threshold] [mask_priority]
#   defaults: seed 110, mask seed 7, threshold 0.9, mask priority "released" (new masks take released neurons first;
#   "none" = plain random masks)
# Needs scripts/env.sh (copy env.sh.example). Runs detached; log in <exp_dir>.log
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$HERE/scripts/env.sh"
EXP=${1:?experiment directory}; SEED=${2:-110}; MASKSEED=${3:-7}; THR=${4:-0.9}; PRI=${5:-released}
[ -e "$EXP/ledger.json" ] && echo "note: $EXP has a ledger; the driver resumes it at task granularity"
mkdir -p "$(dirname "$EXP")"
cd "$HERE/code"
setsid nohup python run_maskdecay.py --exp-dir "$EXP" --env-name cw10 --tasks 10 --candidates 5 --parallel 5 \
  --lam 0.01 --steps 1000000 --final-eval-episodes 50 --eval-interval 20000 --eval-episodes 10 \
  --hidden-layers 4 --critic-width 256 --actor-width 256 --seed "$SEED" --mask-seed "$MASKSEED" \
  --forward-masking 1 --include-biases 1 --mask-rule random --sample-ratio 0.25 \
  --anchor-coef 30 --anchor-states-per-task 4096 --anchor-batch 256 \
  --early-stop-success "$THR" --confirm-episodes 50 \
  --release 1 --release-kl-step 0.1 --release-kl-cum 0.3 --release-frac 0.05 --release-max-tries 16 \
  --mask-priority "$PRI" > "$EXP.log" 2>&1 < /dev/null &
echo "launched C+KL+es$THR+release (seed $SEED, mask seed $MASKSEED, priority $PRI) -> $EXP (pid $!), log $EXP.log"
