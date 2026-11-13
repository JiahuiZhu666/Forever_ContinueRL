#!/usr/bin/env bash
# Unit tests: mask-decay mechanics (test_maskdecay.py) and neuron release (test_release.py; needs a real run with
# start_task03.pkl, given by CKL_TEST_EXP). usage: CKL_TEST_EXP=<run dir> bash scripts/run_tests.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$HERE/scripts/env.sh"
cd "$HERE/code"
echo "== test_maskdecay.py"
python test_maskdecay.py 2>&1 | grep -E "^(PASS|FAIL)|Traceback|Error" | awk '{print $1}' | sort | uniq -c
python test_maskdecay.py 2>&1 | grep -E "^FAIL|Traceback" | head
echo "== test_release.py (CKL_TEST_EXP=${CKL_TEST_EXP:-unset})"
python test_release.py 2>&1 | grep -E "^(PASS|FAIL)|Traceback|Error|needs a run" | awk '{print $1}' | sort | uniq -c
python test_release.py 2>&1 | grep -E "^FAIL|Traceback|needs a run" | head
exit 0   # the counts above are the result; no FAIL line means everything passed
