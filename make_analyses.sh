#!/usr/bin/env bash
# Rerun the analyses that turn the stored simulation results into the derived data files the figures and tables read
# (a few minutes on one core each), in a COPY of data/, and compare the regenerated files with the stored ones.
# data/ itself is never written.
#   bash make_analyses.sh            (copy in output/data_rerun/)
set -uo pipefail
cd "$(dirname "$0")"
source env.sh
COPY="$PWD/output/data_rerun"
rm -rf "$COPY"; mkdir -p output/analyses "$(dirname "$COPY")"
cp -r data "$COPY"
status=0
run() {
  local name=$1; shift
  local t0=$(date +%s)
  if GKP_DATA="$COPY" "$PYTHON" "$@" > "output/analyses/$name.log" 2>&1; then
    echo "ok    $name  $(( $(date +%s) - t0 )) s"
  else
    echo "FAIL  $name  (see output/analyses/$name.log)"; status=1
  fi
}
run flip_gate_summary  simulations/analysis/round3/flip_gate_summary.py      # Supplementary Table 11
run route_test         simulations/analysis/round3/route_test.py             # Supplementary Table 5
run stabilizers_along  simulations/analysis/round3/stabilizers_along.py
run codespace          simulations/analysis/round3/codespace.py
run tradeoff           simulations/analysis/round3/tradeoff.py               # Supplementary Fig. 9
run pauli_frame        simulations/analysis/round3/pauli_frame.py
run rates_parity       simulations/analysis/round3/rates_parity.py           # parity-detectable error share
run audit_photon2      simulations/analysis/photon_flip/audit_photon2.py     # transfers through the vacuum (Table 2)
run transfer_closed    simulations/analysis/round4/transfer_closed.py        # closed-form columns of Table 2
run flip_note          simulations/flip_note/num_core.py                     # Supplementary Fig. 18 data
"$PYTHON" tools/compare_data.py data "$COPY" | tail -n 40 || status=1
exit $status
