#!/usr/bin/env bash
# Regenerate the tables and the quoted numbers of the paper from data/ and check them (about 1 minute).
#   bash make_numbers.sh
# Optional: GKP_PAPER_TEX=/path/to/folder/with/main.tex+supplement.tex bash make_numbers.sh also checks that every
# recomputed number is printed in the paper with the stated wording.
# Logs and outputs: output/numbers/, output/tables/.
set -uo pipefail
cd "$(dirname "$0")"
source env.sh
mkdir -p output/numbers output/tables
status=0
run() {
  local name=$1; shift
  if "$PYTHON" "$@" > "output/numbers/$name.log" 2>&1; then
    echo "ok    $name  ($(tail -n 1 "output/numbers/$name.log" | cut -c1-100))"
  else
    echo "FAIL  $name  (see output/numbers/$name.log)"; status=1
  fi
}
run letter_numbers        numbers/letter_numbers.py
run letter_numbers2       numbers/letter_numbers2.py
run pulses_text_facts     numbers/pulses_text_facts.py
run letter_table1         tables/letter_table1_tracking.py
run si_table15            tables/target_only_preparation.py
run si_tables_10_12       tables/si_tables_10_12.py
run si_table3             tables/si_table3_state_gallery.py
run si_tables_2_7_8_9_13  tables/si_tables_2_7_8_9_13.py
grep -h "^FAIL" output/numbers/letter_numbers.log output/numbers/letter_numbers2.log output/numbers/letter_table1.log \
  output/numbers/si_table15.log output/numbers/si_tables_2_7_8_9_13.log || true
exit $status
