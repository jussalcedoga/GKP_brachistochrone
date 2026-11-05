# Reproduce the figures, tables and numbers of the paper from the stored results in data/.
#   make figures       every figure of the Letter and the SI, then the check against the published PNG files
#   make numbers       the tables and the quoted numbers, with their checks
#   make analyses      rerun the analyses of the stored results in a copy of data/ and compare
#   make test          unit tests of the shared library (common/)
#   make verify        integrity of data/ against data/SHA256SUMS
#   make csv           rewrite the CSV copies of the result tables in data/csv/
#   make Fig1 ... Fig4, make SFig1 ... SFig18      one figure (Letter Fig. N, Supplementary Fig. N)
#   make clean         remove output/
SHELL := /bin/bash
PYTHON ?= python3
JOBS ?= 4
RUN = source env.sh && PYTHON=$(PYTHON)

.PHONY: all figures numbers analyses check test verify csv clean list
all: verify figures numbers

verify:
	$(PYTHON) tools/verify_data.py

csv:
	$(RUN) $(PYTHON) tools/export_csv.py

figures:
	$(RUN) bash make_figures.sh -j $(JOBS)

numbers:
	$(RUN) bash make_numbers.sh

analyses:
	$(RUN) bash make_analyses.sh

check:
	$(RUN) $(PYTHON) tools/compare_figures.py

list:
	$(RUN) $(PYTHON) figures/make_figures.py --list

test:
	$(RUN) $(PYTHON) tests/test_common.py && $(PYTHON) tests/test_tracking.py && $(PYTHON) tests/test_snap.py

Fig% SFig%:
	$(RUN) $(PYTHON) figures/make_figures.py $@

clean:
	rm -rf output
