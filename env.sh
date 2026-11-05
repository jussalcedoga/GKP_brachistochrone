# Source this file from the repository root before running any script by hand:   source env.sh
# It puts the shared library and the simulation modules on the Python path and pins the thread counts of the numerical
# libraries to one (as in the production runs).
_GKP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$_GKP_ROOT/src:$_GKP_ROOT/simulations/tracking:$_GKP_ROOT/simulations/gate_sets:$_GKP_ROOT/simulations/analysis:$_GKP_ROOT/simulations/analysis/review_oct28:$_GKP_ROOT/simulations/analysis/photon_flip${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHON="${PYTHON:-python3}"
unset _GKP_ROOT
