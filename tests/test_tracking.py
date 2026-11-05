"""Finite-difference check of the tracking (hinge) gradient, uniform and free progress, both gate sets.
python tests/test_tracking.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import pathlib
import sys

sys.path.insert(0, str(_L.vfile(__file__).parents[1]))

import numpy as np  # noqa: E402

from common import brachistochrone as B, grape, states, tracking  # noqa: E402

D, N, g = 14, 6, 7.5
S, T = states.vacuum(D), states.fock(2, D)
c0, th, chi = B.geodesic_parts(S, T)
path = tracking.Path(B.path_states(S, T, np.linspace(0, th, 201)))
rng = np.random.default_rng(1)
worst = 0.0
for gate in ("cs", "cd"):
    M = tracking.make_model(gate, D, g, 10 * g)
    psi0 = grape.joint(S)
    for free in (False, True):
        X = grape.random_start(N, rng)
        x = X.ravel() if not free else np.r_[X.ravel(), rng.uniform(0.5, 1.5, N)]
        args = (M, None, N, 0.04, psi0, path, 50.0, 0.97, 1e3, 0.99995, 10.0, free)
        f = lambda y: tracking.cost_grad(args[0], y, *args[2:])[0]  # noqa: E731
        c, gr, _ = tracking.cost_grad(M, x, *args[2:])
        idx = rng.choice(x.size, 12, replace=False)
        h = 1e-6
        fd = np.array([(f(x + h * np.eye(x.size)[i]) - f(x - h * np.eye(x.size)[i])) / (2 * h) for i in idx])
        err = float(np.max(np.abs(fd - gr[idx])) / max(1e-12, np.max(np.abs(fd))))
        worst = max(worst, err)
        print(f"{gate} free={free}: cost {c:.4e}, relative gradient error {err:.2e}")
assert worst < 1e-5, worst
print("tracking gradient OK")
