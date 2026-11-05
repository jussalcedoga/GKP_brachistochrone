"""Checks of the SNAP gate-set model (common/tracking.SNAPModel) before any production run.
python tests/test_snap.py   (prints PASS/FAIL per check)"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import math
import pathlib
import sys

ROOT = _L.vfile(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fig4_cs_path_following"))
import numpy as np  # noqa: E402

from common import brachistochrone as B, grape, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402

g = G_SQ
ok_all = True


def check(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


D = 12
M = tracking.SNAPModel(D, g, 10 * g)
e = lambda n, q: np.eye(2 * D)[q * D + n]  # noqa: E731  |n>|q>, q = 0 (g), 1 (e)

# 1. the dispersive drift: (|0> + |1>)|e> picks up the relative phase chi t
t = 0.37
psi0 = (e(0, 1) + e(1, 1)) / math.sqrt(2)
tr, _, _ = M.propagate(np.zeros((1, 6)), t, psi0)
ph = np.angle(tr[-1][D + 1] / tr[-1][D + 0])
check("dispersive phase", abs(ph - g * t) < 1e-10, f"relative phase {ph:.6f} vs chi t = {g * t:.6f} (H0 = -chi n |e><e|)")

# 2. number selectivity: a weak qubit pi pulse resonant with n = 0 flips |0,g> but hardly |1,g>
Om = g / 5
C = np.zeros((1, 6))
C[0, 0] = Om
tp = math.pi / Om
pe = []
for n in (0, 1):
    tr, _, _ = M.propagate(C, tp, e(n, 0))
    pe.append(float(np.sum(np.abs(tr[-1][D:]) ** 2)))
bound = Om ** 2 / (Om ** 2 + g ** 2)
check("number-selective rotation", pe[0] > 0.999 and pe[1] <= bound + 1e-9,
      f"P_e(n=0) = {pe[0]:.4f}, P_e(n=1) = {pe[1]:.4f} <= Omega^2/(Omega^2+chi^2) = {bound:.4f}")

# 3. the vacuum speed at the bound: Delta H = g/sqrt2 for |eps| = g (the resource shared with CS and CD)
C = np.zeros((1, 6))
C[0, 2] = g
H = M.hamiltonians(C)[0]
v = e(0, 0)
dH = math.sqrt(float(np.real(np.vdot(v, H @ H @ v) - np.vdot(v, H @ v) ** 2)))
check("vacuum Fubini-Study speed", abs(dH - g / math.sqrt(2)) < 1e-12, f"Delta H = {dH:.6f} vs g/sqrt2 = {g / math.sqrt(2):.6f}")
for gate in ("cs", "cd"):
    Mo = tracking.make_model(gate, D, g, 10 * g)
    C = np.zeros((1, 6))
    C[0, 2] = g
    H = Mo.hamiltonians(C)[0]
    dH = math.sqrt(float(np.real(np.vdot(v, H @ H @ v) - np.vdot(v, H @ v) ** 2)))
    check(f"same speed for {gate}", abs(dH - g / math.sqrt(2)) < 1e-12, f"Delta H = {dH:.6f}")

# 4. exact gradient of the tracking loss against central finite differences (random path and controls)
rng = np.random.default_rng(3)
N = 6
S = np.zeros(D, complex)
S[0] = 1
Tg = rng.normal(size=D) + 1j * rng.normal(size=D)
Tg[8:] = 0
Tg /= np.linalg.norm(Tg)
path = tracking.Path(B.path_states(S, Tg, np.linspace(0, B.geodesic_parts(S, Tg)[1], 41)))
x = grape.random_start(N, rng).ravel()
args = (M, None, N, 0.4 / N, grape.joint(S), path, 50.0, 0.97, 200.0, 0.9999, 30.0, False)
f = lambda y: tracking.cost_grad(M, y, *args[2:])[0]  # noqa: E731
_, gx, _ = tracking.cost_grad(M, x, *args[2:])
idx = rng.choice(x.size, 12, replace=False)
fd = np.array([(f(x + 1e-6 * np.eye(x.size)[i]) - f(x - 1e-6 * np.eye(x.size)[i])) / 2e-6 for i in idx])
err = np.max(np.abs(fd - gx[idx])) / max(1e-12, np.max(np.abs(fd)))
check("adjoint gradient", err < 1e-5, f"max relative deviation from finite differences {err:.1e} (12 random variables)")
unused = [i for i in range(x.size) if i % 6 in (4, 5)]
check("unused variables inert", np.max(np.abs(gx[unused])) < 1e-12, "zero gradient for (rho_e, phi_e)")

# 5. tangential capacity of SNAP: only the displacement moves the vacuum, v = g |<1|chi>| / sqrt2 (as for CD)
import compute as CP  # noqa: E402

for name in ("fock2", "cat", "gkp", "binomial", "flip"):
    S, T = CP.transfer(name, 60)
    c0, th, chi = B.geodesic_parts(S, T)
    if name == "flip":
        print(f"      tangential capacity {name}: (start is |0_L>, not the vacuum; see the SM)")
        continue
    vv = g * abs(chi[1]) / math.sqrt(2)
    print(f"      tangential capacity {name}: v/omega = {vv / (g / math.sqrt(2)):.3f}  (|<1|chi>| = {abs(chi[1]):.3f})")
print("\nALL SNAP CHECKS PASSED" if ok_all else "\nSOME SNAP CHECKS FAILED")
