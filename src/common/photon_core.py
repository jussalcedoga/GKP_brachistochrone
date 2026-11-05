# VENDORED, UNCHANGED BELOW THE NEXT LINE: core numerics of the note "Photon-optimal paths for GKP state preparation and
# logical operations" (context/ hand-off package, scripts/core.py; shipped under the name requirements.txt).
# Kept verbatim so that every analytic number of the manuscript is produced by the same code as the note.
"""Core numerics for 'Photon-optimal paths for GKP state preparation and logical operations'.
Real even-parity sector, unit sphere S^{d-1}; weight N = n  (or n + ell (1 - Pi) for the code-space penalty)."""
import numpy as np
from scipy.special import gammaln, hyp2f1
from scipy.optimize import minimize
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

trapz = getattr(np, 'trapezoid', None) or np.trapz     # NumPy >= 2 renamed trapz -> trapezoid
G_D = 7.5e6                      # g_d (s^-1), from the 200 ns squeezing gate of the proposal
OMEGA = G_D / np.sqrt(2)         # energy-variance rate: T = L / OMEGA

# ----------------------------------------------------------------------------- states
def _coh(g, D):                  # Fock amplitudes of the coherent state |g>
    n = np.arange(D); c = np.zeros(D, complex)
    if abs(g) < 1e-14: c[0] = 1; return c
    return np.exp(-abs(g)**2/2 + n*np.log(abs(g)) - 0.5*gammaln(n+1) + 1j*n*np.angle(g))

def gkp(Delta, mu, Dbig=260, M=10):
    """finite-energy square-lattice GKP code word: e^{-Delta^2 n} sum_{m=mu (2), l} e^{-i pi m l/2} D(m a + l b)|0>, a=sqrt(pi/2), b=i a"""
    a = np.sqrt(np.pi/2); psi = np.zeros(Dbig, complex)
    for m in range(-M, M+1):
        if m % 2 != mu: continue
        for l in range(-M, M+1): psi += np.exp(-1j*np.pi*m*l/2) * _coh(m*a + 1j*l*a, Dbig)
    psi *= np.exp(-Delta**2*np.arange(Dbig)); return psi/np.linalg.norm(psi)

def even_real(psi, D):
    """truncate to Fock dimension D, fix the global phase, return (photon numbers, real normalized vector) of the even sector"""
    v = psi[:D]/np.linalg.norm(psi[:D]); v = v*np.exp(-1j*np.angle(v[0]))
    assert np.max(abs(v[1::2])) < 1e-9 and np.max(abs(v.imag)) < 1e-9, 'target is not real/even'
    t = v[0::2].real; return np.arange(0, D, 2).astype(float), t/np.linalg.norm(t)

def target(kind, par, D):
    if kind == 'GKP': return even_real(gkp(par, 0), D)
    if kind == 'Cat':                                        # even cat ~ |alpha> + |-alpha>
        n = np.arange(D); c = np.zeros(D); c[0::2] = np.exp(-par**2/2 + n[0::2]*np.log(par) - 0.5*gammaln(n[0::2]+1)); return even_real(c.astype(complex), D)
    if kind == 'Fock': c = np.zeros(D, complex); c[int(par)] = 1; return even_real(c, D)

class Prep:
    """preparation |0> -> |T>: geodesic quantities (Prop. 3)"""
    def __init__(self, kind, par, D):
        self.nv, self.uT = target(kind, par, D); d = len(self.nv); self.u0 = np.eye(d)[0]; self.N = np.diag(self.nv)
        self.c0 = self.u0 @ self.uT; self.th = np.arccos(self.c0); chi = self.uT - self.c0*self.u0; self.chi = chi/np.linalg.norm(chi)
        self.n_chi = self.chi @ (self.nv*self.chi); self.sig = np.sqrt(self.chi @ (self.nv**2*self.chi) - self.n_chi**2)
    def geo(self, s): s = np.atleast_1d(s); return np.cos(s)[:, None]*self.u0 + np.sin(s)[:, None]*self.chi
    def J_geo(self): th = self.th; return self.n_chi*(th/2 - np.sin(2*th)/4)

class Logical:
    """logical flip |0_L> -> |1_L> at Fock cutoff D (Prop. 5)"""
    def __init__(self, Delta=0.4, D=52):
        nv, self.u0 = even_real(gkp(Delta, 0), D); _, uT = even_real(gkp(Delta, 1), D); self.nv = nv
        self.uT = uT if self.u0 @ uT >= 0 else -uT; self.c = self.u0 @ self.uT; self.th = np.arccos(self.c)
        Q = np.linalg.qr(np.array([self.u0, self.uT]).T)[0]; self.Pi = Q @ Q.T; self.d = len(nv)
        n0, n1, N01 = self.u0 @ (nv*self.u0), self.uT @ (nv*self.uT), self.u0 @ (nv*self.uT); self.n0, self.n1, self.N01 = n0, n1, N01
    def Nl(self, ell): return np.diag(self.nv) + ell*(np.eye(self.d) - self.Pi)
    def geo(self, s): s = np.atleast_1d(s); return (np.sin(self.th - s)[:, None]*self.u0 + np.sin(s)[:, None]*self.uT)/np.sin(self.th)
    def J_geo(self):
        th = self.th; return ((self.n0 + self.n1)*(th/2 - np.sin(2*th)/4) + self.N01*(np.sin(th) - th*np.cos(th)))/np.sin(th)**2
    def detour(self, m=600):
        e0 = np.eye(self.d)[0]; a = np.sign(self.u0 @ e0)*self.u0; b = np.sign(self.uT @ e0)*self.uT; ta, tb = np.arccos(abs(a @ e0)), np.arccos(abs(b @ e0)); x = np.linspace(0, 1, m)[:, None]
        leg = lambda p, q, t: (np.sin((1-x)*t)*p + np.sin(x*t)*q)/np.sin(t)
        return np.vstack([leg(a, e0, ta), leg(e0, b, tb)[1:]])

# ----------------------------------------------------------------------------- exact exposure of piecewise-geodesic paths
def _arcs(U, N):
    NU = U @ N; a = np.sum(U*NU, 1); c = np.sum(U[:-1]*NU[1:], 1); cs = np.clip(np.sum(U[:-1]*U[1:], 1), -1, 1); th = np.arccos(cs)
    return NU, a, c, th

def exposure(U, N):
    """J = int <N> ds along the great-circle arcs through the rows of U (exact arc integral, Prop. 5 formula)"""
    NU, a, c, th = _arcs(U, N); s = np.sin(th); g1 = th/2 - np.sin(2*th)/4; g2 = s - th*np.cos(th)
    return float(np.sum(((a[:-1] + a[1:])*g1 + c*g2)/s**2))

def length(U): return float(np.sum(np.arccos(np.clip(np.sum(U[:-1]*U[1:], 1), -1, 1))))

def _cost_grad(x, N, u0, uT, K):
    X = x.reshape(K, -1); nr = np.linalg.norm(X, axis=1, keepdims=True); U = np.vstack([u0, X/nr, uT])
    NU, a, c, th = _arcs(U, N); s = np.sin(th); co = np.cos(th); g1 = th/2 - np.sin(2*th)/4; g2 = s - th*co
    S = (a[:-1] + a[1:])*g1 + c*g2; J = np.sum(S/s**2)
    Aa = g1/s**2; Ac = g2/s**2; dS = (a[:-1] + a[1:])*s**2 + c*th*s                       # g1' = sin^2, g2' = th sin
    At = dS/s**2 - 2*co*S/s**3
    gu = 2*Aa[:, None]*NU[:-1] + Ac[:, None]*NU[1:] - (At/s)[:, None]*U[1:]
    gv = 2*Aa[:, None]*NU[1:] + Ac[:, None]*NU[:-1] - (At/s)[:, None]*U[:-1]
    g = np.zeros_like(U); g[:-1] += gu; g[1:] += gv; g = g[1:-1]
    g = (g - np.sum(g*U[1:-1], 1, keepdims=True)*U[1:-1])/nr; return J, g.ravel()

def resample(P, K):
    """K+2 nodes at equal arclength along a dense path"""
    seg = np.r_[0, np.cumsum(np.arccos(np.clip(np.sum(P[:-1]*P[1:], 1), -1, 1)))]; t = np.linspace(0, seg[-1], K+2); out = []
    for ti in t:
        i = min(max(np.searchsorted(seg, ti) - 1, 0), len(P) - 2); w = (ti - seg[i])/(seg[i+1] - seg[i]); v = (1-w)*P[i] + w*P[i+1]; out.append(v/np.linalg.norm(v))
    return np.array(out)

def optimize_path(N, u0, uT, starts, K=30, maxiter=4000):
    """minimize J over K free nodes; `starts` = list of (K+2)-node initial paths. Returns best (U, J)."""
    best = None
    for P in starts:
        r = minimize(_cost_grad, P[1:-1].ravel(), args=(N, u0, uT, K), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxfun=maxiter*2, ftol=1e-15, gtol=1e-10))
        X = r.x.reshape(K, -1); U = np.vstack([u0, X/np.linalg.norm(X, axis=1, keepdims=True), uT])
        if best is None or r.fun < best[1]: best = (U, float(r.fun))
    return best

def noisy_geodesic(geo_nodes, seed, scale=0.05):
    rng = np.random.default_rng(seed); P = geo_nodes.copy(); P[1:-1] += scale*rng.standard_normal(P[1:-1].shape); P /= np.linalg.norm(P, axis=1, keepdims=True); return P

def dense(U, sub=41):
    """sample the piecewise great-circle path: returns (states, arclength)"""
    out, L = [U[0]], [0.0]; acc = 0.0
    for k in range(len(U)-1):
        u, v = U[k], U[k+1]; th = np.arccos(np.clip(u @ v, -1, 1)); t = np.linspace(0, th, sub)[1:, None]
        out.append((np.sin(th - t)*u + np.sin(t)*v)/np.sin(th)); L.extend(acc + t[:, 0]); acc += th
    return np.vstack(out), np.array(L)

def hausdorff(A, B, step=3):
    A, B = A[::step], B[::step]; f = lambda X, Y: np.arccos(np.clip(np.max(np.abs(X @ Y.T), 1), 0, 1)).max(); return float(max(f(A, B), f(B, A)))
def dist_to(P, G): return np.arccos(np.clip(np.max(np.abs(P @ G.T), 1), 0, 1))

# ----------------------------------------------------------------------------- analytic solutions
def _build(G, B, a):
    """psi = cos|a| psi_g + sin|a| B a/|a| (a: rows of transverse amplitudes)"""
    na = np.linalg.norm(a, axis=1); out = np.cos(na)[:, None]*G
    nz = na > 1e-14; out[nz] += (np.sin(na[nz])/na[nz])[:, None]*(a[nz] @ B.T); return out

def closed_form_path(P, m=800):
    """Prop. 4: single-mode (|xi>) closed-form first-order solution for preparation from the vacuum"""
    th = P.th; xi = (P.nv*P.chi - P.n_chi*P.chi)/P.sig; nu = xi @ (P.nv*xi); beta = 2*(nu*P.n_chi - 2*P.sig**2)/P.n_chi**2; mu = 0.5*(1 + np.sqrt(1 + 4*beta))
    u1 = lambda s: np.sin(s)**mu*hyp2f1((mu+2)/2, (mu-2)/2, mu+0.5, np.sin(s)**2)
    s = np.linspace(0, th, m); si = s[1:-1]; a = np.zeros(m); a[1:-1] = 2*P.sig/(P.n_chi*(2-beta))*(np.sin(si) - np.sin(th)**2/np.sin(si)*u1(si)/u1(th))
    return np.cos(a)[:, None]*P.geo(s) + np.sin(a)[:, None]*xi, dict(s=s, a=a, beta=beta, mu=mu)

def first_order_prep(P, Mg=600):
    """Prop. 2 for preparation (vector ODE). With a = u/sin s:  u'' + (4 - C/sin^2 s) u = 2 v/n_chi,  u(0)=u(theta0)=0"""
    Bp = np.linalg.svd(np.vstack([P.u0, P.chi]))[2][2:].T; m = Bp.shape[1]; v = Bp.T @ (P.nv*P.chi); Mm = Bp.T @ (P.nv[:, None]*Bp)
    C = 2*(Mm - 2*np.outer(v, v)/P.n_chi)/P.n_chi; s = np.linspace(0, P.th, Mg+1); h = s[1]; si = s[1:-1]
    A = sp.kron(sp.diags([1., -2., 1.], [-1, 0, 1], shape=(Mg-1, Mg-1))/h**2, sp.eye(m)) + sp.block_diag([4*np.eye(m) - C/np.sin(x)**2 for x in si])
    u = spsolve(A.tocsc(), np.tile(2*v/P.n_chi, Mg-1)).reshape(Mg-1, m); a = np.zeros((Mg+1, m)); a[1:-1] = u/np.sin(si)[:, None]
    return _build(P.geo(s), Bp, a)

def first_order_logical(L, ell, Mg=400):
    """Prop. 2 for the logical flip with penalty ell (vector ODE, finite differences)"""
    th = L.th; s = np.linspace(0, th, Mg+1); h = s[1]; G = L.geo(s); dG = (-np.cos(th - s)[:, None]*L.u0 + np.cos(s)[:, None]*L.uT)/np.sin(th)
    Bp = np.linalg.svd(np.vstack([L.u0, L.uT]))[2][2:].T; m = Bp.shape[1]; nv = L.nv
    w = np.sum(G**2*nv, 1); wp = 2*np.sum(dG*nv*G, 1); sg = (G*nv) @ Bp; M = Bp.T @ (nv[:, None]*Bp) + ell*np.eye(m); i = slice(1, Mg)
    p = wp[i]/w[i]; Cb = sp.block_diag([3*np.eye(m) - 2*M/w[k] + 4*np.outer(sg[k], sg[k])/w[k]**2 for k in range(1, Mg)])
    D2 = sp.diags([1., -2., 1.], [-1, 0, 1], shape=(Mg-1, Mg-1))/h**2; D1 = sp.diags([-1., 1.], [-1, 1], shape=(Mg-1, Mg-1))/(2*h)
    A = sp.kron(D2, sp.eye(m)) + sp.diags(np.repeat(p, m)) @ sp.kron(D1, sp.eye(m)) + Cb
    a = np.zeros((Mg+1, m)); a[1:-1] = spsolve(A.tocsc(), (2*sg[i]/w[i][:, None]).ravel()).reshape(Mg-1, m)
    return _build(G, Bp, a)

# ----------------------------------------------------------------------------- analytic lower bound (Lemma 3, Theorem 2)
def min_photon_curve(nv, T, lam=np.logspace(-9, 9, 8000)):
    W = (T**2)[None, :]/(nv[None, :] + lam[:, None]); S1 = W.sum(1); S2 = (W/(nv[None, :] + lam[:, None])).sum(1)
    F = S1**2/S2; nb = (nv[None, :]*W/(nv[None, :] + lam[:, None])).sum(1)/S2; o = np.argsort(F); return F[o], nb[o]

def lower_bound(nv, T):
    """B(T) = int_0^{theta_0} m(cos^2 r) dr, theta_0 = arccos|<0|T>|"""
    F, nb = min_photon_curve(nv, T); r = np.linspace(0, np.arccos(abs(T[0])), 4001); return float(trapz(np.interp(np.cos(r)**2, F, nb, left=0.0), r))

# ----------------------------------------------------------------------------- route test (Sec. 6.2)
def apply_route(U, v, nv, sub=40):
    """apply the rank-2 generators that make |0_L> follow the arcs of U to another input v; return final vector, per-sample states and times (s)"""
    out, t, acc = [v], [0.0], 0.0
    for k in range(len(U)-1):
        u, w = U[k], U[k+1]; th = np.arccos(np.clip(u @ w, -1, 1)); phi = (w - (u @ w)*u)/np.sin(th); Pl = np.outer(u, u) + np.outer(phi, phi); Jm = np.outer(phi, u) - np.outer(u, phi); v0 = v
        for j in range(1, sub+1):
            a = th*j/sub; out.append(v0 + (np.cos(a) - 1)*(Pl @ v0) + np.sin(a)*(Jm @ v0)); t.append((acc + a)/OMEGA)
        v = out[-1]; acc += th
    return np.array(out), np.array(t)

def min_photon_numeric(nv, T, F, starts=4):
    """brute-force min <n> subject to |<T|psi>|^2 = F (SLSQP), to check the analytic curve m(F) of Lemma 3"""
    cons = [{'type': 'eq', 'fun': lambda x: x @ x - 1}, {'type': 'eq', 'fun': lambda x: T @ x - np.sqrt(F)}]; best = np.inf
    for s in range(starts):
        x0 = np.sqrt(F)*T + 0.3*np.random.default_rng(s).standard_normal(len(T)); x0 /= np.linalg.norm(x0)
        r = minimize(lambda x: x @ (nv*x), x0, jac=lambda x: 2*nv*x, constraints=cons, method='SLSQP', options=dict(ftol=1e-15, maxiter=1000))
        if r.success: best = min(best, r.fun)
    return best
