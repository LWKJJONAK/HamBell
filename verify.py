"""Check data/ against the numbers stated in the paper.

    python verify.py

If reproduce.py has been run, results/ is compared with data/ as well.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import hambell as hb

DATA, RESULTS = Path(__file__).parent / "data", Path(__file__).parent / "results"
failed = False


def check(statement, ok):
    global failed
    failed |= not ok
    print("ok  " if ok else "FAIL", statement)


def table(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def solutions(path):
    return {name: {k: np.array(v) if isinstance(v, list) else v for k, v in s.items()}
            for name, s in json.loads(path.read_text()).items()}


def face_dimension(alpha):
    """Dimension of the face of the correlation polytope on which the bound is attained."""
    points = np.array([np.outer(a, b).ravel() for a in itertools.product([1, -1], repeat=alpha.shape[0])
                       for b in itertools.product([1, -1], repeat=alpha.shape[1])])
    values = points @ alpha.ravel()
    face = points[np.isclose(values, values.min())]
    return np.linalg.matrix_rank(face[1:] - face[0])


def simulated_singlet(p):
    """Density-matrix simulation of the circuit of Fig. 5 with the channel of Eq. (11)."""
    X, Y, Z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
    first, second = (lambda U: np.kron(U, np.eye(2))), (lambda U: np.kron(np.eye(2), U))

    def gate(rho, U):
        return U @ rho @ U.conj().T

    def noise(rho, qubit):
        return (1 - 3 * p) * rho + p * sum(gate(rho, qubit(P)) for P in (X, Y, Z))

    rho = np.diag([1, 0, 0, 0]).astype(complex)
    rho = noise(gate(rho, first((X + Z) / np.sqrt(2))), first)                  # H
    rho = noise(noise(gate(rho, np.eye(4)[[0, 1, 3, 2]]), first), second)       # CNOT
    rho = noise(noise(gate(rho, np.kron(Z, X)), first), second)                 # Z, X
    return np.array([[np.trace(rho @ np.kron(P, Q)).real for Q in (X, Y, Z)] for P in (X, Y, Z)])


# Variant of Gisin's elegant inequality
alpha, A, B = hb.gisin(2)
check("Eqs. (4)-(6) give the Hamiltonians of Eqs. (7) and (9)",
      np.allclose(hb.hamiltonian(alpha, A, B), hb.xxz(2))
      and np.allclose(hb.hamiltonian(hb.gisin(1)[0], A, B), hb.xxz(1)))
check("Eq. (4): classical bound -2|D| - |D+2| - |D-2|",
      all(np.isclose(hb.classical_bound(hb.gisin(d)[0]), -2 * abs(d) - abs(d + 2) - abs(d - 2))
          for d in np.linspace(-4, 4, 33)))
check("T alpha = h; minimum-norm solution (A^T)^+ h B^+",
      np.allclose(hb.mapping(A, B) @ alpha.ravel(), hb.xxz(2).ravel())
      and np.allclose(np.linalg.pinv(hb.mapping(A, B)) @ hb.xxz(2).ravel(),
                      hb.bell_coefficients(hb.xxz(2), A, B).ravel()))
pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
energies = np.linalg.eigvalsh(np.einsum("ij,iab,jcd->acbd", hb.xxz(2), pauli, pauli).reshape(4, 4))
check("Eq. (7): ground-state energy -9.24, attained by the singlet",
      np.isclose(energies[0], -16 / np.sqrt(3)) and np.isclose((hb.noisy_singlet(0) * hb.xxz(2)).sum(), energies[0]))

# Optimized inequalities, Figs. 2(a,b) and 3
published, history = solutions(DATA / "ham_to_bell.json"), table(DATA / "ham_to_bell_history.csv")
quoted = {"xxz_3x3": -7.39, "xxz_4x3": -6.56, "heisenberg_3x3": -5.58, "heisenberg_4x3": -5.17}
for name, s in published.items():
    h = hb.xxz(s["delta"])
    check(f"{name}: T alpha = h, minimum norm, classical bound {quoted[name]}",
          np.allclose(np.linalg.norm(s["A"], axis=1), 1) and np.allclose(np.linalg.norm(s["B"], axis=1), 1)
          and np.allclose(hb.hamiltonian(s["alpha"], s["A"], s["B"]), h, rtol=0, atol=1e-12)
          and np.allclose(hb.bell_coefficients(h, s["A"], s["B"]), s["alpha"], rtol=0, atol=1e-12)
          and np.isclose(hb.classical_bound(s["alpha"]), s["classical_bound"], rtol=0, atol=1e-12)
          and history[name].max() == s["classical_bound"] and round(s["classical_bound"], 2) == quoted[name])
check("coefficients printed in the text and in Eq. (8)",
      np.array_equal(np.round(published["xxz_3x3"]["alpha"], 3),
                     [[-1.864, 3.049, 2.272], [0.070, -1.822, 1.766], [-1.932, -1.413, -0.563]])
      and np.array_equal(np.round(published["xxz_4x3"]["alpha"], 3),
                         [[-0.001, 1.625, -1.627], [-1.623, -0.008, 1.636],
                          [1.633, -0.001, 1.638], [0.006, -1.633, -1.649]]))

# Analytical inequality, Eq. (10)
alpha = np.array([[0, 1, -1], [-1, 0, 1], [1, 0, 1], [0, -1, -1]], float)
A = np.array([[0, 1, 1], [-1, 0, -1], [1, 0, -1], [0, -1, 1]]) / np.sqrt(2)
B = np.diag([1.0, 1.0, -1.0])
chsh = alpha * [[1], [0], [0], [1]]  # A_0, A_3 with B_1, B_2; the rest is A_1, A_2 with B_0, B_2
check("Eq. (10): bound -4, operator sqrt(2) (XX + YY + 2 ZZ), rescaled bound -6.53",
      hb.classical_bound(alpha) == -4
      and np.allclose(hb.hamiltonian(4 / np.sqrt(6) * alpha, A, B), hb.xxz(2))
      and round(hb.classical_bound(4 / np.sqrt(6) * alpha), 2) == -6.53)
check("Eq. (10) is Eq. (8) with rounded coefficients and measurements",
      np.array_equal(np.round(published["xxz_4x3"]["alpha"] * np.sqrt(6) / 4), alpha)
      and np.allclose(published["xxz_4x3"]["A"], A, atol=0.01)
      and np.allclose(published["xxz_4x3"]["B"], B, atol=0.01))
check("Eq. (10): sum of two CHSH inequalities; face dimension 10, facet dimension 11",
      hb.classical_bound(chsh) == hb.classical_bound(alpha - chsh) == -2
      and face_dimension(alpha) == 10 and face_dimension(chsh) == 11)

# Noise robustness, Fig. 4(a)
scan = table(DATA / "noise_scan.csv")
p, original, optimized = (scan[k] for k in scan.dtype.names)
check("Fig. 5: correlators -(1-4p)^5 for XX, YY and -(1-4p)^4 for ZZ",
      all(np.allclose(simulated_singlet(q), hb.noisy_singlet(q)) for q in (0, 0.01, 0.1, 0.3)))
check("Fig. 4(a): original quantum values",
      np.allclose(original, [(hb.noisy_singlet(q) * hb.xxz(2)).sum() for q in p]))
check("Fig. 4(a): violation lost at p = 0.008 (original) and p = 0.012 (optimized)",
      np.array_equal(p[original < -8], p[:8]) and np.array_equal(p[optimized < -8], p[:12]))

# Iterative optimization, Fig. 4(b)
loop = table(DATA / "loop.csv")
quantum, classical = loop["quantum_value"], loop["classical_bound"]
check("Fig. 4(b): starts from the original inequality at p = 0.01",
      np.isclose(quantum[0], original[10]) and classical[0] == -8)
check("Fig. 4(b): quantum value below the classical bound after every cycle",
      np.all(quantum[1:] < classical[1:]))

# Rerun
if (RESULTS / "ham_to_bell.json").exists():
    for name, s in solutions(RESULTS / "ham_to_bell.json").items():
        check(f"rerun {name}: {s['classical_bound']:.4f} (published {published[name]['classical_bound']:.4f})",
              abs(s["classical_bound"] - published[name]["classical_bound"]) < 0.05)
if (RESULTS / "noise_scan.csv").exists():
    difference = np.abs(table(RESULTS / "noise_scan.csv")["optimized_quantum_value"] - optimized)
    check(f"rerun noise scan: median deviation {np.median(difference):.0e}", np.median(difference) < 1e-3)
if (RESULTS / "loop.csv").exists():
    rerun = table(RESULTS / "loop.csv")
    difference = max(np.abs(rerun["quantum_value"] - quantum).max(),
                     np.abs(rerun["classical_bound"] - classical).max())
    check(f"rerun loop: largest deviation {difference:.3f}", difference < 0.1)

sys.exit(failed)
