"""Optimizable mappings between Bell inequalities and two-qubit Hamiltonians.

Core routines of "Improved certification of Bell-operator correlations via
optimization", Phys. Rev. A (2026), https://doi.org/10.1103/sglx-xq8z.

A Bell expression  I = sum_ab alpha[a, b] A_a B_b  with qubit observables
A_a = n_a . sigma and B_b = m_b . sigma has the Bell operator

    H = sum_ij h[i, j] sigma_i (x) sigma_j,        h = A^T alpha B,

where the rows of A (ma x 3) and B (mb x 3) are the Bloch vectors n_a and m_b
and i, j run over x, y, z. Flattening alpha and h row by row gives Eq. (3),
T(theta) alpha = h, with the 9 x (ma mb) matrix T = kron(A, B)^T.

bloch, mapping, hamiltonian, bell_coefficients and classical_bound also accept
arrays with (equal) leading batch dimensions.
"""
import numpy as np


def xxz(delta):
    """h of H = 4/sqrt(3) (XX + YY + delta ZZ): Eq. (7) for delta = 2, Eq. (9) for delta = 1."""
    return 4 / np.sqrt(3) * np.diag([1.0, 1.0, delta])


def gisin(delta):
    """Variant of Gisin's elegant inequality: alpha of Eq. (4), A and B of Eqs. (5), (6)."""
    alpha = np.array([[1, 1, delta], [1, -1, -delta], [-1, 1, -delta], [-1, -1, delta]], float)
    A = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    return alpha, A, np.eye(3)


def bloch(x, ma):
    """Angles x = (theta_0, phi_0, theta_1, phi_1, ...) -> Bloch vectors A, B of Eq. (A2).

    The first ma pairs belong to party A, the remaining ones to party B.
    """
    theta, phi = x[..., 0::2], x[..., 1::2]
    n = np.stack([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)], -1)
    return n[..., :ma, :], n[..., ma:, :]


def mapping(A, B):
    """The matrix T(theta) of Eqs. (3) and (A4)."""
    return np.einsum("...ai,...bj->...ijab", A, B).reshape(*A.shape[:-2], 9, -1)


def hamiltonian(alpha, A, B):
    """Bell-operator coefficients h = T alpha, as a 3 x 3 array."""
    return np.swapaxes(A, -1, -2) @ alpha @ B


def bell_coefficients(h, A, B):
    """Minimum-norm solution alpha of T alpha = h, as an ma x mb array.

    Because T = kron(A, B)^T, its pseudoinverse factorizes: alpha = (A^T)^+ h B^+.
    The solution is unique for ma = mb = 3 and full-rank T.
    """
    return np.linalg.pinv(np.swapaxes(A, -1, -2)) @ h @ np.linalg.pinv(B)


def classical_bound(alpha):
    """Exact local bound: minimum of sum_ab alpha[a, b] a_a b_b over a_a, b_b = +-1."""
    ma = alpha.shape[-2]
    a = 1 - 2 * (np.arange(2 ** ma)[:, None] >> np.arange(ma) & 1)  # all strategies of A
    return -np.abs(a @ alpha).sum(-1).max(-1)                       # B answers optimally


def noisy_singlet(p):
    """Correlators c[i, j] = <sigma_i (x) sigma_j> of the noisy circuit of Fig. 5.

    Each depolarizing channel N_p of Eq. (11) scales a Pauli operator by
    q = 1 - 4p. XX and YY pass through five channels, ZZ through four.
    """
    q = 1 - 4 * p
    return -np.diag([q ** 5, q ** 5, q ** 4])


def adam(f, x, steps, lr, eps, maximize=False):
    """Adam(0.9, 0.999) driven by forward finite differences of a batched function f.

    Returns the best visited point and the history f(x_0), ..., f(x_steps).
    """
    x = np.array(x, float)
    sign = 1 if maximize else -1
    shifts = np.vstack([np.zeros(x.size), eps * np.eye(x.size)])
    m, v, best, f_best, history = 0, 0, x.copy(), -np.inf, []
    for k in range(steps + 1):
        fx = f(x + shifts)
        history.append(fx[0])
        if sign * fx[0] > f_best:
            best, f_best = x.copy(), sign * fx[0]
        if k < steps:
            g = (fx[1:] - fx[0]) / eps
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            x += sign * lr * m / (1 - 0.9 ** (k + 1)) / (np.sqrt(v / (1 - 0.999 ** (k + 1))) + 1e-8)
    return best, np.array(history)


def ham_to_bell(h, x, ma, steps=10_000, lr=0.02, eps=1e-5):
    """Fix the Hamiltonian h; maximize the classical bound over the angles x.

    Returns the best angles, the Bell coefficients alpha, and the bound history.
    """
    x, history = adam(lambda x: classical_bound(bell_coefficients(h, *bloch(x, ma))),
                      x, steps, lr, eps, maximize=True)
    A, B = bloch(x, ma)
    alpha = bell_coefficients(h, A, B)
    if not np.allclose(hamiltonian(alpha, A, B), h, rtol=0, atol=1e-8):
        raise ValueError("these measurement directions cannot represent the Hamiltonian")
    return x, alpha, history


def bell_to_ham(alpha, c, x, steps=2_000, lr=0.01, eps=1e-4):
    """Fix the inequality alpha and the correlators c; minimize E = c . T(x) alpha.

    Returns the best angles and the history of E.
    """
    return adam(lambda x: (c * hamiltonian(alpha, *bloch(x, len(alpha)))).sum((-1, -2)),
                x, steps, lr, eps)


def alternate(alpha, c, x, cycles, steps=(10_000, 20_000), lr=0.01, eps=1e-4):
    """Alternate bell_to_ham and ham_to_bell on fixed correlators c, Fig. 4(b).

    Returns the quantum value after each bell_to_ham step, and the classical
    bound of the initial inequality followed by that after each ham_to_bell step.
    """
    ma = len(alpha)
    quantum, classical = [], [classical_bound(alpha)]
    for _ in range(cycles):
        x, energy = bell_to_ham(alpha, c, x, steps[0], lr, eps)
        quantum.append(energy.min())
        x, alpha, bound = ham_to_bell(hamiltonian(alpha, *bloch(x, ma)), x, ma, steps[1], lr, eps)
        classical.append(bound.max())
    return np.array(quantum), np.array(classical)
