"""Rerun the numerical optimizations of the paper and write them to results/.

    python reproduce.py                 # everything, about two minutes
    python reproduce.py noise loop      # selected parts

The files have the same format as the published ones in data/.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import hambell as hb

SCENARIOS = {  # name: (figure, delta, number of measurements of party A, learning rate)
    "xxz_3x3": ("2(a)", 2, 3, 0.02), "xxz_4x3": ("2(b)", 2, 4, 0.02),
    "heisenberg_3x3": ("3(a)", 1, 3, 0.01), "heisenberg_4x3": ("3(b)", 1, 4, 0.01),
}


def angles(ma, *key):
    """Initial angles, uniform in [-pi, pi), from an independent random stream per key."""
    return np.random.default_rng(key).uniform(-np.pi, np.pi, 2 * (ma + 3))


def table(path, header, *columns):
    rows = (",".join(repr(v.item()) for v in row) for row in zip(*map(np.asarray, columns)))
    path.write_text("\n".join([header, *rows]) + "\n")


def ham2bell(out, seed, restarts):
    """Figs. 2(a,b) and 3: maximize the classical bound of a fixed Hamiltonian."""
    solutions, histories = {}, []
    for k, (name, (figure, delta, ma, lr)) in enumerate(SCENARIOS.items()):
        runs = [hb.ham_to_bell(hb.xxz(delta), angles(ma, seed, k, r), ma, lr=lr) for r in range(restarts)]
        bounds = [float(history.max()) for *_, history in runs]
        x, alpha, history = runs[np.argmax(bounds)]
        A, B = hb.bloch(x, ma)
        solutions[name] = dict(figure=figure, delta=delta, classical_bound=max(bounds),
                               A=A.tolist(), B=B.tolist(), alpha=alpha.tolist(), restarts=bounds)
        histories.append(history)
        print(f"{name:15s} best {max(bounds):.4f}   restarts", np.round(bounds, 3))
    (out / "ham_to_bell.json").write_text(json.dumps(solutions, indent=1) + "\n")
    table(out / "ham_to_bell_history.csv", "step," + ",".join(SCENARIOS), range(10_001), *histories)


def noise(out, seed, **_):
    """Fig. 4(a): minimize the quantum value of Eq. (4) on the noisy singlet."""
    alpha = hb.gisin(2)[0]
    p = np.arange(334) / 1000
    original = [(hb.noisy_singlet(q) * hb.xxz(2)).sum() for q in p]
    optimized = [hb.bell_to_ham(alpha, hb.noisy_singlet(q), angles(4, seed, 4, i))[1][-1]
                 for i, q in enumerate(p)]
    table(out / "noise_scan.csv", "p,original_quantum_value,optimized_quantum_value",
          p, original, optimized)
    print("noise scan      p = 0.010: original %.4f, optimized %.4f" % (original[10], optimized[10]))


def loop(out, seed, **_):
    """Fig. 4(b): alternate the two optimizations at p = 0.01."""
    alpha, c = hb.gisin(2)[0], hb.noisy_singlet(0.01)
    quantum, classical = hb.alternate(alpha, c, angles(4, seed, 5), cycles=4)
    quantum = np.append((c * hb.xxz(2)).sum(), quantum)
    table(out / "loop.csv", "cycle,quantum_value,classical_bound", range(5), quantum, classical)
    print("loop            quantum", np.round(quantum, 4), " classical", np.round(classical, 4))


if __name__ == "__main__":
    parts = {"ham2bell": ham2bell, "noise": noise, "loop": loop}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", nargs="*", metavar="part", help="ham2bell, noise, loop (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=8, help="random restarts per scenario")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if set(args.parts) - set(parts):
        parser.error("unknown part; choose from " + ", ".join(parts))
    args.out.mkdir(parents=True, exist_ok=True)
    for part in args.parts or parts:
        parts[part](args.out, args.seed, restarts=args.restarts)
