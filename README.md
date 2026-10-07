# HamBell

Code and data for the numerical results of

> W. Li, *et al.*, **Improved certification of Bell-operator correlations via optimization**,
> Phys. Rev. A (2026), [doi:10.1103/sglx-xq8z](https://doi.org/10.1103/sglx-xq8z).

| File | Content |
| --- | --- |
| `hambell.py` | Mapping $\mathbf T(\boldsymbol\theta)$, classical bound, noise model, and the three optimizations |
| `reproduce.py` | Reruns all optimizations and writes `results/` |
| `verify.py` | Checks `data/`, and `results/` if present, against the numbers in the paper |
| `data/` | Data of Figs. 2(a,b), 3 and 4 |

## Usage

Python 3.10+ with NumPy.

```sh
python verify.py
python reproduce.py    # about two minutes
```

## Conventions

- Bell inequality $\sum_{ab}\alpha_{ab}A_aB_b\ge\beta_C$; a value below $\beta_C$ is a violation.
- $A_a=\vec n_a\cdot\vec\sigma$ with $\vec n=(\cos\theta,\ \sin\theta\cos\phi,\ \sin\theta\sin\phi)$.
  The rows of `A` and `B` are these unit vectors.
- Bell operator $\sum_{ij}h_{ij}\,\sigma_i\otimes\sigma_j$ with $h=A^{\mathsf T}\alpha B$, i.e.
  $\mathbf T\vec\alpha=\vec h$ with $\mathbf T=(A\otimes B)^{\mathsf T}$ and row-major $\vec\alpha$, $\vec h$.
- For fixed $h$, $\alpha=(A^{\mathsf T})^{+}h\,B^{+}$ is the minimum-norm solution.
- Quantum value $E=\sum_{ij}c_{ij}h_{ij}$ with $c_{ij}=\langle\sigma_i\otimes\sigma_j\rangle$.
- `xxz` and `heisenberg` denote $\frac{4}{\sqrt3}(XX+YY+\Delta ZZ)$ with $\Delta=2$ and $\Delta=1$;
  `3x3` and `4x3` denote the $(3,2,3,2)$ and $(4,2,3,2)$ scenarios.

## Optimization

Adam(0.9, 0.999) on forward finite-difference gradients; initial angles uniform in $[-\pi,\pi)$.

| | Function | Steps | Learning rate | Difference step |
| --- | --- | ---: | ---: | ---: |
| Fig. 2(a,b) | `ham_to_bell` | $10^4$ | 0.02 | $10^{-5}$ |
| Fig. 3 | `ham_to_bell` | $10^4$ | 0.01 | $10^{-5}$ |
| Fig. 4(a) | `bell_to_ham` | $2\times10^3$ | 0.01 | $10^{-4}$ |
| Fig. 4(b) | `alternate` | $10^4$, then $2\times10^4$ per cycle | 0.01 | $10^{-4}$ |

`reproduce.py` uses eight random restarts for Figs. 2 and 3 and keeps the best.

## Data

| File | Figure | Content |
| --- | --- | --- |
| `ham_to_bell.json` | 2(a,b), 3 | Optimized inequalities: `A`, `B`, `alpha`, `classical_bound` |
| `ham_to_bell_history.csv` | 2(a,b), 3 | Classical bound at each step |
| `noise_scan.csv` | 4(a) | Original and optimized quantum values against the noise strength $p$ |
| `loop.csv` | 4(b) | Quantum value and classical bound after each cycle at $p=0.01$ |

## Results

| Classical bound | Paper | Rerun |
| --- | ---: | ---: |
| $\Delta=2$, $(3,2,3,2)$ | $-7.39$ | $-7.37$ |
| $\Delta=2$, $(4,2,3,2)$ | $-6.56$ | $-6.55$ |
| $\Delta=1$, $(3,2,3,2)$ | $-5.58$ | $-5.58$ |
| $\Delta=1$, $(4,2,3,2)$ | $-5.17$ | $-5.17$ |

The optimization is stochastic; last digits vary with the seed and the platform.

## License

Code: [MIT](LICENSE). Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
