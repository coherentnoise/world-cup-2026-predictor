# Dixon–Coles correction

Independent Poisson assumes each team's goals are independent.

Dixon–Coles modifies only low scores:

```text
0-0, 0-1, 1-0, 1-1
```

If `rho = 0`, the model is independent Poisson.

Negative `rho` usually boosts 0-0 and 1-1 and reduces 1-0 and 0-1.

Estimate rho with:

```bash
python scripts/estimate_dixon_coles_rho.py --years-back 10
```
