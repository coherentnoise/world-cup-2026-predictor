# Model overview

The project uses this forecasting pipeline:

```text
historical results → Elo ratings → expected goals → scoreline probabilities → tournament simulation
```

## Elo ratings

Elo ratings summarize team strength. Teams gain rating points when they outperform expectation and lose points when they underperform expectation.

The update depends on:

- opponent strength
- match result
- goal margin
- tournament importance
- home advantage

## Expected goals

The baseline expected-goals model is:

```python
log_mu = intercept + elo_coef * rating_gap_per_100 + host_coef * host_indicator
```

This is intentionally simple. A student extension is to estimate these coefficients from historical data.

## Dixon–Coles correction

Independent Poisson models usually misrepresent low football scores. Dixon–Coles modifies the probabilities of:

```text
0-0, 0-1, 1-0, 1-1
```

The correction is controlled by `DIXON_COLES_RHO`.

## Monte Carlo simulation

The model simulates many tournaments and counts outcomes:

```text
P(team wins) = number of simulated wins / number of simulations
```
