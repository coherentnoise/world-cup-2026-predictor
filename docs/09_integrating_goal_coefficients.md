# Integrating fitted goal coefficients into the predictor

After running `estimate_elo_goals_relation.py`, update the predictor scripts so they no longer use hand-tuned expected-goals coefficients.

## 1. Add coefficient settings

Near the other settings in `world_cup_predictor_2026.py` and `knockout_stage_predictor_2026.py`, add:

```python
GOALS_INTERCEPT = 0.17
GOALS_ELO_COEF = 0.19
GOALS_HOST_COEF = 0.17
```

Replace these values with the fitted estimates.

## 2. Update `expected_goals()`

Use:

```python
def expected_goals(team, opponent, elo):
    team_rating = elo.get(team, 1500)
    opponent_rating = elo.get(opponent, 1500)

    rating_gap = (team_rating - opponent_rating) / 100
    host_boost = 1 if team in HOSTS else 0

    log_mu = (
        GOALS_INTERCEPT
        + GOALS_ELO_COEF * rating_gap
        + GOALS_HOST_COEF * host_boost
    )

    mu = math.exp(log_mu)

    return float(np.clip(mu, 0.15, 4.5))
```

## 3. Re-estimate Dixon-Coles rho

Once the expected-goals model changes, the Dixon-Coles correction should be re-estimated.

Why?

The Dixon-Coles parameter adjusts low-score probabilities **around the expected goals model**. If the mean model changes, the best low-score correction may also change.

Recommended workflow:

```text
estimate goal coefficients
→ paste into predictor scripts
→ paste into Dixon-Coles estimator if needed
→ estimate rho
→ paste rho into predictor scripts
→ run simulations
```

## 4. Keep model settings visible

For reproducibility, print the coefficients when the predictor runs:

```python
print(f"GOALS_INTERCEPT = {GOALS_INTERCEPT}")
print(f"GOALS_ELO_COEF = {GOALS_ELO_COEF}")
print(f"GOALS_HOST_COEF = {GOALS_HOST_COEF}")
print(f"DIXON_COLES_RHO = {DIXON_COLES_RHO}")
```

This helps students connect output files to model assumptions.
