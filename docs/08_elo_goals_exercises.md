# Exercises: estimating goals from Elo

## Exercise 1: Run the default model

Run:

```bash
python scripts/estimate_elo_goals_relation.py
```

Questions:

1. What coefficients are estimated?
2. Are they positive or negative?
3. Do the signs make sense?

## Exercise 2: Change the training window

Run:

```bash
python scripts/estimate_elo_goals_relation.py --years-back 5
python scripts/estimate_elo_goals_relation.py --years-back 10
python scripts/estimate_elo_goals_relation.py --years-back 20
```

Questions:

1. How stable are the coefficients?
2. Which training window would you use for 2026 predictions?
3. What is the trade-off between recency and sample size?

## Exercise 3: Compare hand-tuned and fitted coefficients

The original model used:

```python
log_mu = 0.17 + 0.19 * rating_gap + 0.17 * host_boost
```

Questions:

1. Are the fitted coefficients close to these values?
2. How much do fixture-level predictions change?
3. How much do tournament probabilities change?

## Exercise 4: Test for leakage

Modify the script so it accidentally uses final Elo ratings for all matches.

Questions:

1. Why is this leakage?
2. Do model metrics look better?
3. Why would that improvement be misleading?

## Exercise 5: Estimate Dixon-Coles after fitting goals

After estimating the goals model, paste the coefficients into the Dixon-Coles rho script.

Then run:

```bash
python scripts/estimate_dixon_coles_rho.py --years-back 10
```

Questions:

1. Does the estimated rho change?
2. Why should rho be estimated after the mean goals model?
3. Does the model improve low-score probabilities?

## Exercise 6: Add a new covariate

Choose one:

- neutral venue
- tournament type
- continent
- recent form
- FIFA ranking
- squad market value
- rest days

Questions:

1. Where would the covariate enter the model?
2. Could it create leakage?
3. How would you validate whether it helps?
