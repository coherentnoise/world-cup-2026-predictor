# Estimating the Elo-to-goals relationship

The tournament simulator needs two ingredients:

```text
team strength
→ expected goals
```

Elo gives us the first ingredient. This script estimates the second.

## From Elo ratings to goals

Elo ratings are useful for predicting match outcomes, but the score simulator needs expected goals for each team.

The model assumes:

```text
goals scored by a team ~ Poisson(mu)
```

and:

```text
log(mu) = beta_0 + beta_1 * elo_gap_per_100 + beta_2 * host_indicator
```

where:

- `mu` is expected goals
- `elo_gap_per_100` is the team's Elo advantage divided by 100
- `host_indicator` equals 1 if the team is a World Cup host
- `beta_0`, `beta_1`, and `beta_2` are estimated from data

The log link ensures expected goals are positive.

## Why use Poisson regression?

Goals are non-negative counts:

```text
0, 1, 2, 3, ...
```

Poisson regression is a natural baseline model for count data.

It is not perfect for football, but it gives a clear and interpretable starting point.

## How the training data is built

Each historical match creates two rows.

For example:

```text
Brazil 2-1 Argentina
```

becomes:

```text
Brazil row:    target = 2 goals, feature = Brazil Elo - Argentina Elo
Argentina row: target = 1 goal,  feature = Argentina Elo - Brazil Elo
```

This doubles the number of observations and lets the model learn how team strength relates to scoring.

## Avoiding leakage

The most important design choice is that Elo ratings are recorded before each match.

Bad approach:

```text
use final Elo ratings after all matches
→ model past matches
```

This leaks future information.

Good approach:

```text
for each match in chronological order:
    record pre-match Elo
    train on that row
    update Elo after the match
```

This mimics the information that would have been available before kickoff.

## Choosing the training window

Older international matches may be less relevant to current football.

The script allows:

```bash
--years-back 5
--years-back 10
--years-back 20
```

Shorter windows are more current but have fewer observations. Longer windows are more stable but may include older tactical eras.

## Interpreting coefficients

The Elo coefficient tells us how much expected goals increase when a team is stronger.

Because the model uses a log link, the coefficient is multiplicative.

If:

```text
GOALS_ELO_COEF = 0.20
```

then a 100-point Elo advantage changes expected goals by:

```text
exp(0.20) ≈ 1.22
```

So the stronger team is expected to score about 22% more goals, all else equal.

## Limitations

This model does not include:

- attacking and defensive strengths separately
- player injuries
- squad quality
- rest days
- travel distance
- venue effects beyond a simple home/host indicator
- tactical styles
- red cards
- in-match game state
- market odds

It is a strong teaching baseline, not a complete professional forecasting model.

## Recommended extension

A natural next model is an attack-defence Poisson model:

```text
log(mu_home) = attack_home - defence_away + home_advantage
log(mu_away) = attack_away - defence_home
```

That separates a team's ability to score from its ability to prevent goals.
