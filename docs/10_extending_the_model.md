# Extending the World Cup prediction model

This guide explains how to extend the baseline World Cup prediction model with additional covariates and richer modelling choices.

The goal is not just to make the model more complicated. The goal is to make each modelling choice explicit, testable, and teachable.

## Baseline pipeline

```text
historical results
→ Elo ratings
→ expected goals
→ Dixon-Coles score probabilities
→ group-stage simulation
→ third-place qualification mapping
→ knockout simulation
→ tournament probabilities
```

The main place to add covariates is here:

```text
Elo ratings + covariates
→ expected goals
```

In code, this means extending:

```python
expected_goals(team, opponent, elo)
```

to:

```python
expected_goals(team, opponent, elo, covariates=None)
```

## Types of covariates

### Team-level covariates

These describe a team before a match:

```text
FIFA ranking
squad market value
recent form
injury index
average squad age
players in top leagues
manager tenure
```

They usually live in:

```text
data/team_covariates_2026.csv
```

Example:

```csv
team,fifa_rank,fifa_points,market_value_million_eur,recent_form_points,recent_goal_diff,injury_index
Argentina,1,1885.36,850,2.4,1.1,0.05
France,2,1867.71,1050,2.2,0.9,0.08
England,4,1813.81,1250,2.1,0.8,0.10
Brazil,5,1775.85,950,1.9,0.6,0.06
```

### Match-level covariates

These describe a specific fixture:

```text
venue
host country
rest days
travel distance
kickoff time
temperature
altitude
knockout round
```

They usually live in:

```text
data/match_covariates_2026.csv
```

Example:

```csv
match_no,date,team_a,team_b,venue,rest_days_a,rest_days_b,travel_km_a,travel_km_b
1,2026-06-11,Mexico,South Africa,Estadio Azteca,7,6,0,9200
```

## Recommended order of extensions

```text
1. Recent form
2. FIFA ranking or FIFA points
3. Market value
4. Injury index
5. Rest days
6. Travel distance
7. Attack and defence strengths
```

Start with recent form because it can be computed from the same historical results data already used by the project.

## Extension 1: recent form

Recent form can be computed directly from the results data.

Useful features:

```text
recent_form_points
recent_goal_diff
recent_goals_for
recent_goals_against
```

For each team, use the last `N` matches before the prediction date.

Example:

```text
recent_form_points = average points per match over last 10 matches
```

where:

```text
win = 3 points
draw = 1 point
loss = 0 points
```

## Extension 2: FIFA ranking

FIFA ranking or FIFA points can be added as a team-level covariate.

Be careful: FIFA ranking and Elo are both team-strength measures, so they may be strongly correlated.

Useful lesson:

```text
Adding more covariates does not always improve a model.
```

Students should check whether the covariate improves validation performance.

## Extension 3: market value

Squad market value is a proxy for player quality.

Use a log transform:

```python
log_market_value = math.log1p(market_value_million_eur)
```

because the difference between €50m and €150m is usually more meaningful than the difference between €950m and €1050m.

## Extension 4: injuries

Injury data is hard to source consistently.

For a teaching project, use a manual index:

```csv
team,injury_index,notes
France,0.08,one likely starter doubtful
England,0.12,two rotation players missing
Brazil,0.06,key attacker returning
```

Suggested scale:

```text
0.00 = no known issue
0.05 = minor concern
0.10 = one important player missing
0.20 = several important players missing
```

## Code pattern for adding team covariates

### Settings

```python
TEAM_COVARIATES_PATH = "data/team_covariates_2026.csv"
USE_TEAM_COVARIATES = True

GOALS_INTERCEPT = 0.17
GOALS_ELO_COEF = 0.12
GOALS_HOST_COEF = 0.17
GOALS_RECENT_FORM_COEF = 0.05
GOALS_MARKET_VALUE_COEF = 0.04
GOALS_INJURY_COEF = -0.10
```

### Loader

```python
def load_team_covariates(path=TEAM_COVARIATES_PATH):
    if not USE_TEAM_COVARIATES:
        return {}

    if not os.path.exists(path):
        print(f"No team covariates file found at {path}. Continuing without covariates.")
        return {}

    df = pd.read_csv(path)

    if "team" not in df.columns:
        raise ValueError("Team covariate file must contain a 'team' column.")

    df["team"] = df["team"].map(normalize_team_name)

    for col in df.columns:
        if col != "team":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.set_index("team").to_dict(orient="index")
```

### Missing-value helper

```python
def get_covariate(team, covariates, name, default=0.0):
    if covariates is None:
        return default

    if team not in covariates:
        return default

    value = covariates[team].get(name, default)

    if pd.isna(value):
        return default

    return float(value)
```

### Extended expected-goals function

```python
def expected_goals(team, opponent, elo, covariates=None):
    team_rating = elo.get(team, 1500)
    opponent_rating = elo.get(opponent, 1500)

    rating_gap = (team_rating - opponent_rating) / 100
    rating_gap = np.clip(rating_gap, -MAX_ELO_GAP_PER_100, MAX_ELO_GAP_PER_100)
    rating_gap = ELO_GAP_SHRINKAGE * rating_gap

    host_boost = 1 if team in HOSTS else 0

    recent_form_gap = (
        get_covariate(team, covariates, "recent_form_points", 0.0)
        - get_covariate(opponent, covariates, "recent_form_points", 0.0)
    )

    market_value_team = get_covariate(team, covariates, "market_value_million_eur", 0.0)
    market_value_opp = get_covariate(opponent, covariates, "market_value_million_eur", 0.0)
    market_value_gap = math.log1p(market_value_team) - math.log1p(market_value_opp)

    injury_gap = (
        get_covariate(team, covariates, "injury_index", 0.0)
        - get_covariate(opponent, covariates, "injury_index", 0.0)
    )

    log_mu = (
        GOALS_INTERCEPT
        + GOALS_ELO_COEF * rating_gap
        + GOALS_HOST_COEF * host_boost
        + GOALS_RECENT_FORM_COEF * recent_form_gap
        + GOALS_MARKET_VALUE_COEF * market_value_gap
        + GOALS_INJURY_COEF * injury_gap
    )

    mu = math.exp(log_mu)

    return float(np.clip(mu, 0.15, 4.5))
```

## Threading covariates through the simulation

Once `expected_goals()` accepts covariates, every function between the tournament simulator and `expected_goals()` must pass them along.

Update signatures such as:

```python
get_score_matrix_for_match(team_a, team_b, elo, covariates=None)
simulate_score(team_a, team_b, elo, covariates=None)
simulate_group_match(team_a, team_b, elo, covariates=None)
simulate_knockout_match(team_a, team_b, elo, covariates=None)
simulate_group(..., covariates=None)
simulate_group_stage(..., covariates=None)
simulate_tournament(..., covariates=None)
run_simulations(..., covariates=None)
predict_group_fixture_exact(..., covariates=None)
predict_all_group_fixtures(..., covariates=None)
```

This is a good software engineering lesson: once a model input changes, the call chain must be updated consistently.

## Estimating covariate coefficients

Hand-tuned coefficients are useful for experimentation, but the better version is to estimate them.

Extend `estimate_elo_goals_relation.py` to fit:

```text
goals ~ Elo gap + host + recent form gap + market value gap + injury gap
```

Then copy the fitted coefficients into the predictor.

## Avoiding leakage

Leakage means using information that would not have been available at prediction time.

Common leakage risks:

```text
using final tournament results to compute recent form
using post-match Elo ratings as pre-match features
using injury information published after the match
using knockout opponents before they are known
using market values updated after the tournament
```

A safe rule:

```text
Every covariate must have a timestamp.
Only use values known before the match being predicted.
```

## Validation

Suggested validation:

```text
1. Choose a historical prediction date.
2. Build features using only data before that date.
3. Predict future matches.
4. Score predictions using log loss, Brier score, or negative log likelihood.
5. Compare baseline versus extended model.
```

For scoreline models, use negative log likelihood:

```text
- log P(actual score)
```

For win/draw/loss predictions, use log loss or Brier score.

## Suggested student project

Ask students to choose one covariate and answer:

1. What does the covariate measure?
2. Where did the data come from?
3. Was it available before the match?
4. How was it transformed?
5. What coefficient sign do you expect?
6. Does it improve validation performance?
7. Does it change the tournament winner probabilities?
8. Could it create bias or measurement error?

## Recommended classroom progression

```text
Week 1: Elo-only model
Week 2: Fit expected-goals coefficients from historical data
Week 3: Estimate Dixon-Coles rho
Week 4: Add recent form
Week 5: Add one external covariate
Week 6: Backtest and present model comparison
```
