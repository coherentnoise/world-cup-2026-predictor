# 2026 World Cup Predictor

A teaching-oriented Python project for simulating the 2026 FIFA World Cup using historical results, Elo ratings, expected-goals modelling, Dixon–Coles low-score correction, and Monte Carlo simulation.

The repo is designed for students learning forecasting, sports analytics, probability models, and reproducible data science.

## Disclaimer - I do not encourage sports betting based on these predictions.

## Project structure

```text
world-cup-2026-predictor/
├── data/
│   ├── manual_results.csv
│   ├── knockout_matches.csv
│   └── third_place_mapping_starter.csv
├── docs/
│   ├── 01_model_overview.md
│   ├── 02_how_to_run.md
│   ├── 03_live_updates.md
│   ├── 04_knockout_stage.md
│   ├── 05_dixon_coles.md
│   └── 06_student_exercises.md
├── notebooks/
├── outputs/
├── scripts/
│   ├── world_cup_predictor_live_2026.py
│   ├── knockout_stage_predictor_2026.py
│   ├── estimate_elo_goals_relation.py
│   ├── estimate_dixon_coles_rho.py
│   └── create_third_place_mapping_csv.py
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Run the main teaching predictor:

```bash
python scripts/world_cup_predictor_2026.py
```

Run knockout-stage predictions from `data/knockout_matches.csv`:

```bash
python scripts/knockout_stage_predictor_2026.py
```

Estimate the Dixon–Coles correction parameter:

```bash
python scripts/estimate_dixon_coles_rho.py --years-back 10
```

Create the full third-place mapping file:

```bash
python scripts/create_third_place_mapping_csv.py
```

## Main outputs

Outputs are written to `outputs/`:

```text
world_cup_prediction_probabilities.csv
world_cup_group_fixture_predictions.csv
world_cup_knockout_match_predictions.csv
dixon_coles_rho_summary.csv
```

## Teaching goals

Students should be able to explain:

1. How Elo ratings update after matches.
2. How expected goals can be derived from team strength.
3. How scorelines can be simulated with Poisson models.
4. Why independent Poisson scores are limited.
5. How Dixon–Coles adjusts low-score probabilities.
6. How Monte Carlo simulation produces tournament probabilities.
7. How leakage can enter live sports forecasting.


