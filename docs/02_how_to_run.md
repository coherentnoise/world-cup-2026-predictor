# How to run

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main predictor

```bash
python scripts/world_cup_predictor_live_2026.py
```

## Knockout predictor

Edit `data/knockout_matches.csv`, then run:

```bash
python scripts/knockout_stage_predictor_2026.py
```

## Dixon–Coles rho estimation

```bash
python scripts/estimate_dixon_coles_rho.py --years-back 10
```

Copy the printed value into both predictor scripts:

```python
DIXON_COLES_RHO = -0.08
```
