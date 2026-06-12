# Knockout-stage workflow

Use `data/knockout_matches.csv` once knockout fixtures are known.

After each stage, edit the CSV and rerun:

```bash
python scripts/knockout_stage_predictor_2026.py
```

To predict only one stage, edit:

```python
KNOCKOUT_PREDICT_ROUND = "R32"
```

Allowed values:

```text
all_known, R32, R16, QF, SF, Final
```

The script reports:

- average 90-minute score
- 90-minute win/draw/loss probabilities
- estimated advancement probabilities
- most likely 90-minute score
- predicted winner
