# Live updates

For live group-stage use, assume the results URL is updated after every match.

Recommended settings in the main script:

```python
RESULT_SOURCE = "results_url"
UPDATE_ELO_WITH_PLAYED_MATCHES = True
```

This means:

```text
load updated results → detect completed group matches → update Elo → lock scores → simulate remaining matches
```

## Avoid double-counting

If a World Cup match is already in the results URL, do not also append it manually. The script handles this by using the full results URL directly in `results_url` mode.

## Pre-tournament mode

Use:

```python
RESULT_SOURCE = "manual"
UPDATE_ELO_WITH_PLAYED_MATCHES = False
```

with an empty `data/manual_results.csv`.
