# Student exercises

## Exercise 1: Run the baseline

Run the predictor and identify the top five title favourites.

Questions:

1. Which teams have the highest Elo?
2. Are they also the most likely winners?
3. Why can the bracket affect title probabilities?

## Exercise 2: Simulation stability

Try:

```python
N_SIMS = 1000
N_SIMS = 10000
N_SIMS = 50000
```

How stable are the probabilities?

## Exercise 3: Dixon–Coles comparison

Run once with:

```python
USE_DIXON_COLES = False
```

and once with:

```python
USE_DIXON_COLES = True
```

What changes?

## Exercise 4: Estimate rho

Run:

```bash
python scripts/estimate_dixon_coles_rho.py --years-back 5
python scripts/estimate_dixon_coles_rho.py --years-back 10
python scripts/estimate_dixon_coles_rho.py --years-back 20
```

Does rho change across windows?

## Exercise 5: Backtesting

Modify the script to forecast a previous World Cup using only data available before that tournament.

Where might leakage enter?
