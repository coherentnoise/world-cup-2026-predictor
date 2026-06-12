# estimate_elo_goals_relation.py
# ------------------------------------------------------------
# Estimate the Elo-to-goals relationship from historical
# international football results.
#
# Install:
#   pip install pandas numpy scikit-learn
#
# Run examples:
#   python estimate_elo_goals_relation.py
#   python estimate_elo_goals_relation.py --years-back 10
#   python estimate_elo_goals_relation.py --years-back 20
#   python estimate_elo_goals_relation.py --start-date 2014-01-01
#   python estimate_elo_goals_relation.py --end-date 2026-06-11
#
# Output:
#   elo_goals_model_coefficients.csv
#   elo_goals_training_data.csv
# ------------------------------------------------------------

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error


RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DEFAULT_END_DATE = "2026-06-11"


# ------------------------------------------------------------
# Team-name normalization
# ------------------------------------------------------------

TEAM_NAME_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "USMNT": "United States",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea Rep.": "South Korea",
}


def normalize_team_name(name):
    if name is None:
        return None

    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


# ------------------------------------------------------------
# Elo functions
# ------------------------------------------------------------

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_result(home_goals, away_goals):
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def margin_multiplier(goal_diff):
    goal_diff = abs(goal_diff)

    if goal_diff <= 1:
        return 1.0

    return math.log(goal_diff + 1)


def tournament_k_factor(tournament):
    tournament = str(tournament).lower()

    if tournament == "fifa world cup":
        return 60

    if "qualification" in tournament:
        return 40

    major_continental_terms = [
        "uefa euro",
        "copa américa",
        "copa america",
        "africa cup",
        "asian cup",
        "gold cup",
        "nations league",
    ]

    if any(term in tournament for term in major_continental_terms):
        return 50

    if "friendly" in tournament:
        return 20

    return 30


def update_elo(
    ratings,
    home_team,
    away_team,
    home_score,
    away_score,
    tournament,
    neutral,
    home_advantage=60,
):
    home_rating = ratings[home_team]
    away_rating = ratings[away_team]

    home_bonus = 0 if bool(neutral) else home_advantage

    expected_home = expected_score(home_rating + home_bonus, away_rating)
    result_home = actual_result(home_score, away_score)

    goal_diff = home_score - away_score

    k = tournament_k_factor(tournament)
    k *= margin_multiplier(goal_diff)

    change = k * (result_home - expected_home)

    ratings[home_team] += change
    ratings[away_team] -= change


# ------------------------------------------------------------
# Data loading and feature construction
# ------------------------------------------------------------

def load_results():
    results = pd.read_csv(RESULTS_URL)
    results["date"] = pd.to_datetime(results["date"])

    required_cols = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }

    missing = required_cols - set(results.columns)

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    results["home_team"] = results["home_team"].map(normalize_team_name)
    results["away_team"] = results["away_team"].map(normalize_team_name)

    results = results.sort_values("date").reset_index(drop=True)

    return results


def choose_training_window(results, start_date=None, end_date=None, years_back=None):
    if end_date is None:
        end_date = DEFAULT_END_DATE

    end_date = pd.Timestamp(end_date)

    if years_back is not None:
        start_date = end_date - pd.DateOffset(years=int(years_back))
    elif start_date is not None:
        start_date = pd.Timestamp(start_date)
    else:
        start_date = results["date"].min()

    window = results[
        (results["date"] >= start_date) &
        (results["date"] < end_date)
    ].copy()

    return window, start_date, end_date


def build_elo_goal_training_data(
    results,
    start_date=None,
    end_date=None,
    years_back=None,
    base_rating=1500,
    home_advantage=60,
):
    """
    Build row-level goal-model training data.

    Important:
    Elo is updated chronologically using all matches before each match.
    The goal model is fitted only on matches in the chosen training window.
    This avoids using a match result to create its own pre-match Elo feature.
    """
    results = results.sort_values("date").reset_index(drop=True)

    if end_date is None:
        end_date = DEFAULT_END_DATE

    end_date = pd.Timestamp(end_date)

    if years_back is not None:
        start_date = end_date - pd.DateOffset(years=int(years_back))
    elif start_date is not None:
        start_date = pd.Timestamp(start_date)
    else:
        start_date = results["date"].min()

    ratings = defaultdict(lambda: base_rating)

    rows = []

    for _, row in results.iterrows():
        match_date = row["date"]

        if match_date >= end_date:
            break

        home_team = row["home_team"]
        away_team = row["away_team"]

        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        neutral = bool(row["neutral"])

        pre_home_elo = ratings[home_team]
        pre_away_elo = ratings[away_team]

        home_bonus = 0 if neutral else home_advantage

        raw_elo_diff = pre_home_elo - pre_away_elo
        adjusted_elo_diff = pre_home_elo + home_bonus - pre_away_elo

        if match_date >= start_date:
            rows.append(
                {
                    "date": match_date,
                    "team": home_team,
                    "opponent": away_team,
                    "is_home_team": 1,
                    "is_neutral": int(neutral),
                    "home_advantage_applied": 0 if neutral else 1,
                    "goals": home_score,
                    "team_elo": pre_home_elo,
                    "opponent_elo": pre_away_elo,
                    "raw_elo_diff": raw_elo_diff,
                    "adjusted_elo_diff": adjusted_elo_diff,
                    "elo_diff_per_100": adjusted_elo_diff / 100,
                    "tournament": row["tournament"],
                }
            )

            rows.append(
                {
                    "date": match_date,
                    "team": away_team,
                    "opponent": home_team,
                    "is_home_team": 0,
                    "is_neutral": int(neutral),
                    "home_advantage_applied": 0,
                    "goals": away_score,
                    "team_elo": pre_away_elo,
                    "opponent_elo": pre_home_elo,
                    "raw_elo_diff": -raw_elo_diff,
                    "adjusted_elo_diff": -adjusted_elo_diff,
                    "elo_diff_per_100": -adjusted_elo_diff / 100,
                    "tournament": row["tournament"],
                }
            )

        update_elo(
            ratings=ratings,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            tournament=row["tournament"],
            neutral=neutral,
            home_advantage=home_advantage,
        )

    training_data = pd.DataFrame(rows)

    return training_data, start_date, end_date


# ------------------------------------------------------------
# Model fitting
# ------------------------------------------------------------

def fit_poisson_goal_model(training_data, alpha=0.0):
    """
    Fit:
        goals ~ Poisson(exp(intercept + beta * elo_diff_per_100 + home_effect))

    Since each match contributes two rows, the home effect is captured by
    home_advantage_applied.
    """
    if training_data.empty:
        raise ValueError("Training data is empty. Choose a wider date range.")

    feature_cols = [
        "elo_diff_per_100",
        "home_advantage_applied",
    ]

    X = training_data[feature_cols]
    y = training_data["goals"]

    model = PoissonRegressor(
        alpha=alpha,
        fit_intercept=True,
        max_iter=1000,
    )

    model.fit(X, y)

    pred = model.predict(X)

    metrics = {
        "n_goal_rows": len(training_data),
        "n_matches": len(training_data) // 2,
        "mean_goals": float(y.mean()),
        "mean_predicted_goals": float(pred.mean()),
        "mean_absolute_error": float(mean_absolute_error(y, pred)),
        "mean_poisson_deviance": float(mean_poisson_deviance(y, pred)),
    }

    coefficients = {
        "intercept": float(model.intercept_),
        "beta_elo_diff_per_100": float(model.coef_[0]),
        "beta_home_advantage": float(model.coef_[1]),
    }

    return model, coefficients, metrics


def summarize_model(coefficients):
    intercept = coefficients["intercept"]
    beta_elo = coefficients["beta_elo_diff_per_100"]
    beta_home = coefficients["beta_home_advantage"]

    base_goals = math.exp(intercept)
    goals_even_teams_neutral = math.exp(intercept)
    goals_plus_100_elo = math.exp(intercept + beta_elo)
    goals_minus_100_elo = math.exp(intercept - beta_elo)
    goals_home_even_teams = math.exp(intercept + beta_home)

    rows = [
        {
            "quantity": "base_expected_goals_even_teams_neutral",
            "value": goals_even_teams_neutral,
        },
        {
            "quantity": "expected_goals_plus_100_elo",
            "value": goals_plus_100_elo,
        },
        {
            "quantity": "expected_goals_minus_100_elo",
            "value": goals_minus_100_elo,
        },
        {
            "quantity": "expected_goals_even_teams_home_advantage",
            "value": goals_home_even_teams,
        },
        {
            "quantity": "multiplicative_effect_plus_100_elo",
            "value": math.exp(beta_elo),
        },
        {
            "quantity": "multiplicative_effect_home_advantage",
            "value": math.exp(beta_home),
        },
    ]

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Convenience: print code to paste into your World Cup simulator
# ------------------------------------------------------------

def print_replacement_function(coefficients):
    intercept = coefficients["intercept"]
    beta_elo = coefficients["beta_elo_diff_per_100"]
    beta_home = coefficients["beta_home_advantage"]

    print("\nPaste this into your World Cup simulator:")
    print()
    print("def expected_goals(team, opponent, elo, is_home_advantage=False):")
    print('    """Expected goals from fitted Elo-goals relation."""')
    print("    team_rating = elo.get(team, 1500)")
    print("    opponent_rating = elo.get(opponent, 1500)")
    print("    elo_diff_per_100 = (team_rating - opponent_rating) / 100")
    print(f"    intercept = {intercept:.8f}")
    print(f"    beta_elo = {beta_elo:.8f}")
    print(f"    beta_home = {beta_home:.8f}")
    print("    home_indicator = 1 if is_home_advantage else 0")
    print("    log_mu = intercept + beta_elo * elo_diff_per_100 + beta_home * home_indicator")
    print("    mu = math.exp(log_mu)")
    print("    return float(np.clip(mu, 0.15, 4.5))")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate the relationship between Elo difference and goals scored."
    )

    parser.add_argument(
        "--years-back",
        type=int,
        default=None,
        help="Number of years before end-date to use for fitting. Example: --years-back 10",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for fitting, e.g. 2014-01-01. Ignored if --years-back is used.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"End date for fitting, default {DEFAULT_END_DATE}.",
    )

    parser.add_argument(
        "--home-advantage",
        type=float,
        default=60,
        help="Elo home advantage used when creating pre-match Elo features.",
    )

    parser.add_argument(
        "--base-rating",
        type=float,
        default=1500,
        help="Starting Elo rating for all teams.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Regularization strength for PoissonRegressor. Use 0.0 for unregularized.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading historical results...")
    results = load_results()

    print(f"Loaded {len(results):,} matches.")

    print("\nBuilding pre-match Elo features and goal rows...")
    training_data, start_date, end_date = build_elo_goal_training_data(
        results=results,
        start_date=args.start_date,
        end_date=args.end_date,
        years_back=args.years_back,
        base_rating=args.base_rating,
        home_advantage=args.home_advantage,
    )

    print(f"Training window: {start_date.date()} to {end_date.date()}")
    print(f"Training rows: {len(training_data):,}")
    print(f"Training matches: {len(training_data) // 2:,}")

    print("\nFitting Poisson goal model...")
    model, coefficients, metrics = fit_poisson_goal_model(
        training_data=training_data,
        alpha=args.alpha,
    )

    coef_df = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value in coefficients.items()
        ]
    )

    metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
        ]
    )

    summary_df = summarize_model(coefficients)

    print("\nCoefficients")
    print(coef_df.to_string(index=False))

    print("\nModel metrics")
    print(metrics_df.to_string(index=False))

    print("\nInterpretation")
    print(summary_df.to_string(index=False))

    training_output = "elo_goals_training_data.csv"
    coefficients_output = "elo_goals_model_coefficients.csv"
    metrics_output = "elo_goals_model_metrics.csv"
    summary_output = "elo_goals_model_summary.csv"

    training_data.to_csv(training_output, index=False)
    coef_df.to_csv(coefficients_output, index=False)
    metrics_df.to_csv(metrics_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"\nSaved training data to: {training_output}")
    print(f"Saved coefficients to: {coefficients_output}")
    print(f"Saved metrics to: {metrics_output}")
    print(f"Saved summary to: {summary_output}")

    print_replacement_function(coefficients)


if __name__ == "__main__":
    main()