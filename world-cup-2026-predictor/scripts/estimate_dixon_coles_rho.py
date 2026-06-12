# estimate_dixon_coles_rho.py
# ------------------------------------------------------------
# Estimate the Dixon-Coles low-score correction parameter rho
# from historical international football results.
#
# The script:
#   1. Loads international results from RESULTS_URL
#   2. Builds pre-match Elo ratings chronologically
#   3. Converts Elo gaps into expected goals
#   4. Estimates Dixon-Coles rho by maximum likelihood
#
# Install:
#   pip install pandas numpy scipy
#
# Run:
#   python estimate_dixon_coles_rho.py
#   python estimate_dixon_coles_rho.py --years-back 10
# ------------------------------------------------------------

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DEFAULT_CUTOFF_DATE = "2026-06-11"

HOSTS = {"United States", "Mexico", "Canada"}

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

    if name == "" or name.lower() == "nan":
        return None

    return TEAM_NAME_MAP.get(name, name)


def load_results(cutoff_date=None):
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

    results = results.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)

    if cutoff_date is not None:
        cutoff_date = pd.Timestamp(cutoff_date)
        results = results[results["date"] < cutoff_date].copy()

    results = results.sort_values("date").reset_index(drop=True)

    return results


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


def build_pre_match_elo_dataset(
    matches,
    base_rating=1500,
    home_advantage=60,
):
    """
    Process matches chronologically.

    For each match, store the Elo ratings BEFORE the match,
    then update ratings after the match.

    This avoids leakage.
    """
    ratings = defaultdict(lambda: base_rating)
    rows = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])

        neutral = bool(row["neutral"])

        home_rating_pre = ratings[home]
        away_rating_pre = ratings[away]

        home_bonus = 0 if neutral else home_advantage

        rows.append(
            {
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_score": home_goals,
                "away_score": away_goals,
                "tournament": row["tournament"],
                "neutral": neutral,
                "home_elo_pre": home_rating_pre,
                "away_elo_pre": away_rating_pre,
                "home_advantage_applied": home_bonus,
            }
        )

        exp_home = expected_score(
            home_rating_pre + home_bonus,
            away_rating_pre,
        )

        res_home = actual_result(home_goals, away_goals)
        goal_diff = home_goals - away_goals

        k = tournament_k_factor(row["tournament"])
        k *= margin_multiplier(goal_diff)

        change = k * (res_home - exp_home)

        ratings[home] += change
        ratings[away] -= change

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Expected goals baseline
# ------------------------------------------------------------

def expected_goals_from_elo(
    team_rating,
    opponent_rating,
    is_host=False,
    intercept=0.17,
    elo_coef=0.19,
    host_coef=0.17,
):
    """
    Same expected-goals formula as the tournament script.

    log(mu) = intercept + elo_coef * rating_gap_per_100 + host_coef * host_flag

    You can replace these coefficients with fitted values from your
    Elo-goals estimation script.
    """
    rating_gap_per_100 = (team_rating - opponent_rating) / 100
    host_boost = 1 if is_host else 0

    log_mu = intercept + elo_coef * rating_gap_per_100 + host_coef * host_boost
    mu = math.exp(log_mu)

    return float(np.clip(mu, 0.15, 4.5))


def add_expected_goals_columns(
    data,
    intercept=0.17,
    elo_coef=0.19,
    host_coef=0.17,
):
    data = data.copy()

    home_mu = []
    away_mu = []

    for _, row in data.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_is_host = home in HOSTS
        away_is_host = away in HOSTS

        mu_home = expected_goals_from_elo(
            team_rating=row["home_elo_pre"] + row["home_advantage_applied"],
            opponent_rating=row["away_elo_pre"],
            is_host=home_is_host,
            intercept=intercept,
            elo_coef=elo_coef,
            host_coef=host_coef,
        )

        mu_away = expected_goals_from_elo(
            team_rating=row["away_elo_pre"],
            opponent_rating=row["home_elo_pre"] + row["home_advantage_applied"],
            is_host=away_is_host,
            intercept=intercept,
            elo_coef=elo_coef,
            host_coef=host_coef,
        )

        home_mu.append(mu_home)
        away_mu.append(mu_away)

    data["home_mu"] = home_mu
    data["away_mu"] = away_mu

    return data


# ------------------------------------------------------------
# Dixon-Coles likelihood
# ------------------------------------------------------------

def poisson_logpmf(k, mu):
    return -mu + k * math.log(mu) - math.lgamma(k + 1)


def dixon_coles_tau(home_goals, away_goals, home_mu, away_mu, rho):
    if home_goals == 0 and away_goals == 0:
        return 1 - home_mu * away_mu * rho

    if home_goals == 0 and away_goals == 1:
        return 1 + home_mu * rho

    if home_goals == 1 and away_goals == 0:
        return 1 + away_mu * rho

    if home_goals == 1 and away_goals == 1:
        return 1 - rho

    return 1.0


def dixon_coles_log_likelihood(data, rho):
    """
    Log likelihood under independent Poisson goals plus Dixon-Coles correction.
    """
    total = 0.0

    for _, row in data.iterrows():
        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])

        home_mu = float(row["home_mu"])
        away_mu = float(row["away_mu"])

        tau = dixon_coles_tau(
            home_goals=home_goals,
            away_goals=away_goals,
            home_mu=home_mu,
            away_mu=away_mu,
            rho=rho,
        )

        if tau <= 0:
            return -np.inf

        log_prob = (
            poisson_logpmf(home_goals, home_mu)
            + poisson_logpmf(away_goals, away_mu)
            + math.log(tau)
        )

        total += log_prob

    return total


def negative_log_likelihood(data, rho):
    ll = dixon_coles_log_likelihood(data, rho)

    if not np.isfinite(ll):
        return np.inf

    return -ll


def estimate_rho(data, lower=-0.30, upper=0.30):
    """
    Estimate rho by scalar optimization.
    """
    result = minimize_scalar(
        lambda r: negative_log_likelihood(data, r),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not result.success:
        raise RuntimeError(f"rho optimization failed: {result.message}")

    rho_hat = float(result.x)
    log_likelihood = -float(result.fun)

    return rho_hat, log_likelihood, result


def grid_search_rho(data, lower=-0.30, upper=0.30, step=0.01):
    """
    Optional coarse grid search, useful for diagnostics.
    """
    values = np.arange(lower, upper + step / 2, step)

    rows = []

    for rho in values:
        ll = dixon_coles_log_likelihood(data, rho)
        rows.append(
            {
                "rho": rho,
                "log_likelihood": ll,
                "negative_log_likelihood": -ll if np.isfinite(ll) else np.inf,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("log_likelihood", ascending=False).reset_index(drop=True)

    return out


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def low_score_summary(data):
    data = data.copy()

    data["scoreline"] = (
        data["home_score"].astype(str)
        + "-"
        + data["away_score"].astype(str)
    )

    interesting = ["0-0", "0-1", "1-0", "1-1"]

    counts = (
        data["scoreline"]
        .value_counts()
        .reindex(interesting)
        .fillna(0)
        .astype(int)
    )

    proportions = counts / len(data)

    out = pd.DataFrame(
        {
            "scoreline": interesting,
            "count": counts.values,
            "proportion": proportions.values,
        }
    )

    return out


def apply_date_filters(results, years_back=None, start_date=None, end_date=None):
    filtered = results.copy()

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
    else:
        end_date = filtered["date"].max()

    if years_back is not None:
        start_date = end_date - pd.DateOffset(years=int(years_back))

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        filtered = filtered[filtered["date"] >= start_date].copy()

    if end_date is not None:
        filtered = filtered[filtered["date"] <= end_date].copy()

    return filtered.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cutoff-date",
        default=DEFAULT_CUTOFF_DATE,
        help=(
            "Only use results before this date when loading the raw data. "
            "Default: 2026-06-11."
        ),
    )

    parser.add_argument(
        "--years-back",
        type=int,
        default=None,
        help="Estimate rho using only this many years before the end date.",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date for estimation sample.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date for estimation sample.",
    )

    parser.add_argument(
        "--base-rating",
        type=float,
        default=1500,
        help="Starting Elo rating.",
    )

    parser.add_argument(
        "--home-advantage",
        type=float,
        default=60,
        help="Home advantage in Elo points for non-neutral matches.",
    )

    parser.add_argument(
        "--intercept",
        type=float,
        default=0.17,
        help="Expected-goals intercept.",
    )

    parser.add_argument(
        "--elo-coef",
        type=float,
        default=0.19,
        help="Expected-goals coefficient for Elo gap per 100 points.",
    )

    parser.add_argument(
        "--host-coef",
        type=float,
        default=0.17,
        help="Expected-goals coefficient for World Cup host indicator.",
    )

    parser.add_argument(
        "--rho-lower",
        type=float,
        default=-0.30,
        help="Lower bound for rho optimization.",
    )

    parser.add_argument(
        "--rho-upper",
        type=float,
        default=0.30,
        help="Upper bound for rho optimization.",
    )

    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Step size for diagnostic rho grid search.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading results...")
    results = load_results(cutoff_date=args.cutoff_date)

    print(f"Loaded {len(results):,} matches before {args.cutoff_date}.")

    print("Building pre-match Elo dataset...")
    elo_data = build_pre_match_elo_dataset(
        matches=results,
        base_rating=args.base_rating,
        home_advantage=args.home_advantage,
    )

    estimation_data = apply_date_filters(
        elo_data,
        years_back=args.years_back,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if estimation_data.empty:
        raise ValueError("No matches available after applying date filters.")

    print(f"Using {len(estimation_data):,} matches for Dixon-Coles estimation.")
    print(
        f"Estimation window: "
        f"{estimation_data['date'].min().date()} to {estimation_data['date'].max().date()}"
    )

    estimation_data = add_expected_goals_columns(
        estimation_data,
        intercept=args.intercept,
        elo_coef=args.elo_coef,
        host_coef=args.host_coef,
    )

    print("\nLow-score frequencies in estimation data:")
    low_scores = low_score_summary(estimation_data)
    print(low_scores.to_string(index=False))

    print("\nEstimating Dixon-Coles rho...")
    rho_hat, log_likelihood, opt_result = estimate_rho(
        estimation_data,
        lower=args.rho_lower,
        upper=args.rho_upper,
    )

    independent_ll = dixon_coles_log_likelihood(estimation_data, rho=0.0)

    print("\nEstimated Dixon-Coles parameter")
    print(f"rho_hat: {rho_hat:.6f}")
    print(f"log_likelihood_at_rho_hat: {log_likelihood:.3f}")
    print(f"log_likelihood_at_rho_0:   {independent_ll:.3f}")
    print(f"ll_improvement:            {log_likelihood - independent_ll:.3f}")

    print("\nUse this in your predictor script:")
    print(f"DIXON_COLES_RHO = {rho_hat:.6f}")

    print("\nRunning diagnostic rho grid search...")
    grid = grid_search_rho(
        estimation_data,
        lower=args.rho_lower,
        upper=args.rho_upper,
        step=args.grid_step,
    )

    grid_path = "dixon_coles_rho_grid.csv"
    grid.to_csv(grid_path, index=False)

    estimation_path = "dixon_coles_estimation_data.csv"
    estimation_data.to_csv(estimation_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "rho_hat": rho_hat,
                "log_likelihood_at_rho_hat": log_likelihood,
                "log_likelihood_at_rho_0": independent_ll,
                "ll_improvement": log_likelihood - independent_ll,
                "n_matches": len(estimation_data),
                "start_date": estimation_data["date"].min().date(),
                "end_date": estimation_data["date"].max().date(),
                "cutoff_date": args.cutoff_date,
                "years_back": args.years_back,
                "expected_goals_intercept": args.intercept,
                "expected_goals_elo_coef": args.elo_coef,
                "expected_goals_host_coef": args.host_coef,
                "rho_lower": args.rho_lower,
                "rho_upper": args.rho_upper,
            }
        ]
    )

    summary_path = "dixon_coles_rho_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved grid search results to: {grid_path}")
    print(f"Saved estimation data to: {estimation_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()