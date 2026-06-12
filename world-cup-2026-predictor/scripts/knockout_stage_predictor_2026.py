# knockout_stage_predictor_2026.py
# ------------------------------------------------------------
# Predict known 2026 World Cup knockout-stage matches.
#
# This script is designed to be run repeatedly:
#   - after the group stage
#   - after the Round of 32
#   - after the Round of 16
#   - after the quarterfinals
#   - after the semifinals
#
# It reads knockout_matches.csv and predicts every filled fixture,
# or only the round selected by KNOCKOUT_PREDICT_ROUND.
#
# Required file:
#   knockout_matches.csv
#
# Install:
#   pip install pandas numpy tqdm
#
# Run:
#   python knockout_stage_predictor_2026.py
#
# Output:
#   world_cup_knockout_match_predictions.csv
# ------------------------------------------------------------

import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

RANDOM_STATE = 0
rng = np.random.default_rng(RANDOM_STATE)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

#KNOCKOUT_MATCHES_PATH = "knockout_matches.csv"
# Use dummy until we get there
KNOCKOUT_MATCHES_PATH = "knockout_matches_dummy.csv"


# Options:
#   "all_known" -> predict every filled row in knockout_matches.csv
#   "R32"       -> predict only Round of 32 rows
#   "R16"       -> predict only Round of 16 rows
#   "QF"        -> predict only quarterfinal rows
#   "SF"        -> predict only semifinal rows
#   "Final"     -> predict only final row

KNOCKOUT_PREDICT_ROUND = "all_known"
#KNOCKOUT_PREDICT_ROUND = "Final"


# If True, use every available result in RESULTS_URL to build Elo.
# This is what you want once the tournament is live and RESULTS_URL
# is kept up to date.
USE_LIVE_RESULTS_FOR_ELO = True

# Tournament start date. Used only if USE_LIVE_RESULTS_FOR_ELO = False.
CUTOFF_DATE = pd.Timestamp("2026-06-11")

HOSTS = {"United States", "Mexico", "Canada"}

# Dixon-Coles settings.
USE_DIXON_COLES = True

# rho = 0 gives independent Poisson.
# Negative rho usually boosts 0-0 and 1-1 while reducing 1-0 and 0-1.
#DIXON_COLES_RHO = -0.10

# computed from all WC data:
DIXON_COLES_RHO = -0.05

MAX_GOALS_FOR_SCORE_MATRIX = 10


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

    if name == "" or name.lower() == "nan":
        return None

    return TEAM_NAME_MAP.get(name, name)


# ------------------------------------------------------------
# Knockout matches CSV
# ------------------------------------------------------------

def ensure_knockout_matches_file(path):
    """
    Create an empty knockout matches CSV if it does not already exist.
    """
    if os.path.exists(path):
        return

    rows = []

    for match_no in range(73, 89):
        rows.append(
            {
                "match_no": match_no,
                "round": "R32",
                "team_a": "",
                "team_b": "",
            }
        )

    for match_no in range(89, 97):
        rows.append(
            {
                "match_no": match_no,
                "round": "R16",
                "team_a": "",
                "team_b": "",
            }
        )

    for match_no in range(97, 101):
        rows.append(
            {
                "match_no": match_no,
                "round": "QF",
                "team_a": "",
                "team_b": "",
            }
        )

    for match_no in range(101, 103):
        rows.append(
            {
                "match_no": match_no,
                "round": "SF",
                "team_a": "",
                "team_b": "",
            }
        )

    rows.append(
        {
            "match_no": 104,
            "round": "Final",
            "team_a": "",
            "team_b": "",
        }
    )

    empty = pd.DataFrame(rows)
    empty.to_csv(path, index=False)

    print(f"Created empty knockout matches file: {path}")


def load_knockout_matches_file(path, round_filter="all_known"):
    """
    Read actual knockout matches from a CSV file.

    Expected columns:
        match_no, round, team_a, team_b

    round_filter:
        "all_known", "R32", "R16", "QF", "SF", "Final"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find knockout file: {path}")

    matches = pd.read_csv(path)

    required_cols = {"match_no", "round", "team_a", "team_b"}
    missing = required_cols - set(matches.columns)

    if missing:
        raise ValueError(
            f"Knockout matches file is missing columns: {missing}"
        )

    matches = matches.copy()

    matches["team_a"] = matches["team_a"].map(normalize_team_name)
    matches["team_b"] = matches["team_b"].map(normalize_team_name)
    matches["round"] = matches["round"].astype(str).str.strip()

    # Keep only rows where both teams are filled in.
    matches = matches[
        matches["team_a"].notna()
        & matches["team_b"].notna()
        & (matches["team_a"].astype(str).str.strip() != "")
        & (matches["team_b"].astype(str).str.strip() != "")
    ].copy()

    if matches.empty:
        return matches

    matches["match_no"] = matches["match_no"].astype(int)

    if round_filter != "all_known":
        allowed_rounds = {"R32", "R16", "QF", "SF", "Final"}

        if round_filter not in allowed_rounds:
            raise ValueError(
                f"round_filter must be one of {allowed_rounds} or 'all_known'."
            )

        matches = matches[matches["round"] == round_filter].copy()

    return matches.sort_values("match_no").reset_index(drop=True)


# ------------------------------------------------------------
# Historical results and Elo
# ------------------------------------------------------------

def load_results():
    """
    Load all available results from RESULTS_URL.

    If USE_LIVE_RESULTS_FOR_ELO is True, all rows are used.
    If False, only matches before CUTOFF_DATE are used.
    """
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

    if not USE_LIVE_RESULTS_FOR_ELO:
        results = results[results["date"] < CUTOFF_DATE].copy()

    results = results.sort_values("date").reset_index(drop=True)

    return results


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


def build_elo_ratings(matches, base_rating=1500, home_advantage=60):
    """
    Build Elo ratings by processing matches chronologically.
    """
    ratings = defaultdict(lambda: base_rating)

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home is None or away is None:
            continue

        home_score = row["home_score"]
        away_score = row["away_score"]

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_score = int(home_score)
        away_score = int(away_score)

        home_rating = ratings[home]
        away_rating = ratings[away]

        home_bonus = 0 if bool(row["neutral"]) else home_advantage

        exp_home = expected_score(home_rating + home_bonus, away_rating)
        res_home = actual_result(home_score, away_score)

        goal_diff = home_score - away_score

        k = tournament_k_factor(row["tournament"])
        k *= margin_multiplier(goal_diff)

        change = k * (res_home - exp_home)

        ratings[home] += change
        ratings[away] -= change

    return dict(ratings)


# ------------------------------------------------------------
# Expected goals and Dixon-Coles score model
# ------------------------------------------------------------

def expected_goals(team, opponent, elo):
    """
    Convert Elo gap into expected goals.

    These coefficients are the same simple baseline used in the main
    tournament script. Replace with fitted coefficients if desired.
    """
    team_rating = elo.get(team, 1500)
    opponent_rating = elo.get(opponent, 1500)

    rating_gap = (team_rating - opponent_rating) / 100

    host_boost = 1 if team in HOSTS else 0

    log_mu = 0.17 + 0.19 * rating_gap + 0.17 * host_boost

    mu = math.exp(log_mu)

    return float(np.clip(mu, 0.15, 4.5))


def poisson_pmf(k, mu):
    return math.exp(-mu) * (mu ** k) / math.factorial(k)


def dixon_coles_tau(x, y, lambda_x, lambda_y, rho):
    """
    Dixon-Coles low-score correction.

    Affects:
        0-0, 0-1, 1-0, 1-1
    """
    if x == 0 and y == 0:
        return 1 - lambda_x * lambda_y * rho

    if x == 0 and y == 1:
        return 1 + lambda_x * rho

    if x == 1 and y == 0:
        return 1 + lambda_y * rho

    if x == 1 and y == 1:
        return 1 - rho

    return 1.0


def score_probability_matrix(mu_a, mu_b, rho=0.0, max_goals=10):
    """
    Build a normalized score probability matrix.

    If rho=0, this is independent Poisson.
    If rho!=0, this applies the Dixon-Coles low-score correction.
    """
    scores = []
    probs = []

    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            base_prob = poisson_pmf(goals_a, mu_a) * poisson_pmf(goals_b, mu_b)

            tau = dixon_coles_tau(
                goals_a,
                goals_b,
                mu_a,
                mu_b,
                rho,
            )

            prob = max(base_prob * tau, 0.0)

            scores.append((goals_a, goals_b))
            probs.append(prob)

    probs = np.array(probs, dtype=float)

    total = probs.sum()

    if total <= 0:
        raise ValueError(
            "Score probability matrix has non-positive total probability. "
            "Try using a smaller absolute value for DIXON_COLES_RHO."
        )

    probs = probs / total

    return scores, probs


def get_score_matrix_for_match(team_a, team_b, elo):
    mu_a = expected_goals(team_a, team_b, elo)
    mu_b = expected_goals(team_b, team_a, elo)

    rho = DIXON_COLES_RHO if USE_DIXON_COLES else 0.0

    return score_probability_matrix(
        mu_a=mu_a,
        mu_b=mu_b,
        rho=rho,
        max_goals=MAX_GOALS_FOR_SCORE_MATRIX,
    )


# ------------------------------------------------------------
# Knockout prediction
# ------------------------------------------------------------

def predict_knockout_match_result(team_a, team_b, elo):
    """
    Predict a knockout match.

    Reports:
      - average 90-minute score
      - 90-minute W/D/L probabilities
      - most likely 90-minute score
      - estimated advancement probability after tie-breaks
    """
    scores, probs = get_score_matrix_for_match(team_a, team_b, elo)

    goals_a = np.array([s[0] for s in scores])
    goals_b = np.array([s[1] for s in scores])

    avg_goals_a = np.sum(goals_a * probs)
    avg_goals_b = np.sum(goals_b * probs)

    p_a_90 = np.sum(probs[goals_a > goals_b])
    p_draw_90 = np.sum(probs[goals_a == goals_b])
    p_b_90 = np.sum(probs[goals_b > goals_a])

    most_likely_idx = np.argmax(probs)
    most_likely_score = scores[most_likely_idx]
    most_likely_score_prob = probs[most_likely_idx]

    rating_a = elo.get(team_a, 1500)
    rating_b = elo.get(team_b, 1500)

    # Same tie-break logic as the main tournament script:
    # if tied after simulated normal time, the Elo-favoured side has
    # higher probability in extra time / penalties.
    p_a_tiebreak = expected_score(rating_a, rating_b)
    p_b_tiebreak = 1 - p_a_tiebreak

    p_a_advance = p_a_90 + p_draw_90 * p_a_tiebreak
    p_b_advance = p_b_90 + p_draw_90 * p_b_tiebreak

    predicted_winner = team_a if p_a_advance >= p_b_advance else team_b

    return {
        "team_a": team_a,
        "team_b": team_b,
        "elo_a": rating_a,
        "elo_b": rating_b,
        "avg_goals_a": avg_goals_a,
        "avg_goals_b": avg_goals_b,
        "p_team_a_win_90": p_a_90,
        "p_draw_90": p_draw_90,
        "p_team_b_win_90": p_b_90,
        "p_team_a_advance": p_a_advance,
        "p_team_b_advance": p_b_advance,
        "predicted_winner": predicted_winner,
        "most_likely_90_score": f"{most_likely_score[0]}-{most_likely_score[1]}",
        "most_likely_90_score_prob": most_likely_score_prob,
    }


def predict_knockout_matches_from_file(path, elo, round_filter="all_known"):
    """
    Predict all known knockout matches listed in knockout_matches.csv.
    """
    matches = load_knockout_matches_file(
        path=path,
        round_filter=round_filter,
    )

    if matches.empty:
        print(f"\nNo filled knockout matches found for round_filter={round_filter}.")
        return pd.DataFrame()

    rows = []

    for _, row in tqdm(
        matches.iterrows(),
        total=len(matches),
        desc="Predicting knockout matches",
    ):
        pred = predict_knockout_match_result(
            team_a=row["team_a"],
            team_b=row["team_b"],
            elo=elo,
        )

        rows.append(
            {
                "match_no": int(row["match_no"]),
                "round": row["round"],
                **pred,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Printing and output
# ------------------------------------------------------------

def print_elo_table(elo, n=20):
    elo_table = (
        pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {n} Elo ratings")
    print(elo_table.head(n).to_string(index=False))


def print_knockout_predictions(predictions):
    if predictions.empty:
        return

    out = predictions.copy()

    for col in ["elo_a", "elo_b", "avg_goals_a", "avg_goals_b"]:
        out[col] = out[col].round(2)

    percent_cols = [
        "p_team_a_win_90",
        "p_draw_90",
        "p_team_b_win_90",
        "p_team_a_advance",
        "p_team_b_advance",
        "most_likely_90_score_prob",
    ]

    for col in percent_cols:
        out[col] = (100 * out[col]).round(1)

    display_cols = [
        "match_no",
        "round",
        "team_a",
        "avg_goals_a",
        "avg_goals_b",
        "team_b",
        "p_team_a_win_90",
        "p_draw_90",
        "p_team_b_win_90",
        "p_team_a_advance",
        "p_team_b_advance",
        "predicted_winner",
        "most_likely_90_score",
        "most_likely_90_score_prob",
    ]

    print("\nPredicted knockout results")
    print(out[display_cols].to_string(index=False))


def save_predictions(predictions, output_path):
    if predictions.empty:
        print("\nNo predictions to save.")
        return

    predictions.to_csv(output_path, index=False)
    print(f"\nSaved knockout predictions to: {output_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ensure_knockout_matches_file(KNOCKOUT_MATCHES_PATH)

    print("Loading results...")
    results = load_results()

    if USE_LIVE_RESULTS_FOR_ELO:
        print(f"Loaded {len(results):,} matches from RESULTS_URL for live Elo.")
    else:
        print(
            f"Loaded {len(results):,} matches before "
            f"{CUTOFF_DATE.date()} for pre-tournament Elo."
        )

    print("Building Elo ratings...")
    elo = build_elo_ratings(results)

    print_elo_table(elo, n=20)

    print("\nKnockout prediction settings:")
    print(f"KNOCKOUT_MATCHES_PATH = {KNOCKOUT_MATCHES_PATH}")
    print(f"KNOCKOUT_PREDICT_ROUND = {KNOCKOUT_PREDICT_ROUND}")
    print(f"USE_DIXON_COLES = {USE_DIXON_COLES}")
    print(f"DIXON_COLES_RHO = {DIXON_COLES_RHO}")
    print(f"MAX_GOALS_FOR_SCORE_MATRIX = {MAX_GOALS_FOR_SCORE_MATRIX}")

    predictions = predict_knockout_matches_from_file(
        path=KNOCKOUT_MATCHES_PATH,
        elo=elo,
        round_filter=KNOCKOUT_PREDICT_ROUND,
    )

    print_knockout_predictions(predictions)

    output_path = "world_cup_knockout_match_predictions.csv"
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    main()