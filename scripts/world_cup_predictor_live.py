# world_cup_predictor_live_2026.py
# ------------------------------------------------------------
# Predict the 2026 FIFA World Cup winner using:
#   1. Historical international results
#   2. Elo ratings
#   3. Poisson / Dixon-Coles score simulation
#   4. Monte Carlo tournament simulation
#   5. Live completed group-stage results from RESULTS_URL, API, or manual CSV
#   6. Official-style 2026 knockout bracket using third_place_mapping.csv
#
# Required files:
#   - third_place_mapping.csv
#   - manual_results.csv  [optional; auto-created if missing]
#
# Install:
#   pip install pandas numpy requests tqdm
#
# Optional API key:
#   export API_FOOTBALL_KEY="your_api_key_here"
#
# Run:
#   python world_cup_predictor_live_2026.py
# ------------------------------------------------------------

import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

RANDOM_STATE = 0
rng = np.random.default_rng(RANDOM_STATE)

N_SIMS = 50_000

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Tournament starts June 11, 2026.
CUTOFF_DATE = pd.Timestamp("2026-06-11")

HOSTS = {"United States", "Mexico", "Canada"}

# Live mode recommendation:
#   RESULT_SOURCE = "results_url"
#   UPDATE_ELO_WITH_PLAYED_MATCHES = True
#
# Options:
#   "results_url" -> extract completed World Cup group matches from RESULTS_URL
#   "api"         -> use API-Football only
#   "manual"      -> use manual_results.csv only
#   "both"        -> use API-Football, then manual_results.csv overrides duplicates
#   "results_url_plus_manual" -> use RESULTS_URL, then manual_results.csv overrides
RESULT_SOURCE = "results_url"

# If True:
#   - with RESULT_SOURCE="results_url", Elo uses all matches currently in RESULTS_URL.
#   - with API/manual modes, played World Cup matches are appended to pre-tournament results.
#
# If False:
#   - Elo uses only matches before CUTOFF_DATE.
UPDATE_ELO_WITH_PLAYED_MATCHES = True

MANUAL_RESULTS_PATH = "manual_results.csv"
THIRD_PLACE_MAPPING_PATH = "third_place_mapping.csv"

USE_API_FOOTBALL_RESULTS = True
API_FOOTBALL_LEAGUE_ID = 1
API_FOOTBALL_SEASON = 2026


# Dixon-Coles settings.
USE_DIXON_COLES = True

# rho = 0 gives independent Poisson.
# Negative rho usually boosts 0-0 and 1-1 while reducing 1-0 and 0-1.
#DIXON_COLES_RHO = -0.10

# computed from all WC data:
#DIXON_COLES_RHO = -0.05

# computed from 40 years WC data:
DIXON_COLES_RHO = -0.043948



MAX_GOALS_FOR_SCORE_MATRIX = 10


# ------------------------------------------------------------
# 2026 World Cup groups
# ------------------------------------------------------------

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}


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


def all_group_teams(groups):
    return {team for group_teams in groups.values() for team in group_teams}


def team_group_lookup(groups):
    lookup = {}

    for group_name, teams in groups.items():
        for team in teams:
            lookup[team] = group_name

    return lookup


# ------------------------------------------------------------
# Manual results file
# ------------------------------------------------------------

def ensure_manual_results_file(path):
    if os.path.exists(path):
        return

    empty = pd.DataFrame(
        columns=["date", "team_a", "team_b", "goals_a", "goals_b"]
    )

    empty.to_csv(path, index=False)
    print(f"Created empty manual results file: {path}")


def fetch_played_matches_from_manual_file(path):
    if not os.path.exists(path):
        print(f"No manual results file found at {path}.")
        return []

    manual_df = pd.read_csv(path)

    required_cols = {"date", "team_a", "team_b", "goals_a", "goals_b"}
    missing = required_cols - set(manual_df.columns)

    if missing:
        raise ValueError(
            f"Manual results file is missing columns: {missing}. "
            f"Expected columns: {required_cols}"
        )

    played_matches = []

    for _, row in manual_df.iterrows():
        if pd.isna(row["team_a"]) or pd.isna(row["team_b"]):
            continue

        if pd.isna(row["goals_a"]) or pd.isna(row["goals_b"]):
            continue

        played_matches.append(
            {
                "date": str(row["date"])[:10],
                "team_a": normalize_team_name(row["team_a"]),
                "team_b": normalize_team_name(row["team_b"]),
                "goals_a": int(row["goals_a"]),
                "goals_b": int(row["goals_b"]),
            }
        )

    return played_matches


# ------------------------------------------------------------
# Historical / live results data
# ------------------------------------------------------------

def load_results():
    """
    Load the full available RESULTS_URL.

    Important:
    This function no longer cuts off at CUTOFF_DATE.
    The cutoff is applied later depending on whether you want
    pre-tournament or live Elo.
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

    results = results.sort_values("date").reset_index(drop=True)

    return results


def fetch_played_matches_from_results_url(results, groups):
    """
    Extract completed 2026 World Cup group-stage matches from RESULTS_URL.

    This assumes RESULTS_URL is updated after each match.
    Only group-stage matches are kept: both teams must be in GROUPS
    and in the same group.
    """
    group_teams = all_group_teams(groups)
    group_lookup = team_group_lookup(groups)

    tournament_names = results["tournament"].astype(str).str.lower()

    wc_matches = results[
        (results["date"] >= CUTOFF_DATE)
        & (tournament_names == "fifa world cup")
    ].copy()

    played_matches = []

    for _, row in wc_matches.iterrows():
        team_a = normalize_team_name(row["home_team"])
        team_b = normalize_team_name(row["away_team"])

        if team_a not in group_teams:
            continue

        if team_b not in group_teams:
            continue

        if group_lookup[team_a] != group_lookup[team_b]:
            continue

        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        played_matches.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "team_a": team_a,
                "team_b": team_b,
                "goals_a": int(row["home_score"]),
                "goals_b": int(row["away_score"]),
                "source": "results_url",
            }
        )

    return played_matches


# ------------------------------------------------------------
# API-Football result lookup
# ------------------------------------------------------------

def fetch_played_matches_from_api_football(groups):
    if not USE_API_FOOTBALL_RESULTS:
        return []

    api_key = os.getenv("API_FOOTBALL_KEY")

    if not api_key:
        print("No API_FOOTBALL_KEY found. Using manual/results-url results only.")
        return []

    url = "https://v3.football.api-sports.io/fixtures"

    headers = {
        "x-apisports-key": api_key,
    }

    params = {
        "league": API_FOOTBALL_LEAGUE_ID,
        "season": API_FOOTBALL_SEASON,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    if "response" not in payload:
        raise ValueError(f"Unexpected API response: {payload}")

    group_teams = all_group_teams(groups)
    group_lookup = team_group_lookup(groups)

    played_matches = []

    for item in payload["response"]:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})

        status = fixture.get("status", {})
        status_short = status.get("short")

        if status_short not in {"FT", "AET", "PEN"}:
            continue

        home_team = normalize_team_name(teams.get("home", {}).get("name"))
        away_team = normalize_team_name(teams.get("away", {}).get("name"))

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        if home_team is None or away_team is None:
            continue

        if home_goals is None or away_goals is None:
            continue

        if home_team not in group_teams:
            continue

        if away_team not in group_teams:
            continue

        if group_lookup[home_team] != group_lookup[away_team]:
            continue

        played_matches.append(
            {
                "date": str(fixture.get("date", ""))[:10],
                "team_a": home_team,
                "team_b": away_team,
                "goals_a": int(home_goals),
                "goals_b": int(away_goals),
            }
        )

    return played_matches


def get_played_matches(groups, results=None):
    """
    Get completed group-stage matches from the configured source.

    RESULT_SOURCE options:
        "results_url" -> use up-to-date RESULTS_URL dataframe
        "api"         -> API-Football only
        "manual"      -> manual_results.csv only
        "both"        -> API-Football plus manual overrides
        "results_url_plus_manual" -> RESULTS_URL plus manual overrides
    """
    valid_sources = {
        "results_url",
        "api",
        "manual",
        "both",
        "results_url_plus_manual",
    }

    if RESULT_SOURCE not in valid_sources:
        raise ValueError(
            f"RESULT_SOURCE must be one of: {sorted(valid_sources)}"
        )

    results_url_matches = []
    api_matches = []
    manual_matches = []

    if RESULT_SOURCE in {"results_url", "results_url_plus_manual"}:
        if results is None:
            raise ValueError(
                "results must be provided when RESULT_SOURCE uses RESULTS_URL."
            )

        results_url_matches = fetch_played_matches_from_results_url(
            results=results,
            groups=groups,
        )

    if RESULT_SOURCE in {"api", "both"}:
        try:
            api_matches = fetch_played_matches_from_api_football(groups)
        except requests.RequestException as exc:
            print(f"API-Football request failed: {exc}")
            print("Continuing with manual results if available.")
            api_matches = []
        except Exception as exc:
            print(f"Could not parse API-Football response: {exc}")
            print("Continuing with manual results if available.")
            api_matches = []

    if RESULT_SOURCE in {"manual", "both", "results_url_plus_manual"}:
        manual_matches = fetch_played_matches_from_manual_file(MANUAL_RESULTS_PATH)

    combined = {}

    # RESULTS_URL first.
    for match in results_url_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])
        key = frozenset([team_a, team_b])

        combined[key] = {
            **match,
            "team_a": team_a,
            "team_b": team_b,
            "source": "results_url",
        }

    # API second.
    for match in api_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])
        key = frozenset([team_a, team_b])

        combined[key] = {
            **match,
            "team_a": team_a,
            "team_b": team_b,
            "source": "api",
        }

    # Manual last, so manual rows override everything else.
    for match in manual_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])
        key = frozenset([team_a, team_b])

        combined[key] = {
            **match,
            "team_a": team_a,
            "team_b": team_b,
            "source": "manual",
        }

    played_matches = list(combined.values())

    played_matches = sorted(
        played_matches,
        key=lambda x: (
            str(x.get("date", "")),
            x["team_a"],
            x["team_b"],
        ),
    )

    return played_matches


def choose_results_for_elo(results, played_matches):
    """
    Choose the match dataframe used to build Elo ratings.

    This avoids double-counting World Cup matches.

    If RESULT_SOURCE uses RESULTS_URL:
        - UPDATE_ELO_WITH_PLAYED_MATCHES=True:
            use all matches currently in RESULTS_URL
        - UPDATE_ELO_WITH_PLAYED_MATCHES=False:
            use only pre-tournament matches

    If RESULT_SOURCE uses API/manual:
        - start with pre-tournament results
        - optionally append played_matches
    """
    pre_tournament_results = results[results["date"] < CUTOFF_DATE].copy()

    if RESULT_SOURCE in {"results_url", "results_url_plus_manual"}:
        if UPDATE_ELO_WITH_PLAYED_MATCHES:
            print("\nUsing all available RESULTS_URL matches for Elo.")
            return results.copy()

        print("\nUsing pre-tournament Elo only.")
        return pre_tournament_results

    if UPDATE_ELO_WITH_PLAYED_MATCHES:
        print("\nAdding completed World Cup group matches to pre-tournament Elo data...")
        return append_played_matches_to_results(
            pre_tournament_results,
            played_matches,
        )

    print("\nUsing pre-tournament Elo only.")
    return pre_tournament_results


# ------------------------------------------------------------
# Elo model
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


def build_elo_ratings(matches, base_rating=1500, home_advantage=60):
    ratings = defaultdict(lambda: base_rating)

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_rating = ratings[home]
        away_rating = ratings[away]

        home_bonus = 0 if bool(row["neutral"]) else home_advantage

        exp_home = expected_score(home_rating + home_bonus, away_rating)
        res_home = actual_result(row["home_score"], row["away_score"])

        goal_diff = row["home_score"] - row["away_score"]

        k = tournament_k_factor(row["tournament"])
        k *= margin_multiplier(goal_diff)

        change = k * (res_home - exp_home)

        ratings[home] += change
        ratings[away] -= change

    return dict(ratings)


def append_played_matches_to_results(results, played_matches):
    rows = []

    for match in played_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        rows.append(
            {
                "date": pd.Timestamp(match.get("date", "2026-06-11")),
                "home_team": team_a,
                "away_team": team_b,
                "home_score": int(match["goals_a"]),
                "away_score": int(match["goals_b"]),
                "tournament": "FIFA World Cup",
                "city": None,
                "country": None,
                "neutral": True,
            }
        )

    if not rows:
        return results

    played_df = pd.DataFrame(rows)

    combined = pd.concat([results, played_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    return combined


# ------------------------------------------------------------
# Expected goals and Dixon-Coles score model
# ------------------------------------------------------------

def expected_goals(team, opponent, elo):
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


def simulate_score(team_a, team_b, elo):
    if USE_DIXON_COLES:
        scores, probs = get_score_matrix_for_match(team_a, team_b, elo)
        idx = rng.choice(len(scores), p=probs)
        return scores[idx]

    mu_a = expected_goals(team_a, team_b, elo)
    mu_b = expected_goals(team_b, team_a, elo)

    goals_a = rng.poisson(mu_a)
    goals_b = rng.poisson(mu_b)

    return int(goals_a), int(goals_b)


def simulate_group_match(team_a, team_b, elo):
    return simulate_score(team_a, team_b, elo)


def simulate_knockout_match(team_a, team_b, elo):
    goals_a, goals_b = simulate_score(team_a, team_b, elo)

    if goals_a > goals_b:
        return team_a

    if goals_b > goals_a:
        return team_b

    rating_a = elo.get(team_a, 1500)
    rating_b = elo.get(team_b, 1500)

    p_a = expected_score(rating_a, rating_b)

    return team_a if rng.random() < p_a else team_b


# ------------------------------------------------------------
# Played-match lookup and validation
# ------------------------------------------------------------

def build_played_match_lookup(played_matches):
    lookup = {}

    for match in played_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        key = frozenset([team_a, team_b])

        if key in lookup:
            raise ValueError(f"Duplicate played match found: {team_a} vs {team_b}")

        lookup[key] = {
            team_a: int(match["goals_a"]),
            team_b: int(match["goals_b"]),
        }

    return lookup


def validate_played_matches(groups, played_matches):
    lookup = team_group_lookup(groups)

    for match in played_matches:
        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        if team_a not in lookup:
            raise ValueError(f"{team_a} is not in GROUPS.")

        if team_b not in lookup:
            raise ValueError(f"{team_b} is not in GROUPS.")

        if lookup[team_a] != lookup[team_b]:
            raise ValueError(
                f"{team_a} and {team_b} are not in the same group. "
                "This script currently treats fetched/played matches "
                "as group-stage results."
            )


# ------------------------------------------------------------
# Group-stage simulation
# ------------------------------------------------------------

def simulate_group(group_name, teams, elo, played_lookup=None):
    if played_lookup is None:
        played_lookup = {}

    stats = {
        team: {
            "group": group_name,
            "team": team,
            "played": 0,
            "real_played": 0,
            "simulated_played": 0,
            "points": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "wins": 0,
        }
        for team in teams
    }

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a = teams[i]
            team_b = teams[j]

            match_key = frozenset([team_a, team_b])

            if match_key in played_lookup:
                goals_a = played_lookup[match_key][team_a]
                goals_b = played_lookup[match_key][team_b]
                is_real = True
            else:
                goals_a, goals_b = simulate_group_match(team_a, team_b, elo)
                is_real = False

            stats[team_a]["played"] += 1
            stats[team_b]["played"] += 1

            if is_real:
                stats[team_a]["real_played"] += 1
                stats[team_b]["real_played"] += 1
            else:
                stats[team_a]["simulated_played"] += 1
                stats[team_b]["simulated_played"] += 1

            stats[team_a]["gf"] += goals_a
            stats[team_a]["ga"] += goals_b

            stats[team_b]["gf"] += goals_b
            stats[team_b]["ga"] += goals_a

            if goals_a > goals_b:
                stats[team_a]["points"] += 3
                stats[team_a]["wins"] += 1
            elif goals_b > goals_a:
                stats[team_b]["points"] += 3
                stats[team_b]["wins"] += 1
            else:
                stats[team_a]["points"] += 1
                stats[team_b]["points"] += 1

    table = pd.DataFrame(stats.values())

    table["gd"] = table["gf"] - table["ga"]
    table["elo"] = table["team"].map(lambda x: elo.get(x, 1500))

    table = table.sort_values(
        ["points", "gd", "gf", "wins", "elo"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    table["group_rank"] = np.arange(1, len(table) + 1)

    return table


def simulate_group_stage(groups, elo, played_lookup=None):
    if played_lookup is None:
        played_lookup = {}

    all_group_tables = []

    for group_name, teams in groups.items():
        group_table = simulate_group(
            group_name=group_name,
            teams=teams,
            elo=elo,
            played_lookup=played_lookup,
        )
        all_group_tables.append(group_table)

    full_table = pd.concat(all_group_tables, ignore_index=True)

    group_winners = full_table[full_table["group_rank"] == 1].copy()
    runners_up = full_table[full_table["group_rank"] == 2].copy()
    third_place = full_table[full_table["group_rank"] == 3].copy()

    best_thirds = third_place.sort_values(
        ["points", "gd", "gf", "wins", "elo"],
        ascending=[False, False, False, False, False],
    ).head(8)

    qualifiers = pd.concat(
        [group_winners, runners_up, best_thirds],
        ignore_index=True,
    )

    return full_table, qualifiers


# ------------------------------------------------------------
# Actual 2026 knockout bracket with third-place mapping
# ------------------------------------------------------------

def load_third_place_mapping(path=THIRD_PLACE_MAPPING_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Create it first using create_third_place_mapping_csv.py."
        )

    mapping = pd.read_csv(path)

    required_cols = {
        "qualified_groups",
        "slot_1A",
        "slot_1B",
        "slot_1D",
        "slot_1E",
        "slot_1G",
        "slot_1I",
        "slot_1K",
        "slot_1L",
    }

    missing = required_cols - set(mapping.columns)

    if missing:
        raise ValueError(
            f"Third-place mapping file is missing columns: {missing}"
        )

    mapping["qualified_groups"] = (
        mapping["qualified_groups"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.upper()
        .apply(lambda x: "".join(sorted(x)))
    )

    for col in [
        "slot_1A",
        "slot_1B",
        "slot_1D",
        "slot_1E",
        "slot_1G",
        "slot_1I",
        "slot_1K",
        "slot_1L",
    ]:
        mapping[col] = (
            mapping[col]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
        )

    if len(mapping) != 495:
        print(
            f"WARNING: third-place mapping has {len(mapping)} rows. "
            "The full table should have 495 rows."
        )

    if mapping["qualified_groups"].duplicated().any():
        duplicates = mapping.loc[
            mapping["qualified_groups"].duplicated(),
            "qualified_groups",
        ].tolist()
        raise ValueError(
            f"Duplicate qualified_groups keys in mapping file: {duplicates[:10]}"
        )

    return mapping


def get_third_place_slot_mapping(qualified_third_groups, mapping_df):
    key = "".join(sorted(qualified_third_groups))

    row = mapping_df[mapping_df["qualified_groups"] == key]

    if row.empty:
        raise ValueError(
            f"No third-place mapping found for qualified groups: {key}. "
            "Check that third_place_mapping.csv contains the full 495-row table."
        )

    row = row.iloc[0]

    return {
        "1A": row["slot_1A"],
        "1B": row["slot_1B"],
        "1D": row["slot_1D"],
        "1E": row["slot_1E"],
        "1G": row["slot_1G"],
        "1I": row["slot_1I"],
        "1K": row["slot_1K"],
        "1L": row["slot_1L"],
    }


def build_qualifier_slot_lookup(group_table, qualifiers, third_place_mapping_df):
    slot_to_team = {}

    for _, row in group_table.iterrows():
        group = row["group"]
        rank = int(row["group_rank"])
        team = row["team"]

        if rank in {1, 2, 3}:
            slot_to_team[f"{rank}{group}"] = team

    qualified_thirds = qualifiers[qualifiers["group_rank"] == 3].copy()
    qualified_third_groups = sorted(qualified_thirds["group"].tolist())

    if len(qualified_third_groups) != 8:
        raise ValueError(
            f"Expected 8 qualified third-place teams, got {len(qualified_third_groups)}"
        )

    third_slot_mapping = get_third_place_slot_mapping(
        qualified_third_groups=qualified_third_groups,
        mapping_df=third_place_mapping_df,
    )

    return slot_to_team, third_slot_mapping


def simulate_actual_bracket(group_table, qualifiers, elo, third_place_mapping_df):
    slot_to_team, third_slot_mapping = build_qualifier_slot_lookup(
        group_table=group_table,
        qualifiers=qualifiers,
        third_place_mapping_df=third_place_mapping_df,
    )

    def team(slot):
        return slot_to_team[slot]

    round32_matches = {
        73: ("2A", "2B"),
        74: ("1E", third_slot_mapping["1E"]),
        75: ("1F", "2C"),
        76: ("1C", "2F"),
        77: ("1I", third_slot_mapping["1I"]),
        78: ("2E", "2I"),
        79: ("1A", third_slot_mapping["1A"]),
        80: ("1L", third_slot_mapping["1L"]),
        81: ("1D", third_slot_mapping["1D"]),
        82: ("1G", third_slot_mapping["1G"]),
        83: ("2K", "2L"),
        84: ("1H", "2J"),
        85: ("1B", third_slot_mapping["1B"]),
        86: ("1J", "2H"),
        87: ("1K", third_slot_mapping["1K"]),
        88: ("2D", "2G"),
    }

    winners = {}
    round32_teams = []

    for match_no, (slot_a, slot_b) in round32_matches.items():
        team_a = team(slot_a)
        team_b = team(slot_b)

        round32_teams.extend([team_a, team_b])

        winners[match_no] = simulate_knockout_match(team_a, team_b, elo)

    round16_matches = {
        89: (74, 77),
        90: (73, 75),
        91: (76, 78),
        92: (79, 80),
        93: (83, 84),
        94: (81, 82),
        95: (86, 88),
        96: (85, 87),
    }

    for match_no, (m_a, m_b) in round16_matches.items():
        winners[match_no] = simulate_knockout_match(
            winners[m_a],
            winners[m_b],
            elo,
        )

    quarterfinal_matches = {
        97: (89, 90),
        98: (93, 94),
        99: (91, 92),
        100: (95, 96),
    }

    for match_no, (m_a, m_b) in quarterfinal_matches.items():
        winners[match_no] = simulate_knockout_match(
            winners[m_a],
            winners[m_b],
            elo,
        )

    semifinal_matches = {
        101: (97, 98),
        102: (99, 100),
    }

    for match_no, (m_a, m_b) in semifinal_matches.items():
        winners[match_no] = simulate_knockout_match(
            winners[m_a],
            winners[m_b],
            elo,
        )

    winners[104] = simulate_knockout_match(
        winners[101],
        winners[102],
        elo,
    )

    champion = winners[104]

    path = {
        "round32": round32_teams,
        "round16": [winners[m] for m in range(73, 89)],
        "quarterfinals": [winners[m] for m in range(89, 97)],
        "semifinals": [winners[m] for m in range(97, 101)],
        "final": [winners[101], winners[102]],
        "champion": champion,
        "match_winners": winners,
    }

    return champion, path


def simulate_tournament(groups, elo, played_lookup=None, third_place_mapping_df=None):
    if third_place_mapping_df is None:
        third_place_mapping_df = load_third_place_mapping()

    group_table, qualifiers = simulate_group_stage(
        groups=groups,
        elo=elo,
        played_lookup=played_lookup,
    )

    champion, path = simulate_actual_bracket(
        group_table=group_table,
        qualifiers=qualifiers,
        elo=elo,
        third_place_mapping_df=third_place_mapping_df,
    )

    path["group_table"] = group_table
    path["qualifiers"] = qualifiers

    return champion, path


# ------------------------------------------------------------
# Monte Carlo simulation
# ------------------------------------------------------------

def run_simulations(groups, elo, n_sims=50_000, played_matches=None):
    if played_matches is None:
        played_matches = []

    validate_played_matches(groups, played_matches)
    played_lookup = build_played_match_lookup(played_matches)

    third_place_mapping_df = load_third_place_mapping(THIRD_PLACE_MAPPING_PATH)

    stage_counts = {
        "round32": Counter(),
        "round16": Counter(),
        "quarterfinals": Counter(),
        "semifinals": Counter(),
        "final": Counter(),
        "champion": Counter(),
    }

    for _ in tqdm(range(n_sims), desc="Simulating tournaments"):
        champion, path = simulate_tournament(
            groups=groups,
            elo=elo,
            played_lookup=played_lookup,
            third_place_mapping_df=third_place_mapping_df,
        )

        for stage in ["round32", "round16", "quarterfinals", "semifinals", "final"]:
            for team in path[stage]:
                stage_counts[stage][team] += 1

        stage_counts["champion"][champion] += 1

    teams = sorted(all_group_teams(groups))

    rows = []

    for team in teams:
        rows.append(
            {
                "team": team,
                "elo": elo.get(team, 1500),
                "make_r32": stage_counts["round32"][team] / n_sims,
                "make_r16": stage_counts["round16"][team] / n_sims,
                "make_qf": stage_counts["quarterfinals"][team] / n_sims,
                "make_sf": stage_counts["semifinals"][team] / n_sims,
                "make_final": stage_counts["final"][team] / n_sims,
                "win_world_cup": stage_counts["champion"][team] / n_sims,
            }
        )

    probs = pd.DataFrame(rows)

    probs = probs.sort_values(
        ["win_world_cup", "make_final", "make_sf", "elo"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return probs


# ------------------------------------------------------------
# Current real group tables
# ------------------------------------------------------------

def current_real_group_tables(groups, played_matches):
    validate_played_matches(groups, played_matches)
    played_lookup = build_played_match_lookup(played_matches)

    all_tables = []

    for group_name, teams in groups.items():
        stats = {
            team: {
                "group": group_name,
                "team": team,
                "played": 0,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "wins": 0,
            }
            for team in teams
        }

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]

                key = frozenset([team_a, team_b])

                if key not in played_lookup:
                    continue

                goals_a = played_lookup[key][team_a]
                goals_b = played_lookup[key][team_b]

                stats[team_a]["played"] += 1
                stats[team_b]["played"] += 1

                stats[team_a]["gf"] += goals_a
                stats[team_a]["ga"] += goals_b

                stats[team_b]["gf"] += goals_b
                stats[team_b]["ga"] += goals_a

                if goals_a > goals_b:
                    stats[team_a]["points"] += 3
                    stats[team_a]["wins"] += 1
                elif goals_b > goals_a:
                    stats[team_b]["points"] += 3
                    stats[team_b]["wins"] += 1
                else:
                    stats[team_a]["points"] += 1
                    stats[team_b]["points"] += 1

        table = pd.DataFrame(stats.values())
        table["gd"] = table["gf"] - table["ga"]

        table = table.sort_values(
            ["points", "gd", "gf", "wins"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        table["group_rank_now"] = np.arange(1, len(table) + 1)

        all_tables.append(table)

    return pd.concat(all_tables, ignore_index=True)


# ------------------------------------------------------------
# Individual group fixture predictions
# ------------------------------------------------------------

def predict_group_fixture_exact(team_a, team_b, elo):
    scores, probs = get_score_matrix_for_match(team_a, team_b, elo)

    goals_a = np.array([s[0] for s in scores])
    goals_b = np.array([s[1] for s in scores])

    avg_goals_a = np.sum(goals_a * probs)
    avg_goals_b = np.sum(goals_b * probs)

    p_team_a_win = np.sum(probs[goals_a > goals_b])
    p_draw = np.sum(probs[goals_a == goals_b])
    p_team_b_win = np.sum(probs[goals_b > goals_a])

    most_likely_idx = np.argmax(probs)
    most_likely_score = scores[most_likely_idx]
    most_likely_score_prob = probs[most_likely_idx]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "avg_goals_a": avg_goals_a,
        "avg_goals_b": avg_goals_b,
        "p_team_a_win": p_team_a_win,
        "p_draw": p_draw,
        "p_team_b_win": p_team_b_win,
        "most_likely_score": f"{most_likely_score[0]}-{most_likely_score[1]}",
        "most_likely_score_prob": most_likely_score_prob,
    }


def predict_all_group_fixtures(groups, elo):
    rows = []
    fixtures = []

    for group_name, teams in groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                fixtures.append(
                    {
                        "group": group_name,
                        "team_a": teams[i],
                        "team_b": teams[j],
                    }
                )

    for fixture in tqdm(fixtures, desc="Predicting group fixtures"):
        pred = predict_group_fixture_exact(
            team_a=fixture["team_a"],
            team_b=fixture["team_b"],
            elo=elo,
        )

        rows.append(
            {
                "group": fixture["group"],
                **pred,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Diagnostics and printing
# ------------------------------------------------------------

def check_group_teams_have_ratings(groups, elo):
    teams = sorted(all_group_teams(groups))
    missing = [team for team in teams if team not in elo]

    if missing:
        print("\nWARNING: These teams were not found in the historical Elo data.")
        print("They will use the default rating of 1500 unless you fix their names:")
        for team in missing:
            print(f"  - {team}")

    return missing


def print_elo_table(elo, n=20):
    elo_table = (
        pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {n} Elo ratings")
    print(elo_table.head(n).to_string(index=False))


def print_played_matches(played_matches):
    if not played_matches:
        print("\nNo completed group-stage matches found yet.")
        return

    df = pd.DataFrame(played_matches)

    cols = ["date", "team_a", "goals_a", "goals_b", "team_b"]

    if "source" in df.columns:
        cols.append("source")

    df = df[cols]

    print(f"\nCompleted group-stage matches found: {len(df)}")
    print(df.to_string(index=False))


def print_current_group_tables(groups, played_matches):
    if not played_matches:
        print("\nNo played matches entered/fetched yet.")
        return

    table = current_real_group_tables(groups, played_matches)

    print("\nCurrent real group tables from fetched/manual results only")

    for group_name in sorted(groups.keys()):
        group_table = table[table["group"] == group_name].copy()

        print(f"\nGroup {group_name}")
        print(
            group_table[
                ["team", "played", "points", "gf", "ga", "gd", "wins"]
            ].to_string(index=False)
        )


def print_prediction_table(probs, n=25):
    display_cols = [
        "team",
        "elo",
        "make_r32",
        "make_r16",
        "make_qf",
        "make_sf",
        "make_final",
        "win_world_cup",
    ]

    out = probs[display_cols].head(n).copy()

    percent_cols = [
        "make_r32",
        "make_r16",
        "make_qf",
        "make_sf",
        "make_final",
        "win_world_cup",
    ]

    for col in percent_cols:
        out[col] = (100 * out[col]).round(2)

    out["elo"] = out["elo"].round(1)

    print(f"\nTop {n} predicted teams")
    print(out.to_string(index=False))


def print_group_fixture_predictions(fixtures, n=72):
    out = fixtures.copy()

    for col in ["avg_goals_a", "avg_goals_b"]:
        out[col] = out[col].round(2)

    for col in ["p_team_a_win", "p_draw", "p_team_b_win", "most_likely_score_prob"]:
        out[col] = (100 * out[col]).round(1)

    display_cols = [
        "group",
        "team_a",
        "avg_goals_a",
        "avg_goals_b",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "most_likely_score",
        "most_likely_score_prob",
    ]

    print("\nPredicted group-game results")
    print(out[display_cols].head(n).to_string(index=False))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ensure_manual_results_file(MANUAL_RESULTS_PATH)

    print("Loading results...")
    results = load_results()

    print(f"Loaded {len(results):,} matches from RESULTS_URL.")

    print("\nFetching completed World Cup group-stage matches...")
    print(f"Result source: {RESULT_SOURCE}")

    played_matches = get_played_matches(
        groups=GROUPS,
        results=results,
    )

    print_played_matches(played_matches)

    validate_played_matches(GROUPS, played_matches)

    print("\nLoading third-place mapping...")
    third_place_mapping_df = load_third_place_mapping(THIRD_PLACE_MAPPING_PATH)
    print(f"Loaded {len(third_place_mapping_df):,} third-place mapping rows.")

    print("\nScore model settings:")
    print(f"USE_DIXON_COLES = {USE_DIXON_COLES}")
    print(f"DIXON_COLES_RHO = {DIXON_COLES_RHO}")
    print(f"MAX_GOALS_FOR_SCORE_MATRIX = {MAX_GOALS_FOR_SCORE_MATRIX}")

    results_for_elo = choose_results_for_elo(
        results=results,
        played_matches=played_matches,
    )

    print("Building Elo ratings...")
    elo = build_elo_ratings(results_for_elo)

    print_elo_table(elo, n=20)

    check_group_teams_have_ratings(GROUPS, elo)

    print_current_group_tables(GROUPS, played_matches)

    print("\nPredicting individual group fixtures...")
    fixture_predictions = predict_all_group_fixtures(
        groups=GROUPS,
        elo=elo,
    )

    print_group_fixture_predictions(fixture_predictions)

    fixture_output_path = "world_cup_group_fixture_predictions.csv"
    fixture_predictions.to_csv(fixture_output_path, index=False)

    print(f"\nSaved group fixture predictions to: {fixture_output_path}")

    print(f"\nRunning {N_SIMS:,} World Cup simulations...")

    probs = run_simulations(
        groups=GROUPS,
        elo=elo,
        n_sims=N_SIMS,
        played_matches=played_matches,
    )

    print_prediction_table(probs, n=25)

    output_path = "world_cup_prediction_probabilities.csv"
    probs.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()