# world_cup_predictor_2026.py
# Compact teaching version: Elo + Dixon-Coles + Monte Carlo group simulation.
# Replace this file with your full official-bracket script when desired.

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CUTOFF_DATE = pd.Timestamp("2026-06-11")
RANDOM_STATE = 0
rng = np.random.default_rng(RANDOM_STATE)
N_SIMS = 10_000
RESULT_SOURCE = "results_url"
UPDATE_ELO_WITH_PLAYED_MATCHES = True
USE_DIXON_COLES = True
DIXON_COLES_RHO = -0.10
MAX_GOALS_FOR_SCORE_MATRIX = 10
HOSTS = {"United States", "Mexico", "Canada"}

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
TEAM_NAME_MAP = {"USA": "United States", "Türkiye": "Turkey", "Turkiye": "Turkey", "Czech Republic": "Czechia", "Bosnia and Herzegovina": "Bosnia-Herzegovina", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast", "Curaçao": "Curacao", "DR Congo": "Congo DR", "Democratic Republic of the Congo": "Congo DR", "Korea Republic": "South Korea"}


def normalize_team_name(name):
    if name is None:
        return None
    name = str(name).strip()
    if name == "" or name.lower() == "nan":
        return None
    return TEAM_NAME_MAP.get(name, name)


def all_group_teams():
    return {t for teams in GROUPS.values() for t in teams}


def team_group_lookup():
    return {team: group for group, teams in GROUPS.items() for team in teams}


def load_results():
    df = pd.read_csv(RESULTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    return df.sort_values("date").reset_index(drop=True)


def fetch_played_matches_from_results_url(results):
    teams = all_group_teams()
    lookup = team_group_lookup()
    wc = results[(results["date"] >= CUTOFF_DATE) & (results["tournament"].astype(str).str.lower() == "fifa world cup")].copy()
    out = []
    for _, r in wc.iterrows():
        a, b = normalize_team_name(r["home_team"]), normalize_team_name(r["away_team"])
        if a in teams and b in teams and lookup[a] == lookup[b]:
            out.append({"date": r["date"].strftime("%Y-%m-%d"), "team_a": a, "team_b": b, "goals_a": int(r["home_score"]), "goals_b": int(r["away_score"]), "source": "results_url"})
    return out


def get_played_matches(results):
    if RESULT_SOURCE == "results_url":
        return fetch_played_matches_from_results_url(results)
    return []


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def actual_result(ga, gb):
    return 1.0 if ga > gb else 0.5 if ga == gb else 0.0


def margin_multiplier(diff):
    diff = abs(diff)
    return 1.0 if diff <= 1 else math.log(diff + 1)


def tournament_k_factor(t):
    t = str(t).lower()
    if t == "fifa world cup":
        return 60
    if "qualification" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def build_elo(matches, base=1500, home_adv=60):
    ratings = defaultdict(lambda: base)
    for _, r in matches.iterrows():
        home, away = r["home_team"], r["away_team"]
        if home is None or away is None or pd.isna(r["home_score"]) or pd.isna(r["away_score"]):
            continue
        hs, aas = int(r["home_score"]), int(r["away_score"])
        bonus = 0 if bool(r["neutral"]) else home_adv
        exp_home = expected_score(ratings[home] + bonus, ratings[away])
        res_home = actual_result(hs, aas)
        k = tournament_k_factor(r["tournament"]) * margin_multiplier(hs - aas)
        change = k * (res_home - exp_home)
        ratings[home] += change
        ratings[away] -= change
    return dict(ratings)


def choose_results_for_elo(results):
    if UPDATE_ELO_WITH_PLAYED_MATCHES:
        return results.copy()
    return results[results["date"] < CUTOFF_DATE].copy()


def expected_goals(team, opponent, elo):
    gap = (elo.get(team, 1500) - elo.get(opponent, 1500)) / 100
    host = 1 if team in HOSTS else 0
    return float(np.clip(math.exp(0.17 + 0.19 * gap + 0.17 * host), 0.15, 4.5))


def poisson_pmf(k, mu):
    return math.exp(-mu) * mu**k / math.factorial(k)


def dc_tau(x, y, mx, my, rho):
    if x == 0 and y == 0:
        return 1 - mx * my * rho
    if x == 0 and y == 1:
        return 1 + mx * rho
    if x == 1 and y == 0:
        return 1 + my * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(team_a, team_b, elo):
    ma, mb = expected_goals(team_a, team_b, elo), expected_goals(team_b, team_a, elo)
    rho = DIXON_COLES_RHO if USE_DIXON_COLES else 0.0
    scores, probs = [], []
    for a in range(MAX_GOALS_FOR_SCORE_MATRIX + 1):
        for b in range(MAX_GOALS_FOR_SCORE_MATRIX + 1):
            p = poisson_pmf(a, ma) * poisson_pmf(b, mb) * dc_tau(a, b, ma, mb, rho)
            scores.append((a, b))
            probs.append(max(p, 0.0))
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    return scores, probs


def simulate_score(team_a, team_b, elo):
    scores, probs = score_matrix(team_a, team_b, elo)
    return scores[rng.choice(len(scores), p=probs)]


def played_lookup(played):
    out = {}
    for m in played:
        a, b = m["team_a"], m["team_b"]
        out[frozenset([a, b])] = {a: m["goals_a"], b: m["goals_b"]}
    return out


def simulate_group(group, teams, elo, lookup):
    rows = {t: {"group": group, "team": t, "points": 0, "gf": 0, "ga": 0, "wins": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            key = frozenset([a, b])
            if key in lookup:
                ga, gb = lookup[key][a], lookup[key][b]
            else:
                ga, gb = simulate_score(a, b, elo)
            rows[a]["gf"] += ga; rows[a]["ga"] += gb
            rows[b]["gf"] += gb; rows[b]["ga"] += ga
            if ga > gb:
                rows[a]["points"] += 3; rows[a]["wins"] += 1
            elif gb > ga:
                rows[b]["points"] += 3; rows[b]["wins"] += 1
            else:
                rows[a]["points"] += 1; rows[b]["points"] += 1
    table = pd.DataFrame(rows.values())
    table["gd"] = table["gf"] - table["ga"]
    table["elo"] = table["team"].map(lambda t: elo.get(t, 1500))
    table = table.sort_values(["points", "gd", "gf", "wins", "elo"], ascending=False).reset_index(drop=True)
    table["group_rank"] = np.arange(1, len(table) + 1)
    return table


def simulate_tournament(elo, played):
    lookup = played_lookup(played)
    full = pd.concat([simulate_group(g, teams, elo, lookup) for g, teams in GROUPS.items()], ignore_index=True)
    winners = full[full["group_rank"] == 1]
    runners = full[full["group_rank"] == 2]
    thirds = full[full["group_rank"] == 3].sort_values(["points", "gd", "gf", "wins", "elo"], ascending=False).head(8)
    qualifiers = pd.concat([winners, runners, thirds], ignore_index=True)["team"].tolist()
    qualifiers = sorted(qualifiers, key=lambda t: elo.get(t, 1500), reverse=True)
    alive = qualifiers
    while len(alive) > 1:
        next_round = []
        for i in range(len(alive) // 2):
            a, b = alive[i], alive[-(i + 1)]
            ga, gb = simulate_score(a, b, elo)
            if ga > gb:
                next_round.append(a)
            elif gb > ga:
                next_round.append(b)
            else:
                next_round.append(a if np.random.random() < expected_score(elo.get(a, 1500), elo.get(b, 1500)) else b)
        alive = next_round
    return alive[0], qualifiers


def predict_group_fixtures(elo):
    rows = []
    for group, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                scores, probs = score_matrix(a, b, elo)
                ga = np.array([s[0] for s in scores]); gb = np.array([s[1] for s in scores])
                idx = int(np.argmax(probs))
                rows.append({"group": group, "team_a": a, "team_b": b, "avg_goals_a": float((ga * probs).sum()), "avg_goals_b": float((gb * probs).sum()), "p_team_a_win": float(probs[ga > gb].sum()), "p_draw": float(probs[ga == gb].sum()), "p_team_b_win": float(probs[gb > ga].sum()), "most_likely_score": f"{scores[idx][0]}-{scores[idx][1]}", "most_likely_score_prob": float(probs[idx])})
    return pd.DataFrame(rows)


def main():
    print("Loading results...")
    results = load_results()
    played = get_played_matches(results)
    print(f"Detected {len(played)} completed group-stage matches.")
    elo = build_elo(choose_results_for_elo(results))
    fixtures = predict_group_fixtures(elo)
    fixtures.to_csv(OUTPUT_DIR / "world_cup_group_fixture_predictions.csv", index=False)
    counts = Counter(); make_r32 = Counter()
    for _ in tqdm(range(N_SIMS), desc="Simulating tournaments"):
        champ, qualifiers = simulate_tournament(elo, played)
        counts[champ] += 1
        for q in qualifiers:
            make_r32[q] += 1
    teams = sorted(all_group_teams())
    probs = pd.DataFrame({"team": teams, "elo": [elo.get(t, 1500) for t in teams], "make_r32": [make_r32[t] / N_SIMS for t in teams], "win_world_cup": [counts[t] / N_SIMS for t in teams]}).sort_values("win_world_cup", ascending=False)
    probs.to_csv(OUTPUT_DIR / "world_cup_prediction_probabilities.csv", index=False)
    print(probs.head(20).to_string(index=False))
    print("\nSaved outputs to outputs/.")


if __name__ == "__main__":
    main()
