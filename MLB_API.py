# pip install requests pandas python-dateutil
import requests, pandas as pd, time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date
from dateutil import parser as dtparser

BASE = "https://statsapi.mlb.com/api/v1"
TEAM_ID_DODGERS = 119                  # Los Angeles Dodgers
SEASONS = range(1990, 2025)            # 1990..2024 inclusive

def session():
    s = requests.Session()
    r = Retry(total=5, connect=3, read=3, backoff_factor=1.5,
              status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

S = session()

def safe(d,*path,default=None):
    cur=d
    for k in path:
        if not isinstance(cur,dict) or k not in cur: return default
        cur=cur[k]
    return cur

def get_postseason_game_pks_for_team(season: int, team_id: int):
    """Return gamePk for postseason games in 'season' where team_id (Dodgers) played."""
    data = S.get(f"{BASE}/schedule/postseason", params={"season": season}, timeout=30).json()
    pks = set()
    for date_node in data.get("dates", []):
        for g in date_node.get("games", []):
            home_id = safe(g, "teams", "home", "team", "id")
            away_id = safe(g, "teams", "away", "team", "id")
            if g.get("gamePk") and (home_id == team_id or away_id == team_id):
                pks.add(g["gamePk"])
    return sorted(pks)

def boxscore_batting_components_for_team(gamePk: int, team_id_target: int, season: int):
    """
    Return ONLY batting rows for Dodgers with components needed to compute
    postseason player-season summaries (AB, H, 2B, 3B, HR, BB, HBP, SF) + per-game position.
    """
    box = S.get(f"{BASE}/game/{gamePk}/boxscore", timeout=30).json()
    rows=[]
    for side in ["home","away"]:
        team = safe(box, "teams", side, default={}) or {}
        teamId = safe(team,"team","id")
        if teamId != team_id_target:
            continue
        players = safe(team,"players",default={}) or {}
        for _, pdata in players.items():
            b = safe(pdata,"stats","batting",default={}) or {}
            if not b:
                continue
            person = pdata.get("person",{}) or {}
            posabbr = (pdata.get("position") or {}).get("abbreviation")
            rows.append({
                "season": season,
                "playerId": person.get("id"),
                "playerName": person.get("fullName"),
                "position": posabbr,
                # components
                "AB": b.get("atBats"),
                "H":  b.get("hits"),
                "2B": b.get("doubles"),
                "3B": b.get("triples"),
                "HR": b.get("homeRuns"),
                "BB": b.get("baseOnBalls"),
                "HBP": b.get("hitByPitch"),
                "SF": b.get("sacFlies"),
            })
    return rows

def fetch_birthdates(player_ids):
    """
    Fetch birthDate for a set of playerIds via /people.
    Returns dict {playerId: birthDate_str or None}.
    """
    ids = [str(int(pid)) for pid in player_ids if pd.notna(pid)]
    out = {}
    chunk_size = 100
    for i in range(0, len(ids), chunk_size):
        chunk = ",".join(ids[i:i+chunk_size])
        resp = S.get(f"{BASE}/people", params={"personIds": chunk}, timeout=30).json()
        for p in resp.get("people", []):
            out[p.get("id")] = p.get("birthDate")
        time.sleep(0.1)
    return out

def compute_age_on(birthDate_str, ref_date):
    if not birthDate_str:
        return None
    try:
        b = dtparser.parse(birthDate_str).date()
    except Exception:
        return None
    years = ref_date.year - b.year
    if (ref_date.month, ref_date.day) < (b.month, b.day):
        years -= 1
    return years

def build_dodgers_postseason_batting_summaries_with_pos_age(
    seasons=SEASONS,
    team_id=TEAM_ID_DODGERS,
    out_csv="dodgers_postseason_batting_summaries_1990_2024_pos_age.csv"
):
    all_rows=[]
    for season in seasons:
        print(f"Season {season}: fetching postseason schedule…")
        pks = get_postseason_game_pks_for_team(season, team_id)
        print(f"  postseason games: {len(pks)}")
        for i, pk in enumerate(pks, 1):
            all_rows.extend(boxscore_batting_components_for_team(pk, team_id, season))
            if i % 20 == 0:
                print(f"   processed {i}/{len(pks)} games")
            time.sleep(0.08)

    df = pd.DataFrame(all_rows)

    # Ensure numeric for components
    num_cols = ["AB","H","2B","3B","HR","BB","HBP","SF"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Primary position per player-season (mode; fallback to first non-null)
    def mode_or_first(s):
        m = s.mode(dropna=True)
        if not m.empty:
            return m.iloc[0]
        s = s.dropna()
        return s.iloc[0] if not s.empty else None

    pos_ps = (
        df.groupby(["season","playerId"], as_index=False)["position"]
          .agg(primaryPosition=mode_or_first)
    )

    # Aggregate components to player-season
    agg = (df.groupby(["season","playerId","playerName"], as_index=False)
             .agg({"AB":"sum","H":"sum","2B":"sum","3B":"sum","HR":"sum",
                   "BB":"sum","HBP":"sum","SF":"sum"}))

    # Merge position
    agg = agg.merge(pos_ps, on=["season","playerId"], how="left")

    # Compute rate stats from totals
    singles = (agg["H"] - agg["2B"] - agg["3B"] - agg["HR"]).clip(lower=0)
    total_bases = singles + 2*agg["2B"] + 3*agg["3B"] + 4*agg["HR"]
    obp_den = agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]

    def safe_div(num, den):
        den = den.where(den != 0, pd.NA)
        return (num / den).fillna(0)

    agg["AVG"] = safe_div(agg["H"], agg["AB"])
    agg["OBP"] = safe_div(agg["H"] + agg["BB"] + agg["HBP"], obp_den)
    agg["SLG"] = safe_div(total_bases, agg["AB"])

    # Ages (as of Oct 1 each season)
    unique_players = agg["playerId"].unique().tolist()
    birth_map = fetch_birthdates(unique_players)
    ages = []
    for _, row in agg.iterrows():
        bdate = birth_map.get(int(row["playerId"])) if pd.notna(row["playerId"]) else None
        ref = date(int(row["season"]), 10, 1)  # Oct 1
        ages.append(compute_age_on(bdate, ref))
    agg["Age"] = ages

    # ---- Final output (omit playerId) ----
    out = agg[["season","playerName","primaryPosition","Age","AB","AVG","OBP","SLG"]].copy()
    out = out.sort_values(["season","playerName"]).reset_index(drop=True)

    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (rows: {len(out)})")
    return out

# Run it
summaries = build_dodgers_postseason_batting_summaries_with_pos_age()


