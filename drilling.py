# Groundwater Commons Game (Streamlit) — Simplified Core Version (4–5 players)
# ------------------------------------------------------------------------------
# Changes in this edit:
#   • Stage B: "Join as Player" is the default tab (Host is second).
#   • Auto-end after T=12 rounds (both Solo + Multiplayer).
#   • Host can create rooms with up to 6 players (per-room capacity; fewer can still play).
#   • Default c1 remains 0.006 (as in your version).
#
# Save as streamlit_app.py (or your filename) and run: streamlit run streamlit_app.py

from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import json
import uuid
import random
import string
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ------------------------------
# Page Config / Styling
# ------------------------------
st.set_page_config(page_title="Groundwater Commons Game (Simplified)", layout="wide")

# Minimal CSS polish
st.markdown(
    """
    <style>
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
      .muted {color:#9ca3af}
      .divider {height:1px; background:#1f2937; margin:10px 0 14px}
      .gauge-label {margin-top:8px; color:#9ca3af; font-size:13px}
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "commons_game.db"
SOLO_WELLS = 4              # Solo: player owns 4 wells
DEFAULT_ROOM_CAPACITY = 6   # Stage B default; host can pick 4 or 5 per room

# ------------------------------
# SQLite helpers (multiplayer coordination)
# ------------------------------

def get_conn():
    """Return a new sqlite3 connection (fresh per call keeps things simple in Streamlit)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rooms (
              code TEXT PRIMARY KEY,
              created_at TEXT,
              params_json TEXT,
              state_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
              player_id TEXT PRIMARY KEY,
              room_code TEXT,
              name TEXT,
              well_index INTEGER,
              cumulative_profit REAL,
              joined_at TEXT,
              last_active TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rounds (
              room_code TEXT,
              round_index INTEGER,
              S REAL,
              recharge REAL,
              group_q REAL,
              event TEXT,
              PRIMARY KEY (room_code, round_index)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
              room_code TEXT,
              round_index INTEGER,
              player_id TEXT,
              q REAL,
              submitted INTEGER,
              profit REAL,
              timestamp TEXT,
              PRIMARY KEY (room_code, round_index, player_id)
            );
            """
        )
        conn.commit()


init_db()

# ------------------------------
# Core economics
# ------------------------------

def profit_one_period(q: float, S: float, params: Dict) -> float:
    """Private profit for one player in one period.
    π(q, S) = P·q − ½γ q² − c0·q − c1·(Smax − S)·q
    """
    P = params["P"]
    gamma = params["gamma"]
    c0 = params["c0"]
    c1 = params["c1"]
    Smax = params["Smax"]
    revenue = P * q - 0.5 * gamma * q * q
    cost = c0 * q + c1 * (Smax - S) * q
    return revenue - cost


def next_stock(S: float, R: float, q_total: float) -> float:
    """Stock transition with non-negativity."""
    return max(0.0, S + R - q_total)

# ------------------------------
# Multiplayer (Stage B) helpers
# ------------------------------

def room_exists(code: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM rooms WHERE code=?", (code,))
        return cur.fetchone() is not None


def get_room_capacity(code: str) -> int:
    """Read per-room capacity (4 or 5) from room state."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT state_json FROM rooms WHERE code=?", (code,))
        row = cur.fetchone()
        if not row:
            return DEFAULT_ROOM_CAPACITY
        state = json.loads(row[0])
        return int(state.get("players_expected", DEFAULT_ROOM_CAPACITY))


def create_room(params: Dict) -> str:
    """Create a new room with initial round 0 row."""
    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    now = datetime.utcnow().isoformat()
    capacity = int(params.get("players_expected", DEFAULT_ROOM_CAPACITY))
    state = {
        "status": "lobby",  # lobby -> running -> finished
        "current_round": 0,
        "players_expected": capacity,
    }
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO rooms(code, created_at, params_json, state_json) VALUES(?,?,?,?)",
            (code, now, json.dumps(params), json.dumps(state)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO rounds(room_code, round_index, S, recharge, group_q, event) VALUES(?,?,?,?,?,?)",
            (code, 0, params["S0"], params["R"], 0.0, json.dumps({})),
        )
        conn.commit()
    return code


def load_room(code: str) -> Tuple[Dict, Dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT params_json, state_json FROM rooms WHERE code=?", (code,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Room not found")
        params = json.loads(row[0])
        state = json.loads(row[1])
        return params, state


def save_room_state(code: str, state: Dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE rooms SET state_json=? WHERE code=?", (json.dumps(state), code))
        conn.commit()


def update_room_params(code: str, params: Dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE rooms SET params_json=? WHERE code=?", (json.dumps(params), code))
        conn.commit()


def list_players(code: str) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT player_id, name, well_index, cumulative_profit, joined_at, last_active, COALESCE(is_host,0) FROM players WHERE room_code=? ORDER BY well_index ASC",
            (code,),
        )
        return [
            {
                "player_id": r[0],
                "name": r[1],
                "well_index": r[2],
                "cumulative_profit": r[3],
                "joined_at": r[4],
                "last_active": r[5],
                "is_host": int(r[6]),
            }
            for r in cur.fetchall()
        ]


def add_or_get_player(code: str, name: str, make_host: bool=False) -> Dict:
    """Join a room; if name already exists, reuse it; else allocate next well index within capacity."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT player_id, well_index, cumulative_profit FROM players WHERE room_code=? AND name=?",
            (code, name),
        )
        row = cur.fetchone()
        if row:
            pid, win, cum = row[0], row[1], row[2]
        else:
            capacity = get_room_capacity(code)
            cur.execute("SELECT well_index FROM players WHERE room_code=?", (code,))
            taken = {r[0] for r in cur.fetchall()}
            available = [i for i in range(capacity) if i not in taken]
            if not available:
                raise ValueError("Room is full")
            win = min(available)
            pid = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            cur.execute(
                "INSERT INTO players(player_id, room_code, name, well_index, cumulative_profit, joined_at, last_active, is_host) VALUES(?,?,?,?,?,?,?,?)",
                (pid, code, name, win, 0.0, now, now, 1 if (make_host and not is_room_host_assigned(code)) else 0),
            )
            conn.commit()
            cum = 0.0
        return {"player_id": pid, "well_index": win, "cumulative_profit": cum}


def touch_player(player_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE players SET last_active=? WHERE player_id=?", (datetime.utcnow().isoformat(), player_id))
        conn.commit()


def get_round_row(code: str, round_index: int) -> Dict:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT S, recharge, group_q, event FROM rounds WHERE room_code=? AND round_index=?",
            (code, round_index),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Round not found")
        return {"S": row[0], "recharge": row[1], "group_q": row[2], "event": json.loads(row[3]) if row[3] else {}}


def upsert_action(code: str, round_index: int, player_id: str, q: float, submitted: bool = False, profit_val: Optional[float] = None):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO actions(room_code, round_index, player_id, q, submitted, profit, timestamp) VALUES(?,?,?,?,?,?,?)\n             ON CONFLICT(room_code, round_index, player_id) DO UPDATE SET q=excluded.q, submitted=excluded.submitted, profit=excluded.profit, timestamp=excluded.timestamp",
            (code, round_index, player_id, float(q), int(submitted), profit_val if profit_val is not None else None, now),
        )
        conn.commit()


def fetch_actions(code: str, round_index: int) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT player_id, q, submitted, profit, timestamp FROM actions WHERE room_code=? AND round_index=?",
            (code, round_index),
        )
        return [{"player_id": r[0], "q": r[1], "submitted": bool(r[2]), "profit": r[3], "timestamp": r[4]} for r in cur.fetchall()]


def write_round_result(code: str, round_index: int, S: float, R: float, group_q: float, event: Dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO rounds(room_code, round_index, S, recharge, group_q, event) VALUES(?,?,?,?,?,?)",
            (code, round_index, S, R, group_q, json.dumps(event)),
        )
        conn.commit()


def add_to_player_cumprofit(player_id: str, delta: float):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE players SET cumulative_profit = COALESCE(cumulative_profit,0) + ? WHERE player_id=?", (float(delta), player_id))
        conn.commit()

# ------------------------------
# Multiplayer engine: advance when all have submitted
# ------------------------------

def maybe_advance_round(code: str):
    """Advance one round when all currently joined players have submitted.
    Steps: compute profits at current S, update cumulative scores, update stock, then bump round or finish.
    """
    params, state = load_room(code)
    if state["status"] != "running":
        return

    t = state["current_round"]
    rr = get_round_row(code, t)
    S_t = rr["S"]
    R_t = params["R"]

    players = list_players(code)
    capacity = get_room_capacity(code)
    n_players = min(len(players), capacity)
    if n_players == 0:
        return

    acts = fetch_actions(code, t)
    joined_ids = {p["player_id"] for p in players[:capacity]}
    submitted = [a for a in acts if a["player_id"] in joined_ids and a["submitted"]]
    if len(submitted) < n_players:
        return  # wait for all

    # Aggregate q and compute profits
    q_map = {a["player_id"]: float(a["q"]) for a in submitted}
    group_q = sum(q_map.values())

    # Per-player profits at S_t
    per_player_profit = {}
    for p in players[:capacity]:
        pid = p["player_id"]
        q_i = q_map.get(pid, 0.0)
        pi_i = profit_one_period(q_i, S_t, params)
        per_player_profit[pid] = pi_i

    # Update stock
    S_next = next_stock(S_t, R_t, group_q)

    # Persist realized profits and update cumulative
    with get_conn() as conn:
        cur = conn.cursor()
        for a in submitted:
            pid = a["player_id"]
            pi_i = per_player_profit.get(pid, 0.0)
            cur.execute(
                "UPDATE actions SET profit=? WHERE room_code=? AND round_index=? AND player_id=?",
                (pi_i, code, t, pid),
            )
            cur.execute(
                "UPDATE players SET cumulative_profit = COALESCE(cumulative_profit,0) + ? WHERE player_id=?",
                (pi_i, pid),
            )
        conn.commit()

    # Write next round stock row and advance/finish
    write_round_result(code, t + 1, S_next, R_t, group_q, {})

    T = params.get("T", 12)
    if S_next <= 0.0 or (t + 1) >= T:
        state["status"] = "finished"
    else:
        state["current_round"] = t + 1
    save_room_state(code, state)

# ------------------------------
# UI helpers
# ------------------------------


def get_player(player_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT player_id, room_code, name, well_index, cumulative_profit, joined_at, last_active, COALESCE(is_host,0) FROM players WHERE player_id=?", (player_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"player_id": row[0], "room_code": row[1], "name": row[2], "well_index": row[3], "cumulative_profit": row[4], "joined_at": row[5], "last_active": row[6], "is_host": int(row[7])}

def is_room_host_assigned(code: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM players WHERE room_code=? AND COALESCE(is_host,0)=1 LIMIT 1", (code,))
        return cur.fetchone() is not None

def set_player_host(code: str, player_id: str) -> bool:
    """Promote a player to host if room has no host yet. Returns True on success."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM players WHERE room_code=? AND COALESCE(is_host,0)=1 LIMIT 1", (code,))
        if cur.fetchone():
            return False
        cur.execute("UPDATE players SET is_host=1 WHERE player_id=? AND room_code=?", (player_id, code))
        conn.commit()
        return cur.rowcount > 0

def nice_metric(label: str, value: str, help_text: Optional[str] = None):
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.markdown(f"<div class='mono' style='font-size:22px'>{value}</div>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def stock_gauge(S: float, Smax: float, label: str = "Aquifer stock"):
    st.markdown(f"<div class='gauge-label'>{label}</div>", unsafe_allow_html=True)
    st.progress(min(1.0, S / Smax))  # no numbers, just the bar


def room_params_form(defaults: Dict) -> Dict:
    """Host's parameter form with only essential knobs, plus capacity (4 or 5)."""
    with st.form("room_params_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input("Initial stock S0", min_value=0.0, value=float(defaults["S0"]))
            R = st.number_input("Recharge R per round", min_value=0.0, value=float(defaults["R"]))
            T = st.number_input("Number of rounds T", min_value=1, step=1, value=int(defaults["T"]))
        with col2:
            qmax = st.number_input("Max q per player", min_value=0.0, value=float(defaults["qmax"]))
            P = st.number_input("Price P", min_value=0.0, value=float(defaults["P"]))
            players_expected = st.selectbox("Players per room", [6, 5, 4], index=0)
        with col3:
            gamma = st.number_input("Diminishing returns γ", min_value=0.0, value=float(defaults["gamma"]))
            c0 = st.number_input("Base pumping cost c0", min_value=0.0, value=float(defaults["c0"]))
            c1 = st.number_input("Depth cost slope c1", min_value=0.0, step=0.001, format="%.3f", value=float(defaults["c1"]))
        submitted = st.form_submit_button("Create room")
    if submitted:
        return {
            "S0": float(S0), "Smax": float(S0), "R": float(R), "T": int(T), "qmax": float(qmax),
            "P": float(P), "gamma": float(gamma), "c0": float(c0), "c1": float(c1),
            "players_expected": int(players_expected),
        }
    return {}

# ------------------------------
# Stage A: Solo (private control of all wells)
# ------------------------------

def stage_a_solo():
    st.subheader("Stage A — Private Control (Solo)")
    st.caption("You own all four wells and fully internalize future stock. Compare your path to Stage B.")

    # Baseline defaults
    defaults = {
        "S0": 1400.0, "Smax": 1400.0, "R": 60.0, "T": 8, "qmax": 80.0,
        "P": 10.0, "gamma": 0.08, "c0": 2.0, "c1": 0.006,
    }

    with st.expander("Solo parameters (optional)", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            S0 = st.number_input("S0", min_value=0.0, value=defaults["S0"])  # also Smax
            R = st.number_input("R", min_value=0.0, value=defaults["R"])
        with s2:
            T = st.number_input("T rounds", min_value=1, step=1, value=defaults["T"])
            qmax = st.number_input("Max q (per well)", min_value=0.0, value=defaults["qmax"])
        with s3:
            P = st.number_input("P", min_value=0.0, value=defaults["P"])
            gamma = st.number_input("γ", min_value=0.0, value=defaults["gamma"])
        with s4:
            c0 = st.number_input("c0", min_value=0.0, value=defaults["c0"])
            c1 = st.number_input("c1", min_value=0.0, step=0.001, format="%.3f", value=defaults["c1"])
        if st.button("Reset Solo Game with these params"):
            st.session_state.solo_state = {"t": 0, "S": float(S0), "cum": 0.0, "history": []}
            st.session_state.solo_params = {"S0": float(S0), "Smax": float(S0), "R": float(R), "T": int(T), "qmax": float(qmax), "P": float(P), "gamma": float(gamma), "c0": float(c0), "c1": float(c1)}
            st.rerun()

    # Initialize if needed
    if "solo_params" not in st.session_state:
        st.session_state.solo_params = defaults.copy()
    if "solo_state" not in st.session_state:
        st.session_state.solo_state = {"t": 0, "S": float(st.session_state.solo_params["S0"]), "cum": 0.0, "history": []}

    params = st.session_state.solo_params
    state = st.session_state.solo_state

    # Auto-end after T rounds (disable controls and show message)
    game_finished = state["t"] >= params["T"]

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Decision first (so metrics can reflect current choice) ---
    q = st.slider("Choose pumping rate per well (you own 4 wells)", 0.0, float(params["qmax"]), 40.0, 1.0, disabled=game_finished)
    group_q_preview = q * SOLO_WELLS
    pi_per_well = profit_one_period(q, state["S"], params)
    pi_total = pi_per_well * SOLO_WELLS

    # --- Metrics with PROFIT front and center ---
    cols = st.columns(3)
    with cols[0]:
        nice_metric("Round", str(state["t"]))
    with cols[1]:
        nice_metric("This round profit (undiscounted)", f"{pi_total:.1f}")
    with cols[2]:
        nice_metric("Cumulative profit", f"{state['cum']:.1f}")

    # --- Aquifer gauge (no numbers) ---
    stock_gauge(state["S"], params["Smax"], label="Aquifer stock")

    if game_finished:
        st.success(f"Game finished automatically after {params['T']} rounds.")
        return

    # --- Actions ---
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Finalize Pumping Decision (Solo)", disabled=game_finished):
            t = state["t"]
            S_t = state["S"]
            group_q = group_q_preview
            S_next = next_stock(S_t, params["R"], group_q)
            st.session_state.solo_state["cum"] += pi_total
            st.session_state.solo_state["history"].append({"t": t, "S_t": S_t, "q_per_well": q, "group_q": group_q, "profit": pi_total, "S_next": S_next})
            st.session_state.solo_state["S"] = S_next
            st.session_state.solo_state["t"] = t + 1
            st.rerun()
    with c2:
        if st.button("Reset Solo Game"):
            st.session_state.pop("solo_state", None)
            st.session_state.pop("solo_params", None)
            st.rerun()
    with c3:
        st.download_button("Download Solo History (JSON)", data=json.dumps(state.get("history", []), indent=2), file_name="solo_history.json", mime="application/json")

    if state.get("history"):
        st.markdown("### Solo Round History")
        st.dataframe(state["history"], use_container_width=True)

# ------------------------------
# Stage B: Multiplayer (common pool)
# ------------------------------


def stage_b_multiplayer():
    st.subheader("Stage B — Common Pool (Multiplayer)")
    st.caption("Up to 6 students share one aquifer. Each controls one well. Highest cumulative profit wins.")

    # Defaults (kept from original behavior where possible)
    defaults = {"S0": 1400.0, "Smax": 1400.0, "R": 60.0, "T": 8, "qmax": 80.0, "P": 10.0, "gamma": 0.08, "c0": 2.0, "c1": 0.006}

    tab_player, tab_host = st.tabs(["Join as Player", "Host a Room"])

    # ---- Player tab ----
    with tab_player:
        st.markdown("#### Join a Room")
        with st.form("join_form"):
            join_code = st.text_input("Enter room code")
            player_name = st.text_input("Your name (display)")
            requested = st.form_submit_button("Join")
        if requested:
            if not room_exists(join_code):
                st.error("Room not found. Check the code with your instructor.")
            else:
                try:
                    player = add_or_get_player(join_code, (player_name.strip() or f"Player-{random.randint(100,999)}"))
                    st.session_state["room_code"] = join_code
                    st.session_state["player_id"] = player["player_id"]
                    st.session_state["player_name"] = player_name
                    st.session_state["well_index"] = player["well_index"]
                    st.success(f"Joined room {join_code}. You are Well #{player['well_index']+1}.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        # Player console
        if st.session_state.get("room_code") and st.session_state.get("player_id"):
            code = st.session_state["room_code"]
            pid = st.session_state["player_id"]
            try:
                params, state = load_room(code)
            except Exception:
                st.error("Room was removed or expired. Rejoin.")
                return

            touch_player(pid)

            st.markdown(f"**Room:** `{code}`  |  **You:** `{st.session_state.get('player_name','(unnamed)')}`  |  **Well:** `{st.session_state.get('well_index',0)+1}`")

            rr = get_round_row(code, state["current_round"])  # S at start of this round

            if state["status"] == "lobby":
                st.info("Waiting for host to start the game.")
            # Host mini-panel: show host controls inline if this player is host
            me = get_player(pid)
            am_host = bool(me and int(me.get("is_host", 0)) == 1)

            if state["status"] == "lobby":
                if am_host:
                    st.success("You are the HOST for this room. You can start the game when ready.")
                    if st.button("Start Game (host)", key="player_host_start"):
                        state["status"] = "running"
                        save_room_state(code, state)
                        st.rerun()
                else:
                    if not is_room_host_assigned(code) and st.button("Claim Host Role for this Room", key="player_claim_host"):
                        if set_player_host(code, pid):
                            st.success("You are now the host for this room.")
                        else:
                            st.warning("Host already assigned.")
                        st.rerun()

                if st.button("Refresh"):
                    st.rerun()
                return

            if state["status"] == "finished":
                st.success("Game finished. See leaderboard on the Host tab.")
                return


            if am_host and state["status"] == "running":
                st.markdown("##### Host Controls (inline)")
                cA, cB = st.columns(2)
                with cA:
                    if st.button("Force Advance (host override)", key="player_host_force_adv"):
                        acts = fetch_actions(code, state["current_round"])
                        submitted_pids = {a["player_id"] for a in acts if a["submitted"]}
                        joined = list_players(code)
                        for pz in joined:
                            if pz["player_id"] not in submitted_pids:
                                upsert_action(code, state["current_round"], pz["player_id"], 0.0, submitted=True, profit_val=0.0)
                        maybe_advance_round(code)
                        st.rerun()
                with cB:
                    if st.button("End Game Now (host)", key="player_host_end"):
                        state["status"] = "finished"
                        save_room_state(code, state)
                        st.rerun()
            # --- Decision ---
            acts = fetch_actions(code, state["current_round"])  # my row may or may not exist
            my_act = next((a for a in acts if a["player_id"] == pid), None)
            already_submitted = bool(my_act and my_act["submitted"])

            # Light client auto-refresh while waiting (faster auto-advance)
            capacity_now = get_room_capacity(code)
            joined_players = list_players(code)
            n_players_now = min(len(joined_players), capacity_now)
            acts_now = fetch_actions(code, state["current_round"])  # current round actions
            joined_ids_now = {p["player_id"] for p in joined_players[:capacity_now]}
            submitted_count = sum(1 for a in acts_now if a["submitted"] and a["player_id"] in joined_ids_now)
            if state["status"] == "running" and submitted_count < n_players_now:
                components.html("<script>setTimeout(function(){window.parent.location.reload();}, 1200);</script>", height=0)

            qmax = float(params.get("qmax", 80.0))
            q_val = my_act["q"] if my_act else min(40.0, qmax)
            q = st.slider("Choose your pumping rate q (acre-ft)", 0.0, qmax, float(q_val), 1.0, disabled=already_submitted)

            pi_preview = profit_one_period(q, rr["S"], params)

            cols = st.columns(4)
            with cols[0]:
                nice_metric("Status", state["status"].upper())
            with cols[1]:
                nice_metric("Round", str(state["current_round"]))
            with cols[2]:
                nice_metric("This round profit (undiscounted)", f"{pi_preview:.1f}")
            with cols[3]:
                nice_metric("Max q", f"{params.get('qmax',80.0):.0f}")

            stock_gauge(rr["S"], params["Smax"], label="Aquifer stock")

            colb1, colb2, colb3 = st.columns([1,1,1])
            with colb1:
                if st.button("Finalize Pumping Decision", disabled=already_submitted):
                    upsert_action(code, state["current_round"], pid, q, submitted=True)
                    maybe_advance_round(code)  # advance if all submitted
                    st.rerun()
            with colb2:
                if st.button("Update (not final)", disabled=already_submitted):
                    upsert_action(code, state["current_round"], pid, q, submitted=False)
                    st.success("Saved draft. Click Finalize when ready.")
            with colb3:
                if st.button("Refresh status"):
                    maybe_advance_round(code)
                    st.rerun()

            st.markdown("---")
            st.markdown("#### Round Info")
            last_t = max(0, state["current_round"] - 1)
            rr_last = get_round_row(code, last_t)
            st.write({"last_resolved_round": last_t, "group_q": rr_last["group_q"], "S": rr_last["S"]})

            my_row = next((p for p in list_players(code) if p["player_id"] == pid), None)
            if my_row:
                st.success(f"Your cumulative profit: {my_row['cumulative_profit']:.1f}")

    # ---- Host tab ----
    with tab_host:
        st.markdown("#### Host Controls")

        # --- Collapsed parameter inputs (NO form) ---
        # Store values directly in session_state so the Create Room button can live outside
        host_defaults = {
            "host_S0": 1400.0, "host_R": 60.0, "host_T": 8, "host_qmax": 80.0,
            "host_P": 10.0, "host_gamma": 0.08, "host_c0": 2.0, "host_c1": 0.006, "host_capacity": 6
        }
        for k, v in host_defaults.items():
            st.session_state.setdefault(k, v)

        with st.expander("Room setup (optional)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Initial stock S0", min_value=0.0, key="host_S0")
                st.number_input("Recharge R per round", min_value=0.0, key="host_R")
                st.number_input("Number of rounds T", min_value=1, step=1, key="host_T")
            with c2:
                st.number_input("Max q per player", min_value=0.0, key="host_qmax")
                st.number_input("Price P", min_value=0.0, key="host_P")
                st.selectbox("Players per room", [6, 5, 4], index=[6,5,4].index(st.session_state["host_capacity"]), key="host_capacity")
            with c3:
                st.number_input("Diminishing returns γ", min_value=0.0, key="host_gamma")
                st.number_input("Base pumping cost c0", min_value=0.0, key="host_c0")
                st.number_input("Depth cost slope c1", min_value=0.0, step=0.001, format="%.3f", key="host_c1")

        # --- Create Room button OUTSIDE the expander ---
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Create Room", type="primary", key="host_create_room_btn"):
                new_params = {
                    "S0": float(st.session_state["host_S0"]),
                    "Smax": float(st.session_state["host_S0"]),
                    "R": float(st.session_state["host_R"]),
                    "T": int(st.session_state["host_T"]),
                    "qmax": float(st.session_state["host_qmax"]),
                    "P": float(st.session_state["host_P"]),
                    "gamma": float(st.session_state["host_gamma"]),
                    "c0": float(st.session_state["host_c0"]),
                    "c1": float(st.session_state["host_c1"]),
                    "players_expected": int(st.session_state["host_capacity"]),
                }
                code = create_room(new_params)
                st.session_state["host_room_code"] = code
                st.session_state["just_created_code"] = code
                st.success(f"Room created. Code: **{code}**")
                st.info("Share this code. Use the join box below if you'd like to play in this room.")

        with colB:
            # Optional immediate join after creation: works on any existing code in the input below
            pass

        # --- Host join fix: allow host to join and appear in lobby ---
        code_existing = st.text_input("Manage / Join room code", value=st.session_state.get("host_room_code", st.session_state.get("just_created_code", "")), key="host_manage_code")
        join_name = st.text_input("Join this room as (display name)", value="Host", key="host_join_name")
        if st.button("Join as Player in this Room", key="host_join_btn"):
            if not code_existing or not room_exists(code_existing):
                st.error("Room not found. Create or enter a valid code.")
            else:
                try:
                    player = add_or_get_player(code_existing, (join_name.strip() or "Host"), make_host=True)
                    # Persist session so the Host also sees the player console on the Join tab
                    st.session_state["room_code"] = code_existing
                    st.session_state["player_id"] = player["player_id"]
                    st.session_state["player_name"] = join_name.strip() or "Host"
                    st.session_state["well_index"] = player["well_index"]
                    st.success(f"Joined room {code_existing} as {join_name.strip() or 'Host'} (Well #{player['well_index']+1}).")
                except Exception as e:
                    st.error(f"Could not join: {e}")

        # --- Room dashboard for host ---
        if code_existing and room_exists(code_existing):
            params, state = load_room(code_existing)
            players = list_players(code_existing)
            capacity = get_room_capacity(code_existing)

            cols = st.columns(4)
            with cols[0]:
                nice_metric("Status", state["status"].upper())
            with cols[1]:
                nice_metric("Current round", str(state["current_round"]))
            with cols[2]:
                nice_metric("Players joined", f"{len(players)}/{capacity}")
            with cols[3]:
                rr = get_round_row(code_existing, state["current_round"])
                nice_metric("Aquifer S (host)", f"{rr['S']:.1f}")

            st.progress(min(1.0, rr["S"] / params["Smax"]))

            if state["status"] == "lobby":
                st.write("**Players in lobby**:")
                st.table([{k: p[k] for k in ("name", "well_index", "cumulative_profit")} for p in players])
                if len(players) > 0 and st.button("Start Game", key="host_start_btn"):
                    state["status"] = "running"
                    save_room_state(code_existing, state)
                    st.rerun()

            elif state["status"] == "running":
                st.write("**Game is running.** It advances automatically once all players submit.")
                cA, cB = st.columns(2)
                with cA:
                    if st.button("Force Advance (host override)", key="host_force_adv"):
                        acts = fetch_actions(code_existing, state["current_round"])
                        submitted_pids = {a["player_id"] for a in acts if a["submitted"]}
                        joined = list_players(code_existing)
                        for p in joined:
                            if p["player_id"] not in submitted_pids:
                                upsert_action(code_existing, state["current_round"], p["player_id"], 0.0, submitted=True, profit_val=0.0)
                        maybe_advance_round(code_existing)
                        st.rerun()
                with cB:
                    if st.button("End Game Now", key="host_end_now"):
                        state["status"] = "finished"
                        save_room_state(code_existing, state)
                        st.rerun()

            elif state["status"] == "finished":
                st.success("Game finished.")

            st.markdown("### Leaderboard (cumulative profits)")
            players_sorted = sorted(players, key=lambda p: p["cumulative_profit"], reverse=True)
            st.table([{ "Rank": i+1, "Player": p["name"], "Score": round(p["cumulative_profit"], 1)} for i, p in enumerate(players_sorted)])

            last_t = max(0, state["current_round"] - (1 if state["status"] != "lobby" else 0))
            rr_last = get_round_row(code_existing, last_t)
            st.markdown("#### Last resolved round summary")
            st.write({"round": last_t, "S": rr_last["S"], "group_q": rr_last["group_q"]})

            if st.button("Refresh status", key="host_refresh_status"):
                maybe_advance_round(code_existing)
                st.rerun()
        else:
            if code_existing:
                st.warning("Room not found. Create a new room above.")

def main():
    st.title("Groundwater Commons Game")
    st.caption("You are a farmer pumping groundwater to irrigate your crops. Pumping water lowers the water level, which makes pumping slightly more expensive. Meanwhile, the aquifer naturally replenishes at some rate.")

    with st.sidebar:
        st.markdown("### Model in one glance")
        st.markdown(
            "- Profit each round: **P·q − ½γq² − c₀q − c₁(Smax−S)q**."
            "\n- Stock update: **Sₜ₊₁ = Sₜ + R − Σqᵢ**."
            "\n- Score: simple cumulative profit."
        )
        st.markdown("---")
        st.markdown("Created by Jenna Anders, University of Virginia")
        st.markdown("---")
        st.markdown("Instructions: Run Stage A first, then a Stage B room. Keep parameters fixed to compare behavior.")
    
    tab_a, tab_b = st.tabs(["Stage A (Solo)", "Stage B (Multiplayer)"])
    with tab_a:
        stage_a_solo()
    with tab_b:
        stage_b_multiplayer()


if __name__ == "__main__":
    main()
