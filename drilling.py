# Groundwater Commons Game (Streamlit)
# -----------------------------------
# A teaching game to illustrate the tragedy of the commons for groundwater pumping.
#
# Features
# - Stage A (Solo): One player controls all four wells on their own aquifer.
# - Stage B (Multiplayer): Up to 4 players share one aquifer (join via a room code).
# - Clear private payoffs with diminishing returns and depth-dependent costs.
# - Dynamic aquifer stock with recharge, thresholds (optional), discounting, and policy toggles (tax/cap).
# - SQLite backing store for multiplayer rooms so multiple browser sessions can join the same game.
#
# How to run
#   1) pip install streamlit
#   2) Save this file as streamlit_app.py
#   3) streamlit run streamlit_app.py
#
# Notes on multiplayer
# - The app uses a local SQLite database (commons_game.db) to coordinate across sessions.
# - One host creates a room and shares the code. Up to 4 students join that room.
# - A round advances automatically once all joined players finalize their decisions.
# - Data are ephemeral and scoped to this server instance. Restarting the app resets rooms.
#
# Pedagogical choices embedded
# - Players see immediate private profit, cumulative discounted profit, aquifer level, and group extraction totals.
# - Players do NOT see others' individual pumping choices for the current round (information externality).
# - Optional nonlinear threshold (salinity/subsidence) that permanently reduces price when stock crosses a level.
# - Policy toggles (per-unit tax; group cap with ex-post pro-rata scaling when binding).

from __future__ import annotations
import streamlit as st
import sqlite3
import json
import uuid
import random
import string
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ------------------------------
# Page Config / Styling
# ------------------------------
st.set_page_config(page_title="Groundwater Commons Game", layout="wide")

# Minimal CSS polish
st.markdown(
    """
    <style>
      .metric-card {padding: 12px 14px; border-radius: 10px; background: #111827; border: 1px solid #1f2937;}
      .soft {color:#94a3b8}
      .ok {color:#10b981}
      .warn {color:#f59e0b}
      .bad {color:#ef4444}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
      .muted {color:#9ca3af}
      .divider {height:1px; background:#1f2937; margin:10px 0 14px}
      .tiny {font-size: 12px}
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "commons_game.db"
N_WELLS = 4  # fixed in this version; each player controls one well in Stage B

# ------------------------------
# Utility: SQLite helpers
# ------------------------------

def get_conn():
    """Return a new sqlite3 connection. Using a fresh connection per call keeps things simple under Streamlit."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
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
# Game economics (shared)
# ------------------------------

def profit_one_period(q: float, S: float, params: Dict, round_index: int) -> float:
    """Compute private profit for one player in one period given their q and current stock S.
    Profit = P_eff*q - 0.5*gamma*q^2 - c0*q - c1*(Smax - S)*q - tau*q
    where P_eff can be reduced by a threshold event that might have happened earlier.
    Discounting is applied outside this function when accumulating scores.
    """
    P = params["P"] * params.get("price_factor", 1.0)  # factor may be <1 if threshold triggered previously
    gamma = params["gamma"]
    c0 = params["c0"]
    c1 = params["c1"]
    Smax = params["Smax"]
    tau = params.get("tax", 0.0)
    # Private profit this period
    rev = P * q - 0.5 * gamma * q * q
    cost = c0 * q + c1 * (Smax - S) * q + tau * q
    return rev - cost


def next_stock(S: float, R: float, q_total: float) -> float:
    """Stock transition. Enforce non-negativity."""
    return max(0.0, S + R - q_total)


def discounted(x: float, r: float, t: int) -> float:
    """Discount x at rate r with period t (starting from t=0)."""
    return x / ((1.0 + r) ** t)


# ------------------------------
# Multiplayer (Stage B) data accessors
# ------------------------------

def room_exists(code: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM rooms WHERE code=?", (code,))
        return cur.fetchone() is not None


def create_room(params: Dict) -> str:
    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    now = datetime.utcnow().isoformat()
    state = {
        "status": "lobby",  # lobby -> running -> finished
        "current_round": 0,
        "players_expected": N_WELLS,
        "threshold_triggered": False,
        "threshold_type": None,  # "price_drop" or None
    }
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO rooms(code, created_at, params_json, state_json) VALUES(?,?,?,?)",
            (code, now, json.dumps(params), json.dumps(state)),
        )
        # Insert round 0 row (pre-play state)
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
            "SELECT player_id, name, well_index, cumulative_profit, joined_at, last_active FROM players WHERE room_code=? ORDER BY well_index ASC",
            (code,),
        )
        out = []
        for r in cur.fetchall():
            out.append(
                {
                    "player_id": r[0],
                    "name": r[1],
                    "well_index": r[2],
                    "cumulative_profit": r[3],
                    "joined_at": r[4],
                    "last_active": r[5],
                }
            )
        return out


def add_or_get_player(code: str, name: str) -> Dict:
    """Join a room; if name already exists, return that player; otherwise allocate next well index."""
    with get_conn() as conn:
        cur = conn.cursor()
        # Return existing by name, if present
        cur.execute(
            "SELECT player_id, well_index, cumulative_profit FROM players WHERE room_code=? AND name=?",
            (code, name),
        )
        row = cur.fetchone()
        if row:
            pid = row[0]
            win = row[1]
            cum = row[2]
        else:
            # Determine next available well index 0..3
            cur.execute(
                "SELECT well_index FROM players WHERE room_code=?",
                (code,),
            )
            taken = {r[0] for r in cur.fetchall()}
            available = [i for i in range(N_WELLS) if i not in taken]
            if not available:
                raise ValueError("Room is full")
            win = min(available)
            pid = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            cur.execute(
                "INSERT INTO players(player_id, room_code, name, well_index, cumulative_profit, joined_at, last_active) VALUES(?,?,?,?,?,?,?)",
                (pid, code, name, win, 0.0, now, now),
            )
            conn.commit()
            cum = 0.0
        return {"player_id": pid, "well_index": win, "cumulative_profit": cum}


def touch_player(player_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE players SET last_active=? WHERE player_id=?",
            (datetime.utcnow().isoformat(), player_id),
        )
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
        out = []
        for r in cur.fetchall():
            out.append({"player_id": r[0], "q": r[1], "submitted": bool(r[2]), "profit": r[3], "timestamp": r[4]})
        return out


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
        cur.execute("UPDATE players SET cumulative_profit = COALESCE(cumulative_profit,0)+? WHERE player_id=?", (float(delta), player_id))
        conn.commit()


# ------------------------------
# Multiplayer engine tick
# ------------------------------

def maybe_advance_round(code: str):
    """Advance the game one step if all joined players have submitted the current round.
    Handles policy (tax/cap) and threshold logic, updates stock, stores profits (discounted).
    """
    params, state = load_room(code)
    if state["status"] != "running":
        return  # only process when running

    t = state["current_round"]
    # Get current stock and recharge
    round_row = get_round_row(code, t)
    S_t = round_row["S"]
    R_t = params["R"]

    # Joined players (limit to N_WELLS)
    players = list_players(code)
    n_players = min(len(players), N_WELLS)
    if n_players == 0:
        return

    acts = fetch_actions(code, t)
    # Count submissions among joined players only
    joined_ids = {p["player_id"] for p in players[:N_WELLS]}
    submitted = [a for a in acts if a["player_id"] in joined_ids and a["submitted"]]
    if len(submitted) < n_players:
        return  # wait until all current players have submitted

    # Aggregate intended q_i
    q_map = {a["player_id"]: float(a["q"]) for a in submitted}
    sum_q = sum(q_map.values())

    # Apply CAP policy ex-post with proportional scaling if binding
    cap_val = params.get("cap") if params.get("cap_enabled", False) else None
    cap_binding = False
    scale = 1.0
    if cap_val is not None and sum_q > cap_val:
        cap_binding = True
        scale = cap_val / sum_q if sum_q > 0 else 1.0

    q_scaled = {pid: q * scale for pid, q in q_map.items()}
    group_q = sum(q_scaled.values())

    # Compute profits at current S_t (policy tax embedded in profit function)
    r = params.get("discount_rate", 0.0)
    per_player_profit = {}
    for p in players[:N_WELLS]:
        pid = p["player_id"]
        q_i = q_scaled.get(pid, 0.0)
        pi_i = profit_one_period(q_i, S_t, params, t)
        dpi = discounted(pi_i, r, t)
        per_player_profit[pid] = (pi_i, dpi)

    # Update S_{t+1}
    S_next = next_stock(S_t, R_t, group_q)

    # Threshold logic (e.g., price drop for future rounds)
    event = {}
    if (not state.get("threshold_triggered", False)) and params.get("threshold_enabled", False):
        thresh = params.get("threshold_level", 0.0)
        if S_next <= thresh:
            state["threshold_triggered"] = True
            state["threshold_type"] = "price_drop"
            params["price_factor"] = params.get("threshold_price_factor", 0.7)
            update_room_params(code, params)
            event = {"type": "threshold", "detail": f"Price factor set to {params['price_factor']:.2f} due to low aquifer"}

    # Write end-of-round row for t+1 (the new S)
    write_round_result(code, t + 1, S_next, R_t, group_q, event)

    # Persist player profits and mark current-round actions with realized profit
    with get_conn() as conn:
        cur = conn.cursor()
        for a in submitted:
            pid = a["player_id"]
            pi_i, dpi = per_player_profit.get(pid, (0.0, 0.0))
            cur.execute(
                "UPDATE actions SET profit=? WHERE room_code=? AND round_index=? AND player_id=?",
                (pi_i, code, t, pid),
            )
            # Add discounted profit to cumulative score
            cur.execute(
                "UPDATE players SET cumulative_profit = COALESCE(cumulative_profit,0) + ? WHERE player_id=?",
                (dpi, pid),
            )
        conn.commit()

    # Advance round or finish
    T = params.get("T", 12)
    if S_next <= 0.0 or (t + 1) >= T:
        state["status"] = "finished"
    else:
        state["current_round"] = t + 1

    save_room_state(code, state)


# ------------------------------
# UI Helpers
# ------------------------------

def nice_metric(label: str, value: str, help_text: Optional[str] = None):
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.markdown(f"<div class='mono' style='font-size:22px'>{value}</div>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def room_params_form(defaults: Dict) -> Dict:
    with st.form("room_params_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input("Initial stock S0 (acre-ft)", min_value=0.0, value=float(defaults["S0"]))
            R = st.number_input("Recharge R per round", min_value=0.0, value=float(defaults["R"]))
            T = st.number_input("Number of rounds T", min_value=1, step=1, value=int(defaults["T"]))
            qmax = st.number_input("Max q per player", min_value=0.0, value=float(defaults["qmax"]))
        with col2:
            P = st.number_input("Price P", min_value=0.0, value=float(defaults["P"]))
            gamma = st.number_input("Diminishing returns γ", min_value=0.0, value=float(defaults["gamma"]))
            c0 = st.number_input("Base pumping cost c0", min_value=0.0, value=float(defaults["c0"]))
            c1 = st.number_input("Depth cost slope c1", min_value=0.0, value=float(defaults["c1"]))
        with col3:
            r = st.number_input("Discount rate r", min_value=0.0, value=float(defaults["discount_rate"]))
            tax = st.number_input("Per-unit tax τ (optional)", min_value=0.0, value=float(defaults.get("tax", 0.0)))
            cap_enabled = st.checkbox("Enable group extraction CAP", value=bool(defaults.get("cap_enabled", False)))
            cap = st.number_input("CAP value (sum q)", min_value=0.0, value=float(defaults.get("cap", 0.0))) if cap_enabled else 0.0
        st.markdown("---")
        colt1, colt2, colt3 = st.columns(3)
        with colt1:
            threshold_enabled = st.checkbox("Enable threshold (price drop)", value=bool(defaults.get("threshold_enabled", False)))
        with colt2:
            threshold_level = st.number_input("Threshold stock level", min_value=0.0, value=float(defaults.get("threshold_level", 250.0))) if threshold_enabled else 0.0
        with colt3:
            threshold_price_factor = st.slider("Post-threshold price factor", 0.1, 1.0, float(defaults.get("threshold_price_factor", 0.7))) if threshold_enabled else 1.0

        submitted = st.form_submit_button("Create room")
    if submitted:
        return {
            "S0": float(S0), "Smax": float(S0), "R": float(R), "T": int(T), "qmax": float(qmax),
            "P": float(P), "gamma": float(gamma), "c0": float(c0), "c1": float(c1),
            "discount_rate": float(r), "tax": float(tax),
            "cap_enabled": bool(cap_enabled), "cap": float(cap),
            "threshold_enabled": bool(threshold_enabled),
            "threshold_level": float(threshold_level),
            "threshold_price_factor": float(threshold_price_factor),
            # Runtime-evolving params
            "price_factor": 1.0,
        }
    return {}


# ------------------------------
# Stage A: Solo mode (private control of all wells)
# ------------------------------

def stage_a_solo():
    st.subheader("Stage A — Private Control (Solo)")
    st.caption("You own all four wells and fully internalize future stock. Compare your path to Stage B.")

    # Default parameters (same as paper design)
    defaults = {
        "S0": 1000.0, "Smax": 1000.0, "R": 60.0, "T": 12, "qmax": 80.0,
        "P": 10.0, "gamma": 0.08, "c0": 2.0, "c1": 0.05, "discount_rate": 0.10,
        "tax": 0.0, "price_factor": 1.0,
        "threshold_enabled": True, "threshold_level": 250.0, "threshold_price_factor": 0.7,
    }

    # Allow quick tuning in the sidebar
    with st.expander("Solo parameters (optional)", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            S0 = st.number_input("S0", min_value=0.0, value=defaults["S0"])
            R = st.number_input("R", min_value=0.0, value=defaults["R"])
        with s2:
            T = st.number_input("T rounds", min_value=1, step=1, value=defaults["T"])
            qmax = st.number_input("Max q (you control per well)", min_value=0.0, value=defaults["qmax"])
        with s3:
            P = st.number_input("P", min_value=0.0, value=defaults["P"])
            gamma = st.number_input("γ", min_value=0.0, value=defaults["gamma"])
        with s4:
            c0 = st.number_input("c0", min_value=0.0, value=defaults["c0"])
            c1 = st.number_input("c1", min_value=0.0, value=defaults["c1"])
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            r = st.number_input("discount r", min_value=0.0, value=defaults["discount_rate"])
        with tcol2:
            tax = st.number_input("tax τ", min_value=0.0, value=defaults["tax"])
        with tcol3:
            th_on = st.checkbox("Enable threshold", value=True)
        if th_on:
            t2c1, t2c2 = st.columns(2)
            with t2c1:
                th_level = st.number_input("Threshold level", min_value=0.0, value=defaults["threshold_level"])
            with t2c2:
                th_factor = st.slider("Price factor post-threshold", 0.1, 1.0, defaults["threshold_price_factor"])
        else:
            th_level, th_factor = 0.0, 1.0

    # Initialize session state for solo
    if "solo_state" not in st.session_state:
        st.session_state.solo_state = {
            "t": 0,
            "S": float(defaults["S0"]),
            "cum": 0.0,
            "price_factor": 1.0,
            "history": [],  # list of dicts per round
        }

    # Let the user override params after init
    local_params = {
        "S0": float(S0), "Smax": float(S0), "R": float(R), "T": int(T), "qmax": float(qmax),
        "P": float(P), "gamma": float(gamma), "c0": float(c0), "c1": float(c1),
        "discount_rate": float(r), "tax": float(tax), "price_factor": st.session_state.solo_state["price_factor"],
        "threshold_enabled": bool(th_on), "threshold_level": float(th_level), "threshold_price_factor": float(th_factor),
    }

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Header metrics
    cols = st.columns(4)
    with cols[0]:
        nice_metric("Round", str(st.session_state.solo_state["t"]))
    with cols[1]:
        nice_metric("Aquifer stock S", f"{st.session_state.solo_state['S']:.1f}")
    with cols[2]:
        nice_metric("Cumulative discounted profit", f"{st.session_state.solo_state['cum']:.1f}")
    with cols[3]:
        nice_metric("Price factor", f"{st.session_state.solo_state['price_factor']:.2f}", "<1 once threshold is crossed")

    # Decision: since you own all 4 wells, you pick ONE q which applies to each well
    q = st.slider("Choose pumping rate per well (you own 4 wells)", 0.0, float(local_params["qmax"]), 40.0, 1.0)
    group_q_preview = q * N_WELLS

    # Show immediate profit preview (sum across 4 wells)
    pi_per_well = profit_one_period(q, st.session_state.solo_state["S"], local_params, st.session_state.solo_state["t"])  # uses current price_factor
    pi_total = pi_per_well * N_WELLS
    st.info(f"This round profit (undiscounted): {pi_total:.1f}  |  Group extraction preview: {group_q_preview:.1f}")

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn1:
        if st.button("Finalize Pumping Decision (Solo)"):
            # Resolve round
            t = st.session_state.solo_state["t"]
            S_t = st.session_state.solo_state["S"]
            group_q = group_q_preview
            S_next = next_stock(S_t, local_params["R"], group_q)
            # Threshold for future price
            price_factor = st.session_state.solo_state["price_factor"]
            event = None
            if (price_factor >= 1.0) and local_params.get("threshold_enabled", False) and S_next <= local_params.get("threshold_level", 0.0):
                price_factor = local_params.get("threshold_price_factor", 0.7)
                event = f"Threshold hit: price factor -> {price_factor:.2f}"
            # Profit and discounting
            dpi = discounted(pi_total, local_params["discount_rate"], t)
            st.session_state.solo_state["cum"] += dpi
            st.session_state.solo_state["history"].append({
                "t": t,
                "S_t": S_t,
                "q_per_well": q,
                "group_q": group_q,
                "profit": pi_total,
                "disc_profit": dpi,
                "S_next": S_next,
                "event": event,
            })
            st.session_state.solo_state["S"] = S_next
            st.session_state.solo_state["price_factor"] = price_factor
            st.session_state.solo_state["t"] = t + 1
            st.rerun()
    with col_btn2:
        if st.button("Reset Solo Game"):
            st.session_state.pop("solo_state", None)
            st.rerun()
    with col_btn3:
        st.download_button(
            "Download Solo History (JSON)",
            data=json.dumps(st.session_state.solo_state.get("history", []), indent=2),
            file_name="solo_history.json",
            mime="application/json",
        )

    # Show table of history
    if st.session_state.solo_state.get("history"):
        st.markdown("### Solo Round History")
        st.dataframe(st.session_state.solo_state["history"], use_container_width=True)


# ------------------------------
# Stage B: Multiplayer (Common Pool)
# ------------------------------

def stage_b_multiplayer():
    st.subheader("Stage B — Common Pool (Multiplayer)")
    st.caption("Up to 4 students share one aquifer. Each controls one well. Highest discounted cumulative profit wins.")

    # Default parameters for new rooms (mirrors Solo defaults with a few extras)
    defaults = {
        "S0": 1000.0, "Smax": 1000.0, "R": 60.0, "T": 12, "qmax": 80.0,
        "P": 10.0, "gamma": 0.08, "c0": 2.0, "c1": 0.05, "discount_rate": 0.10,
        "tax": 0.0, "cap_enabled": False, "cap": 0.0,
        "threshold_enabled": True, "threshold_level": 250.0, "threshold_price_factor": 0.7,
        "price_factor": 1.0,
    }

    # Tabs: Host | Player
    tab_host, tab_player = st.tabs(["Host a Room", "Join as Player"])

    with tab_host:
        st.markdown("#### Host Controls")
        new_params = room_params_form(defaults)
        if new_params:
            code = create_room(new_params)
            st.success(f"Room created. Code: **{code}**")
            st.info("Share the code with students. The game starts when you click 'Start Game' below.")
            st.session_state["host_room_code"] = code

        code_existing = st.text_input("Or manage existing room code", value=st.session_state.get("host_room_code", ""))
        if code_existing and room_exists(code_existing):
            params, state = load_room(code_existing)
            players = list_players(code_existing)
            cols = st.columns(4)
            with cols[0]:
                nice_metric("Status", state["status"].upper())
            with cols[1]:
                nice_metric("Current round", str(state["current_round"]))
            with cols[2]:
                nice_metric("Players joined", f"{len(players)}/{N_WELLS}")
            with cols[3]:
                nice_metric("Price factor", f"{params.get('price_factor',1.0):.2f}")

            # Show current stock
            rr = get_round_row(code_existing, state["current_round"])  # S at start of this round
            st.progress(min(1.0, rr["S"] / params["Smax"]), text=f"Aquifer stock S = {rr['S']:.1f} / {params['Smax']:.0f}")

            # Lobby controls
            if state["status"] == "lobby":
                st.write("**Players in lobby**:")
                st.table([{k: p[k] for k in ("name", "well_index", "cumulative_profit")} for p in players])
                if len(players) > 0 and st.button("Start Game"):
                    state["status"] = "running"
                    save_room_state(code_existing, state)
                    st.rerun()

            elif state["status"] == "running":
                st.write("**Game is running.** It advances automatically each round once all joined players submit.")
                if st.button("Force Advance (host override)"):
                    # Force submissions for missing players as q=0
                    acts = fetch_actions(code_existing, state["current_round"])
                    submitted_pids = {a["player_id"] for a in acts if a["submitted"]}
                    joined = list_players(code_existing)
                    for p in joined:
                        if p["player_id"] not in submitted_pids:
                            upsert_action(code_existing, state["current_round"], p["player_id"], 0.0, submitted=True, profit_val=0.0)
                    maybe_advance_round(code_existing)
                    st.rerun()
                if st.button("End Game Now"):
                    state["status"] = "finished"
                    save_room_state(code_existing, state)
                    st.rerun()

            elif state["status"] == "finished":
                st.success("Game finished.")

            # Leaderboard & last round stats
            st.markdown("### Leaderboard (discounted cumulative profits)")
            players_sorted = sorted(players, key=lambda p: p["cumulative_profit"], reverse=True)
            st.table([
                {"Rank": i+1, "Player": p["name"], "Score": round(p["cumulative_profit"], 1)}
                for i, p in enumerate(players_sorted)
            ])

            last_t = max(0, state["current_round"] - (1 if state["status"] != "lobby" else 0))
            rr_last = get_round_row(code_existing, last_t)
            st.markdown("#### Last resolved round summary")
            st.write({"round": last_t, "S": rr_last["S"], "group_q": rr_last["group_q"], "event": rr_last.get("event")})

            if st.button("Refresh status"):
                maybe_advance_round(code_existing)
                st.rerun()
        else:
            if code_existing:
                st.warning("Room not found. Create a new room above.")

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
                    player = add_or_get_player(join_code, player_name.strip() or f"Player-{random.randint(100,999)}")
                    st.session_state["room_code"] = join_code
                    st.session_state["player_id"] = player["player_id"]
                    st.session_state["player_name"] = player_name
                    st.session_state["well_index"] = player["well_index"]
                    st.success(f"Joined room {join_code}. You are Well #{player['well_index']+1}.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        # If already joined, render the player console
        if st.session_state.get("room_code") and st.session_state.get("player_id"):
            code = st.session_state["room_code"]
            pid = st.session_state["player_id"]
            try:
                params, state = load_room(code)
            except Exception:
                st.error("Room was removed or expired. Rejoin.")
                return

            # Keep-alive
            touch_player(pid)

            st.markdown(f"**Room:** `{code}`  |  **You:** `{st.session_state.get('player_name','(unnamed)')}`  |  **Well:** `{st.session_state.get('well_index',0)+1}`")

            rr = get_round_row(code, state["current_round"])  # S at start of this round
            cols = st.columns(4)
            with cols[0]:
                nice_metric("Status", state["status"].upper())
            with cols[1]:
                nice_metric("Round", str(state["current_round"]))
            with cols[2]:
                nice_metric("Aquifer S", f"{rr['S']:.1f}")
            with cols[3]:
                nice_metric("Price factor", f"{params.get('price_factor',1.0):.2f}")

            st.progress(min(1.0, rr["S"] / params["Smax"]), text=f"Aquifer stock S = {rr['S']:.1f} / {params['Smax']:.0f}")

            if state["status"] == "lobby":
                st.info("Waiting for host to start the game.")
                if st.button("Refresh"):
                    st.rerun()
                return

            if state["status"] == "finished":
                st.success("Game finished. See leaderboard on the Host tab.")
                return

            # Player decision panel for current round
            # Check if already submitted
            acts = fetch_actions(code, state["current_round"])  # my row may or may not exist
            my_act = next((a for a in acts if a["player_id"] == pid), None)
            already_submitted = bool(my_act and my_act["submitted"])

            # Slider uses room qmax
            qmax = float(params.get("qmax", 80.0))
            q_val = my_act["q"] if my_act else min(40.0, qmax)
            q = st.slider("Choose your pumping rate q (acre-ft)", 0.0, qmax, float(q_val), 1.0, disabled=already_submitted)

            # Profit preview at current S
            pi_preview = profit_one_period(q, rr["S"], params, state["current_round"])  # undiscounted
            st.info(f"This round profit (undiscounted): {pi_preview:.1f}")

            # Submit decision
            colb1, colb2, colb3 = st.columns([1,1,1])
            with colb1:
                if st.button("Finalize Pumping Decision", disabled=already_submitted):
                    upsert_action(code, state["current_round"], pid, q, submitted=True)
                    maybe_advance_round(code)  # check if everyone is done
                    st.rerun()
            with colb2:
                if st.button("Update (not final)", disabled=already_submitted):
                    upsert_action(code, state["current_round"], pid, q, submitted=False)
                    st.success("Saved draft. Click Finalize when ready.")
            with colb3:
                if st.button("Refresh status"):
                    maybe_advance_round(code)
                    st.rerun()

            # After-action info
            st.markdown("---")
            st.markdown("#### Round Info")
            # Only show last resolved totals (not individual q)
            last_t = max(0, state["current_round"] - 1)
            rr_last = get_round_row(code, last_t)
            st.write({"last_resolved_round": last_t, "group_q": rr_last["group_q"], "S": rr_last["S"], "event": rr_last.get("event")})

            # Personal cumulative score
            my_row = next((p for p in list_players(code) if p["player_id"] == pid), None)
            if my_row:
                st.success(f"Your discounted cumulative profit: {my_row['cumulative_profit']:.1f}")


# ------------------------------
# Main App
# ------------------------------

def main():
    st.title("Groundwater Commons Game")
    st.caption("Tragedy of the Commons in groundwater pumping: private payoffs vs shared stock.")

    with st.sidebar:
        st.markdown("### How this game works")
        st.markdown(
            "- Profit each round: **P·q − ½γq² − c₀q − c₁(Smax−S)q − τq** (if tax τ is on)."\
            "\n- Stock update: **Sₜ₊₁ = Sₜ + R − Σqᵢ**. If a threshold is enabled and **S** falls below it, future price is multiplied by a factor (<1)."\
            "\n- Scoring: discounted cumulative profit ∑ₜ πₜ/(1+r)ᵗ."
        )
        st.markdown("---")
        st.markdown("**Instructor tip:** Run Stage A first, then host a Stage B room. Optionally enable a tax or cap, or a price-drop threshold to create a vivid inflection point.")

    tab_a, tab_b = st.tabs(["Stage A (Solo)", "Stage B (Multiplayer)"])
    with tab_a:
        stage_a_solo()
    with tab_b:
        stage_b_multiplayer()


if __name__ == "__main__":
    main()
