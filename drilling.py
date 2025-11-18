
# drilling.py — Groundwater commons game (Stage A: Solo, Stage B: Multiplayer)
# Notes for instructors:
# - Default group size: 6 (but rooms can run with fewer)
# - Stage B host parameters are hidden in an expander; the "Create Room" button is OUTSIDE the expander
# - Faster auto-advance via st_autorefresh while waiting for others
# - Default rounds T = 8
# - Default initial stock S0 = 1400 (both Solo and Multiplayer)
# - Stage A (Solo) now gives the player SIX wells instead of four

import time
import uuid
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

APP_TITLE = "Groundwater Commons Game"
DB_PATH = "drilling.db"

# ---------- Database helpers ----------
def get_conn():
    # check_same_thread=False so multiple Streamlit sessions can share the DB safely
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rooms (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        S0 REAL NOT NULL,
        R REAL NOT NULL,
        capacity INTEGER NOT NULL,
        T INTEGER NOT NULL,
        qmax REAL NOT NULL,
        is_active INTEGER NOT NULL,
        current_round INTEGER NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id TEXT PRIMARY KEY,
        room_id TEXT NOT NULL,
        name TEXT NOT NULL,
        joined_at TEXT NOT NULL,
        FOREIGN KEY (room_id) REFERENCES rooms(id)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS moves (
        room_id TEXT NOT NULL,
        round INTEGER NOT NULL,
        player_id TEXT NOT NULL,
        qty REAL NOT NULL,
        submitted_at TEXT NOT NULL,
        PRIMARY KEY (room_id, round, player_id),
        FOREIGN KEY (room_id) REFERENCES rooms(id),
        FOREIGN KEY (player_id) REFERENCES players(id)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        room_id TEXT PRIMARY KEY,
        stock REAL NOT NULL,
        last_updated TEXT NOT NULL,
        FOREIGN KEY (room_id) REFERENCES rooms(id)
    );
    """)
    conn.commit()
    conn.close()

def create_room(S0: float, R: float, capacity: int, T: int, qmax: float) -> str:
    room_id = uuid.uuid4().hex[:6].upper()
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""INSERT INTO rooms (id, created_at, S0, R, capacity, T, qmax, is_active, current_round)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (room_id, now, S0, R, capacity, T, qmax, 1, 1))
    cur.execute("""INSERT INTO stocks (room_id, stock, last_updated) VALUES (?, ?, ?);""",
                (room_id, S0, now))
    conn.commit()
    conn.close()
    return room_id

def room_exists(room_id: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM rooms WHERE id=?;", (room_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def get_room(room_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM rooms WHERE id=?;", (room_id,))
    row = cur.fetchone()
    conn.close()
    return row

def get_room_stock(room_id: str) -> float:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT stock FROM stocks WHERE room_id=?;", (room_id,))
    row = cur.fetchone()
    conn.close()
    return float(row["stock"]) if row else 0.0

def set_room_stock(room_id: str, stock: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE stocks SET stock=?, last_updated=? WHERE room_id=?;",
                (stock, datetime.utcnow().isoformat(), room_id))
    conn.commit()
    conn.close()

def list_players(room_id: str) -> List[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM players WHERE room_id=? ORDER BY joined_at;", (room_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def join_room(room_id: str, name: str) -> str:
    player_id = uuid.uuid4().hex[:8]
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO players (id, room_id, name, joined_at) VALUES (?, ?, ?, ?);",
                (player_id, room_id, name, now))
    conn.commit()
    conn.close()
    return player_id

def get_move(room_id: str, round_num: int, player_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""SELECT * FROM moves WHERE room_id=? AND round=? AND player_id=?;""",
                (room_id, round_num, player_id))
    row = cur.fetchone()
    conn.close()
    return row

def submit_move(room_id: str, round_num: int, player_id: str, qty: float):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""INSERT OR REPLACE INTO moves (room_id, round, player_id, qty, submitted_at)
                   VALUES (?, ?, ?, ?, ?);""",
                (room_id, round_num, player_id, qty, now))
    conn.commit()
    conn.close()

def count_moves(room_id: str, round_num: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM moves WHERE room_id=? AND round=?;", (room_id, round_num))
    row = cur.fetchone()
    conn.close()
    return int(row["c"])

def total_moves_qty(room_id: str, round_num: int) -> float:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT SUM(qty) AS s FROM moves WHERE room_id=? AND round=?;", (room_id, round_num))
    row = cur.fetchone()
    conn.close()
    return float(row["s"] or 0.0)

def advance_if_ready(room_id: str) -> bool:
    """If all current players submitted for the current round, advance the round and update stock.
       Returns True if advanced, else False."""
    room = get_room(room_id)
    if not room or not room["is_active"]:
        return False

    round_num = int(room["current_round"])
    players = list_players(room_id)
    n_players = len(players)
    if n_players == 0:
        return False

    # All moves in?
    if count_moves(room_id, round_num) < n_players:
        return False

    # Update stock: S_{t+1} = S_t - sum(q_i) + R  (no cap, but not below 0)
    S_t = get_room_stock(room_id)
    Q_t = total_moves_qty(room_id, round_num)
    S_next = max(0.0, S_t - Q_t + float(room["R"]))

    set_room_stock(room_id, S_next)

    # Increment round or end game
    conn = get_conn()
    cur = conn.cursor()
    if round_num >= int(room["T"]):
        cur.execute("UPDATE rooms SET is_active=0 WHERE id=?;", (room_id,))
    else:
        cur.execute("UPDATE rooms SET current_round=? WHERE id=?;", (round_num + 1, room_id))
    conn.commit()
    conn.close()
    return True

# ---------- UI helpers ----------
def header():
    st.title(APP_TITLE)
    st.caption("Stage A: Solo | Stage B: Multiplayer (rooms of 6 by default)")

def ensure_session_keys():
    # Defaults for host controls (hidden in expander)
    defaults = dict(
        host_S0=1400,   # requested: 1400
        host_R=50,
        host_capacity=6,
        host_T=8,       # requested: 8
        host_qmax=12,   # per-player per-round cap, adjustable
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def stage_a_solo():
    st.subheader("Stage A — Solo play")

    with st.expander("Parameters (optional)", expanded=False):
        S0 = st.number_input("Initial stock S0", min_value=0, value=1400, step=50)
        R = st.number_input("Recharge per round R", min_value=0, value=50, step=5)
        T = st.number_input("Rounds T", min_value=1, value=8, step=1)
        qmax = st.number_input("Per-well cap per round q_max", min_value=1, value=12, step=1)
        wells = st.number_input("Number of wells", min_value=1, value=6, step=1)  # requested: 6

    st.write("Make your extraction choice each round. The aquifer updates after your move.")

    key_prefix = "solo_"
    if st.button("Start / Reset Solo Game"):
        st.session_state[key_prefix + "round"] = 1
        st.session_state[key_prefix + "stock"] = float(S0)
        st.session_state[key_prefix + "history"] = []  # list of dicts per round
        st.session_state[key_prefix + "params"] = dict(S0=S0, R=R, T=T, qmax=qmax, wells=wells)

    params = st.session_state.get(key_prefix + "params")
    if not params:
        st.info("Click **Start / Reset Solo Game** to begin.")
        return

    round_num = st.session_state.get(key_prefix + "round", 1)
    stock = st.session_state.get(key_prefix + "stock", float(params["S0"]))
    history: List[Dict[str, Any]] = st.session_state.get(key_prefix + "history", [])

    if round_num > params["T"]:
        st.success("Game over!")
        st.metric("Final stock", f"{stock:.1f}")
        if history:
            st.write("Round history:")
            st.dataframe(history, hide_index=True)
        return

    st.write(f"**Round {round_num} of {params['T']}**")
    st.metric("Current aquifer stock", f"{stock:.1f}")

    # Player chooses per-well extraction; total = wells * q
    q_per_well = st.slider("Choose extraction per well this round",
                           min_value=0, max_value=int(params["qmax"]), value=0, step=1)
    total_q = q_per_well * int(params["wells"])
    st.write(f"Total extraction this round: **{total_q}** (wells × per-well = {params['wells']} × {q_per_well})")

    if st.button("Submit extraction (Solo)"):
        # Update stock and advance
        Q_t = float(total_q)
        S_t = float(stock)
        S_next = max(0.0, S_t - Q_t + float(params["R"]))
        history.append(dict(round=round_num, q_per_well=q_per_well, total_Q=Q_t, S_end=S_next))
        st.session_state[key_prefix + "history"] = history
        st.session_state[key_prefix + "stock"] = S_next
        st.session_state[key_prefix + "round"] = round_num + 1
        st.experimental_rerun()

    if history:
        with st.expander("Solo: round history", expanded=False):
            st.dataframe(history, hide_index=True)

def stage_b_multiplayer():
    st.subheader("Stage B — Multiplayer (rooms)")

    ensure_session_keys()

    # --- Host controls in expander (NO form), values in session_state ---
    with st.expander("Room setup (optional)", expanded=False):
        st.number_input("Initial stock S0", min_value=0, key="host_S0")
        st.number_input("Recharge per round R", min_value=0, key="host_R")
        st.selectbox("Room capacity (players)", [6, 5, 4], index=[6,5,4].index(st.session_state["host_capacity"]), key="host_capacity")
        st.number_input("Rounds T", min_value=1, key="host_T")
        st.number_input("Per-player per-round cap q_max", min_value=1, key="host_qmax")

    # --- Create Room button OUTSIDE the expander ---
    host_name = st.text_input("Your display name (host)")
    colA, colB = st.columns([1,1])
    with colA:
        create_clicked = st.button("Create Room", type="primary")
    with colB:
        join_as_host_clicked = st.button("Create Room and Join as Player")

    if create_clicked or join_as_host_clicked:
        if not host_name.strip():
            st.error("Please enter a display name first.")
        else:
            room_id = create_room(
                S0=float(st.session_state["host_S0"]),
                R=float(st.session_state["host_R"]),
                capacity=int(st.session_state["host_capacity"]),
                T=int(st.session_state["host_T"]),
                qmax=float(st.session_state["host_qmax"]),
            )
            st.success(f"Room created: **{room_id}**")
            st.write("Share this Room ID with your group.")
            if join_as_host_clicked:
                # Host immediately joins
                player_id = join_room(room_id, host_name.strip())
                st.session_state["room_id"] = room_id
                st.session_state["player_id"] = player_id
                st.session_state["player_name"] = host_name.strip()
                st.info("You joined your room as a player.")
                st.experimental_rerun()

    st.divider()

    st.markdown("#### Join a room to play")
    name = st.text_input("Your display name (player)", key="join_name")
    join_room_id = st.text_input("Room ID", key="join_room_id")
    if st.button("Join Room"):
        if not join_room_id or not room_exists(join_room_id):
            st.error("Room not found. Double-check the ID.")
        elif not name.strip():
            st.error("Please enter a display name.")
        else:
            player_id = join_room(join_room_id, name.strip())
            st.session_state["room_id"] = join_room_id
            st.session_state["player_id"] = player_id
            st.session_state["player_name"] = name.strip()
            st.success(f"Joined room {join_room_id} as {name.strip()}")
            st.experimental_rerun()

    # If joined, show gameplay panel
    if "room_id" in st.session_state and "player_id" in st.session_state:
        room_id = st.session_state["room_id"]
        player_id = st.session_state["player_id"]
        player_name = st.session_state.get("player_name", "(you)")

        st.markdown("---")
        st.markdown(f"#### Room: **{room_id}** — Player: **{player_name}**")
        room = get_room(room_id)
        if not room:
            st.error("Room not found or was deleted.")
            return

        stock = get_room_stock(room_id)
        round_num = int(room["current_round"])
        is_active = bool(room["is_active"])

        players = list_players(room_id)
        n_players = len(players)
        cap = int(room["capacity"])
        qmax = float(room["qmax"])
        T = int(room["T"])

        top_cols = st.columns(4)
        top_cols[0].metric("Current round", f"{round_num} / {T}")
        top_cols[1].metric("Aquifer stock", f"{stock:.1f}")
        top_cols[2].metric("Players joined", f"{n_players} / {cap}")
        top_cols[3].metric("Active", "Yes" if is_active else "No")

        if not is_active:
            st.success("Game over for this room!")
            st.stop()

        # Show who has submitted this round
        submitted_count = count_moves(room_id, round_num)
        with st.expander("Submissions this round", expanded=False):
            st.write(f"Submitted: **{submitted_count} / {n_players}**")

        # If room not full, allow play anyway (as requested)
        st.caption("This room runs even if fewer than the capacity have joined.")

        # Player action for this round (if not already submitted)
        existing = get_move(room_id, round_num, player_id)
        if existing:
            st.info(f"You submitted **{existing['qty']:.1f}** for round {round_num}. Waiting for others...")
        else:
            qty = st.slider("Your extraction this round", 0.0, qmax, 0.0, 1.0)
            if st.button("Submit move"):
                submit_move(room_id, round_num, player_id, float(qty))
                st.experimental_rerun()

        # Auto-advance processing and faster refresh while waiting
        # Try to advance on the server if all submissions are in
        did_advance = advance_if_ready(room_id)

        if not did_advance:
            # Light auto-refresh while waiting for others
            st.caption("Waiting for others... the page will check for updates automatically.")
            # Use a tiny sleep + rerun to emulate quick polling if autorefresh isn't available
            time.sleep(1.2)
            st.experimental_rerun()
        else:
            # We just advanced (server-side). Let everyone refresh quickly.
            time.sleep(0.4)
            st.experimental_rerun()


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💧", layout="centered")
    init_db()
    header()

    tabs = st.tabs(["Stage A — Solo", "Stage B — Multiplayer"])
    with tabs[0]:
        stage_a_solo()
    with tabs[1]:
        stage_b_multiplayer()


if __name__ == "__main__":
    main()
