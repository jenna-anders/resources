"""
Groundwater Commons Game - Flask + Socket.io Version
Real-time multiplayer with session persistence
"""
from flask import Flask, render_template, request, session, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import sqlite3
import json
import uuid
import random
import string
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")

DB_PATH = "commons_game.db"
SOLO_WELLS = 4

# ------------------------------
# Database functions
# ------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
              code TEXT PRIMARY KEY,
              created_at TEXT,
              params_json TEXT,
              state_json TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS players (
              player_id TEXT PRIMARY KEY,
              room_code TEXT,
              name TEXT,
              well_index INTEGER,
              cumulative_profit REAL,
              joined_at TEXT,
              last_active TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rounds (
              room_code TEXT,
              round_index INTEGER,
              S REAL,
              recharge REAL,
              group_q REAL,
              event TEXT,
              PRIMARY KEY (room_code, round_index)
            );
        """)
        cur.execute("""
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
        """)
        conn.commit()

init_db()

# ------------------------------
# Core economics
# ------------------------------

def profit_one_period(q: float, S: float, params: Dict) -> float:
    P = params["P"]
    gamma = params["gamma"]
    c0 = params["c0"]
    c1 = params["c1"]
    Smax = params["Smax"]
    revenue = P * q - 0.5 * gamma * q * q
    cost = c0 * q + c1 * (Smax - S) * q
    return revenue - cost

def next_stock(S: float, R: float, q_total: float) -> float:
    return max(0.0, S + R - q_total)

# ------------------------------
# Database helpers
# ------------------------------

def room_exists(code: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM rooms WHERE code=?", (code,))
        return cur.fetchone() is not None

def get_room_capacity(code: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT state_json FROM rooms WHERE code=?", (code,))
        row = cur.fetchone()
        if not row:
            return 6
        state = json.loads(row[0])
        return int(state.get("players_expected", 6))

def create_room(params: Dict) -> str:
    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    now = datetime.utcnow().isoformat()
    capacity = int(params.get("players_expected", 6))
    state = {
        "status": "lobby",
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

def list_players(code: str) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT player_id, name, well_index, cumulative_profit, joined_at, last_active FROM players WHERE room_code=? ORDER BY well_index ASC",
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
            }
            for r in cur.fetchall()
        ]

def add_or_get_player(code: str, name: str) -> Dict:
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
                "INSERT INTO players(player_id, room_code, name, well_index, cumulative_profit, joined_at, last_active) VALUES(?,?,?,?,?,?,?)",
                (pid, code, name, win, 0.0, now, now),
            )
            conn.commit()
            cum = 0.0
        return {"player_id": pid, "well_index": win, "cumulative_profit": cum}

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
            """INSERT INTO actions(room_code, round_index, player_id, q, submitted, profit, timestamp) VALUES(?,?,?,?,?,?,?)
             ON CONFLICT(room_code, round_index, player_id) DO UPDATE SET q=excluded.q, submitted=excluded.submitted, profit=excluded.profit, timestamp=excluded.timestamp""",
            (code, round_index, player_id, float(q), int(submitted), profit_val, now),
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

def maybe_advance_round(code: str):
    """Advance when all players have submitted. Broadcasts update via Socket.io."""
    params, state = load_room(code)
    if state["status"] != "running":
        return False
    
    t = state["current_round"]
    rr = get_round_row(code, t)
    S_t = rr["S"]
    R_t = params["R"]
    
    players = list_players(code)
    capacity = get_room_capacity(code)
    n_players = min(len(players), capacity)
    if n_players == 0:
        return False
    
    acts = fetch_actions(code, t)
    joined_ids = {p["player_id"] for p in players[:capacity]}
    submitted = [a for a in acts if a["player_id"] in joined_ids and a["submitted"]]
    if len(submitted) < n_players:
        return False
    
    # Compute profits
    q_map = {a["player_id"]: float(a["q"]) for a in submitted}
    group_q = sum(q_map.values())
    
    per_player_profit = {}
    for p in players[:capacity]:
        pid = p["player_id"]
        q_i = q_map.get(pid, 0.0)
        pi_i = profit_one_period(q_i, S_t, params)
        per_player_profit[pid] = pi_i
    
    S_next = next_stock(S_t, R_t, group_q)
    
    # Update DB
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
    
    write_round_result(code, t + 1, S_next, R_t, group_q, {})
    
    T = params.get("T", 8)
    if S_next <= 0.0 or (t + 1) >= T:
        state["status"] = "finished"
    else:
        state["current_round"] = t + 1
    save_room_state(code, state)
    
    # Broadcast update to all players in room
    socketio.emit('round_advanced', {'round': state["current_round"], 'status': state["status"]}, room=code)
    return True

# ------------------------------
# Flask routes
# ------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create_room', methods=['POST'])
def api_create_room():
    data = request.json
    params = {
        "P": data.get("P", 10.0),
        "gamma": data.get("gamma", 0.08),
        "c0": data.get("c0", 2.0),
        "c1": data.get("c1", 0.006),
        "S0": data.get("S0", 1200.0),
        "Smax": data.get("Smax", 1200.0),
        "R": data.get("R", 60.0),
        "qmax": data.get("qmax", 80.0),
        "T": data.get("T", 8),
        "players_expected": data.get("players_expected", 6)
    }
    code = create_room(params)
    return jsonify({"code": code})

@app.route('/api/join_room', methods=['POST'])
def api_join_room():
    data = request.json
    code = data.get("code")
    name = data.get("name")
    
    if not room_exists(code):
        return jsonify({"error": "Room not found"}), 404
    
    try:
        player_info = add_or_get_player(code, name)
        session['player_id'] = player_info['player_id']
        session['room_code'] = code
        session['player_name'] = name
        return jsonify(player_info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/start_game', methods=['POST'])
def api_start_game():
    data = request.json
    code = data.get("code")
    
    params, state = load_room(code)
    if state["status"] == "lobby":
        state["status"] = "running"
        save_room_state(code, state)
        socketio.emit('game_started', {}, room=code)
        return jsonify({"success": True})
    return jsonify({"error": "Game already started"}), 400

@app.route('/api/submit_action', methods=['POST'])
def api_submit_action():
    data = request.json
    code = data.get("code")
    player_id = session.get('player_id')
    q = data.get("q")
    
    params, state = load_room(code)
    t = state["current_round"]
    
    # Calculate profit preview
    rr = get_round_row(code, t)
    profit = profit_one_period(q, rr["S"], params)
    
    upsert_action(code, t, player_id, q, submitted=True, profit_val=profit)
    
    # Notify room that someone submitted
    socketio.emit('player_submitted', {'player_id': player_id}, room=code)
    
    # Check if round should advance
    advanced = maybe_advance_round(code)
    
    return jsonify({"success": True, "advanced": advanced, "profit": profit})

@app.route('/api/room_state', methods=['GET'])
def api_room_state():
    code = request.args.get("code")
    player_id = session.get('player_id')
    
    params, state = load_room(code)
    players = list_players(code)
    t = state["current_round"]
    rr = get_round_row(code, t)
    
    acts = fetch_actions(code, t)
    my_act = next((a for a in acts if a["player_id"] == player_id), None)
    has_act = my_act is not None
    
    submitted_ids = {a["player_id"] for a in acts if a["submitted"]}
    player_status = []
    for p in players:
        player_status.append({
            "name": p["name"],
            "well": p["well_index"] + 1,
            "submitted": p["player_id"] in submitted_ids,
            "cumulative_profit": p["cumulative_profit"]
        })
    
    return jsonify({
        "status": state["status"],
        "current_round": state["current_round"],
        "S": rr["S"],
        "Smax": params["Smax"],
        "qmax": params["qmax"],
        # key change: no default; None means “no prior choice saved”
        "my_q": (my_act["q"] if has_act else None),
        "my_has_act": has_act,
        "my_submitted": bool(my_act and my_act["submitted"]) if has_act else False,
        "players": player_status,
        "my_cumulative": next((p["cumulative_profit"] for p in players if p["player_id"] == player_id), 0.0),
        "params": params
    })


@app.route('/api/force_advance', methods=['POST'])
def api_force_advance():
    data = request.json
    code = data.get("code")
    
    params, state = load_room(code)
    t = state["current_round"]
    
    # Submit 0 for all non-submitted players
    players = list_players(code)
    acts = fetch_actions(code, t)
    submitted_ids = {a["player_id"] for a in acts if a["submitted"]}
    
    for p in players:
        if p["player_id"] not in submitted_ids:
            upsert_action(code, t, p["player_id"], 0.0, submitted=True, profit_val=0.0)
    
    maybe_advance_round(code)
    return jsonify({"success": True})

@app.route('/api/end_game', methods=['POST'])
def api_end_game():
    data = request.json
    code = data.get("code")
    
    params, state = load_room(code)
    state["status"] = "finished"
    save_room_state(code, state)
    socketio.emit('game_ended', {}, room=code)
    return jsonify({"success": True})



# ---------- SOLO MODE ----------
from flask import session, request, jsonify, render_template

# ---------- SOLO MODE HELPERS ----------

def _solo_defaults():
    return {
        "P": 10.0,
        "gamma": 0.08,
        "c0": 2.0,
        "c1": 0.006,
        "S0": 1200.0,
        "Smax": 1200.0,
        "R": 60.0,
        "qmax": 80.0,
        "T": 8,
        "wells": 6,
    }

def _solo_state():
    return session.get("solo_state")

def _solo_save(state):
    session["solo_state"] = state

def _solo_profit_per_well(q, P, c0, c1):
    return max(0.0, P*q - c0*q - c1*(q**2))

def _solo_next_S(S, R, total_q, gamma, Smax):
    # For solo mode, use the same stock transition as the group game:
    # S_{t+1} = max(0, S_t + R - total_q)
    return next_stock(S, R, total_q)

# ---------- SOLO ROUTES ----------

@app.route("/solo")
def solo_page():
    return render_template("solo.html")

@app.route("/api/solo/create", methods=["POST"])
def solo_create():
    data = request.get_json(force=True, silent=True) or {}
    params = _solo_defaults()
    # allow overrides from client
    for k, v in data.items():
        if k in params:
            params[k] = type(params[k])(v)

    state = {
        "params": params,
        "round": 1,
        "S": float(params["S0"]),
        "cumulative_profit": 0.0,
        "last_profit": None,
        "submitted": False,
    }
    _solo_save(state)
    return jsonify({
        "ok": True,
        "state": {
            "round": state["round"],
            "S": state["S"],
            "Smax": params["Smax"],
            "qmax": params["qmax"],
            "wells": params["wells"],
            "T": params["T"],
            "submitted": state["submitted"],
            "last_profit": state["last_profit"],
            "cumulative_profit": state["cumulative_profit"],
            "params": params,
        }
    })

@app.route("/api/solo/state", methods=["GET"])
def solo_state():
    st = _solo_state()
    if not st:
        # lazily initialize if needed
        params = _solo_defaults()
        st = {
            "params": params,
            "round": 1,
            "S": float(params["S0"]),
            "cumulative_profit": 0.0,
            "last_profit": None,
            "submitted": False,
        }
        _solo_save(st)

    p = st["params"]
    return jsonify({
        "round": st["round"],
        "S": st["S"],
        "Smax": p["Smax"],
        "qmax": p["qmax"],
        "wells": p["wells"],
        "T": p["T"],
        "submitted": st["submitted"],
        "last_profit": st["last_profit"],
        "cumulative_profit": st["cumulative_profit"],
        "params": p,
    })

@app.route("/api/solo/submit", methods=["POST"])
def solo_submit():
    st = _solo_state()
    if not st:
        return jsonify({"error": "No solo session"}), 400

    data = request.get_json(force=True, silent=True) or {}
    try:
        q = float(data.get("q", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid q"}), 400

    p = st["params"]
    q = max(0.0, min(p["qmax"], q))
    wells = int(p["wells"])
    total_q = wells * q

    per_profit = _solo_profit_per_well(q, p["P"], p["c0"], p["c1"])
    total_profit = wells * per_profit

    # *** This is where S changes ***
    S_old = st["S"]
    S_new = _solo_next_S(S_old, p["R"], total_q, p["gamma"], p["Smax"])

    st["S"] = S_new
    st["cumulative_profit"] += total_profit
    st["last_profit"] = total_profit
    st["submitted"] = True

    _solo_save(st)
    return jsonify({
        "ok": True,
        "round": st["round"],
        "S": S_new,
        "Smax": p["Smax"],
        "last_profit": total_profit,
        "cumulative_profit": st["cumulative_profit"],
    })

@app.route("/api/solo/next", methods=["POST"])
def solo_next():
    st = _solo_state()
    if not st:
        return jsonify({"error": "No solo session"}), 400

    st["round"] += 1
    if st["round"] > st["params"]["T"]:
        st["round"] = st["params"]["T"]
        st["submitted"] = True
    else:
        st["submitted"] = False

    _solo_save(st)
    return jsonify({"ok": True, "round": st["round"], "submitted": st["submitted"]})

@app.route("/api/solo/reset", methods=["POST"])
def solo_reset():
    session.pop("solo_state", None)
    return jsonify({"ok": True})
# ---------- END SOLO MODE ----------




# ------------------------------
# Socket.io events
# ------------------------------

@socketio.on('join')
def on_join(data):
    code = data['code']
    join_room(code)
    emit('joined', {'code': code})

@socketio.on('leave')
def on_leave(data):
    code = data['code']
    leave_room(code)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
