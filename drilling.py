import streamlit as st
import uuid
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
import random

# -------------------------------
# Utilities & App Constants
# -------------------------------

APP_TITLE = "Drilling Decisions: Solo & Paired Game"
DEFAULT_PIN = "1234"  # Instructor PIN (change in sidebar)
REFRESH_SEC = 1.2     # UI auto-refresh frequency while waiting for a match / partner

def gen_user_id() -> str:
    return str(uuid.uuid4())[:8]

def now_ts() -> float:
    return time.time()

# -------------------------------
# Economic Model (configurable)
# -------------------------------

@dataclass
class EconConfig:
    # Four sites with private profits if drilled
    site_labels: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    site_private_profits: List[float] = field(default_factory=lambda: [80.0, 70.0, 60.0, 50.0])

    # External damage function D(T) where T = total wells drilled across both players
    # D(T) = d0 + d1*T + d2*T^2 (flexible enough to match many worksheets)
    d0: float = 0.0
    d1: float = 0.0
    d2: float = 10.0

    # Share of damage borne by each player in paired play (e.g., 0.5 when two symmetric players)
    damage_share_per_player: float = 0.5

    # Site assignment in paired play (by index into site_labels/profits)
    player1_sites: List[int] = field(default_factory=lambda: [0, 1])  # Sites A,B
    player2_sites: List[int] = field(default_factory=lambda: [2, 3])  # Sites C, D

    # Number of paired rounds before a room auto-closes (students can replay by re-queueing)
    paired_rounds: int = 3

    def damage(self, T: int) -> float:
        return self.d0 + self.d1 * T + self.d2 * (T ** 2)

# -------------------------------
# Room & Matchmaking State
# -------------------------------

@dataclass
class Decision:
    # drill decisions mapped by site index: True/False
    drills: Dict[int, bool] = field(default_factory=dict)
    submitted_at: float = 0.0

@dataclass
class RoundResult:
    round_id: int
    T: int
    damage: float
    p1_profit: float
    p2_profit: float
    p1_drills: Dict[int, bool]
    p2_drills: Dict[int, bool]
    finished_at: float

@dataclass
class Room:
    class_code: str
    room_id: str
    created_at: float
    econ: EconConfig
    p1_id: Optional[str] = None
    p2_id: Optional[str] = None
    p1_name: Optional[str] = None
    p2_name: Optional[str] = None
    # per-round state
    round_index: int = 1
    p1_decision: Optional[Decision] = None
    p2_decision: Optional[Decision] = None
    results: List[RoundResult] = field(default_factory=list)
    # lifecycle
    closed: bool = False

    def both_ready(self) -> bool:
        return (self.p1_id is not None) and (self.p2_id is not None) and (not self.closed)

    def round_open(self) -> bool:
        if self.closed:
            return False
        # Round is open if decisions are incomplete
        return not (self.p1_decision and self.p2_decision)

    def reset_round(self):
        self.p1_decision = None
        self.p2_decision = None

    def close_if_done(self):
        if self.round_index > self.econ.paired_rounds:
            self.closed = True

# Global “in-memory” store for matchmaking and results
# Works well for small/medium classes on a single Streamlit instance.
@st.cache_resource
def get_store():
    return {
        "econ_configs": {},     # class_code -> EconConfig (dict-serialized)
        "queue": {},            # class_code -> [ (user_id, name), ... ]
        "rooms": {},            # room_id -> Room
        "archive": [],          # closed rooms' results (list of dicts)
    }

def load_econ(class_code: str) -> EconConfig:
    store = get_store()
    raw = store["econ_configs"].get(class_code)
    if raw:
        ec = EconConfig(**raw)
        return ec
    # Default if not set yet for this class code
    ec = EconConfig()
    store["econ_configs"][class_code] = asdict(ec)
    return ec

def save_econ(class_code: str, ec: EconConfig):
    store = get_store()
    store["econ_configs"][class_code] = asdict(ec)

def enqueue_student(class_code: str, user_id: str, name: str):
    store = get_store()
    q = store["queue"].setdefault(class_code, [])
    # Avoid duplicate entries for same user
    q = [item for item in q if item[0] != user_id]
    q.append((user_id, name))
    store["queue"][class_code] = q

def try_match(class_code: str) -> Optional[str]:
    store = get_store()
    q = store["queue"].get(class_code, [])
    if len(q) >= 2:
        (u1, n1) = q.pop(0)
        (u2, n2) = q.pop(0)
        store["queue"][class_code] = q

        econ = load_econ(class_code)
        room_id = str(uuid.uuid4())[:8]
        room = Room(
            class_code=class_code,
            room_id=room_id,
            created_at=now_ts(),
            econ=econ,
            p1_id=u1,
            p2_id=u2,
            p1_name=n1,
            p2_name=n2,
        )
        store["rooms"][room_id] = room
        return room_id
    return None

def get_room(room_id: str) -> Optional[Room]:
    store = get_store()
    return store["rooms"].get(room_id)

def save_room(room: Room):
    store = get_store()
    store["rooms"][room.room_id] = room

def archive_room(room: Room):
    store = get_store()
    # flatten each result to a row
    for r in room.results:
        row = {
            "class_code": room.class_code,
            "room_id": room.room_id,
            "round_id": r.round_id,
            "T": r.T,
            "damage": r.damage,
            "p1_name": room.p1_name,
            "p2_name": room.p2_name,
            "p1_profit": r.p1_profit,
            "p2_profit": r.p2_profit,
            "p1_drills": json.dumps(r.p1_drills),
            "p2_drills": json.dumps(r.p2_drills),
            "finished_at": r.finished_at,
        }
        store["archive"].append(row)
    # Mark closed (already in room), keep record in rooms for display

def export_archive_df() -> pd.DataFrame:
    store = get_store()
    return pd.DataFrame(store["archive"])

def reset_class(class_code: str):
    store = get_store()
    store["queue"][class_code] = []
    # Close and archive all current rooms for this class_code
    to_close = []
    for rid, r in list(store["rooms"].items()):
        if r.class_code == class_code and not r.closed:
            r.closed = True
            archive_room(r)
            save_room(r)
        if r.class_code == class_code:
            to_close.append(rid)
    # Keep rooms but they are closed; you can also purge them if you want a hard reset.

# -------------------------------
# Economics: Payoff Computation
# -------------------------------

def solo_profit(econ: EconConfig, solo_drills: Dict[int, bool]) -> Tuple[int, float, Dict[int, float]]:
    """Returns (T, total_profit, per_site_private) for solo stage."""
    per_site_private = {}
    T = 0
    private_sum = 0.0
    for idx, drill in solo_drills.items():
        if drill:
            T += 1
            private_sum += econ.site_private_profits[idx]
            per_site_private[idx] = econ.site_private_profits[idx]
        else:
            per_site_private[idx] = 0.0
    damage = econ.damage(T)
    total_profit = private_sum - damage  # internalizes 100% of damage when solo
    return T, total_profit, per_site_private

def paired_profits(econ: EconConfig, p1_drills: Dict[int, bool], p2_drills: Dict[int, bool]) -> Tuple[int, float, float]:
    """Returns (T, p1_profit, p2_profit) for paired stage."""
    # Total wells
    T = sum(int(x) for x in p1_drills.values()) + sum(int(x) for x in p2_drills.values())
    # Private profits per player
    p1_private = sum(econ.site_private_profits[i] for i, d in p1_drills.items() if d)
    p2_private = sum(econ.site_private_profits[i] for i, d in p2_drills.items() if d)
    # External damage shared
    D = econ.damage(T)
    share = econ.damage_share_per_player
    p1 = p1_private - share * D
    p2 = p2_private - share * D
    return T, p1, p2

# -------------------------------
# UI Components
# -------------------------------

def sidebar_instructor_panel():
    st.sidebar.subheader("Instructor Panel")
    pin = st.sidebar.text_input("Instructor PIN", type="password", value=DEFAULT_PIN)
    entered = st.sidebar.text_input("Enter PIN to unlock settings", type="password")

    if entered != pin:
        st.sidebar.info("Enter correct PIN to edit settings.")
        return

    st.sidebar.success("Instructor settings unlocked.")
    class_code = st.sidebar.text_input("Class Code (share with students)", value=st.session_state.get("class_code", "ENVECON-101"))
    st.session_state["class_code"] = class_code

    econ = load_econ(class_code)

    with st.sidebar.expander("Payoff Parameters", expanded=False):
        labels = st.text_input("Site Labels (comma-separated)", value=",".join(econ.site_labels)).split(",")
        profits_str = st.text_input("Site Private Profits (comma-separated)", value=",".join(str(x) for x in econ.site_private_profits))
        profits = [float(x.strip()) for x in profits_str.split(",") if x.strip() != ""]

        col_d0, col_d1, col_d2 = st.columns(3)
        with col_d0:
            d0 = st.number_input("d0 in D(T)=d0+d1*T+d2*T^2", value=float(econ.d0), step=1.0)
        with col_d1:
            d1 = st.number_input("d1 in D(T)=d0+d1*T+d2*T^2", value=float(econ.d1), step=1.0)
        with col_d2:
            d2 = st.number_input("d2 in D(T)=d0+d1*T+d2*T^2", value=float(econ.d2), step=1.0)

        share = st.slider("Damage share per player (paired)", min_value=0.0, max_value=1.0, value=float(econ.damage_share_per_player), step=0.05)

        st.markdown("**Paired site assignments (indexes)**")
        st.caption("Indexes correspond to the order of site labels above (starting at 0).")
        p1_idx_str = st.text_input("Player 1 site indexes (comma-separated)", value=",".join(str(i) for i in econ.player1_sites))
        p2_idx_str = st.text_input("Player 2 site indexes (comma-separated)", value=",".join(str(i) for i in econ.player2_sites))
        p1_idx = [int(x.strip()) for x in p1_idx_str.split(",") if x.strip() != ""]
        p2_idx = [int(x.strip()) for x in p2_idx_str.split(",") if x.strip() != ""]

        rounds = st.number_input("Number of paired rounds", min_value=1, max_value=20, value=int(econ.paired_rounds))

        if st.button("Save Settings"):
            econ_new = EconConfig(
                site_labels=labels,
                site_private_profits=profits,
                d0=d0, d1=d1, d2=d2,
                damage_share_per_player=share,
                player1_sites=p1_idx,
                player2_sites=p2_idx,
                paired_rounds=rounds,
            )
            save_econ(class_code, econ_new)
            st.success("Saved class settings.")

    with st.sidebar.expander("Admin / Monitoring", expanded=False):
        store = get_store()
        q_len = len(store["queue"].get(class_code, []))
        st.write(f"Students waiting in queue: **{q_len}**")
        rooms = [r for r in store["rooms"].values() if r.class_code == class_code]
        st.write(f"Active rooms: **{sum(1 for r in rooms if not r.closed)}**, Closed rooms: **{sum(1 for r in rooms if r.closed)}**")

        if st.button("Export All Results (CSV)"):
            df = export_archive_df()
            if df.empty:
                st.warning("No results to export yet.")
            else:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{class_code}_drilling_results.csv",
                    mime="text/csv",
                )

        if st.button("Reset / Close All Rooms for this Class Code"):
            reset_class(class_code)
            st.success("Rooms closed & queue cleared for this Class Code.")

def student_header(class_code: str, econ: EconConfig):
    st.markdown(f"### Class Code: `{class_code}`")
    st.caption("If this code doesn't match what your instructor gave you, change it in the sidebar (Instructor) or ask your instructor.")

def stage_header(title: str, subtitle: Optional[str] = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

# -------------------------------
# Solo Stage UI
# -------------------------------

def solo_stage_ui(econ: EconConfig):
    stage_header("Stage 1 — Solo (You own all sites)",
                 "Decide whether to drill at each site. Your profit = sum of private profits − the full damage D(T).")

    if "solo_drills" not in st.session_state:
        st.session_state["solo_drills"] = {i: False for i in range(len(econ.site_labels))}
    drills = st.session_state["solo_drills"]

    st.write("**Your sites**")
    cols = st.columns(len(econ.site_labels))
    for i, label in enumerate(econ.site_labels):
        with cols[i]:
            st.toggle(f"Drill at {label} (profit {econ.site_private_profits[i]:.0f})", key=f"solo_{i}", value=drills[i], help="Toggle your decision for this site.")
            drills[i] = st.session_state[f"solo_{i}"]

    if st.button("Compute Solo Outcome"):
        T, total_profit, per_site = solo_profit(econ, drills)
        with st.container(border=True):
            st.markdown("**Solo Results**")
            st.write(f"Total wells drilled: **{T}**")
            st.write(f"Damage \(D(T)\) = **{econ.damage(T):.2f}**")
            st.write(f"Total private profits (sum of drilled sites) = **{sum(per_site.values()):.2f}**")
            st.write(f"**Your Solo Profit** = **{total_profit:.2f}**")
        st.session_state["solo_done"] = True

# -------------------------------
# Paired Stage UI
# -------------------------------

def paired_stage_ui(class_code: str, econ: EconConfig, user_id: str, name: str):
    stage_header("Stage 2 — Paired Game",
                 "You control your assigned sites. Decisions are simultaneous. Each player pays their share of damage.")

    st.markdown("**Step 1. Join the matchmaking queue**")
    if st.button("Join Queue / Find a Partner"):
        enqueue_student(class_code, user_id, name)
        st.info("You are in the queue. Waiting for a partner...")

    # Try to match (if recently enqueued)
    room_id = st.session_state.get("room_id")
    if room_id is None:
        new_room = try_match(class_code)
        if new_room:
            st.session_state["room_id"] = new_room
            room_id = new_room

    # Polling / show status
    placeholder = st.empty()
    with placeholder.container():
        if room_id is None:
            store = get_store()
            qlen = len(store["queue"].get(class_code, []))
            st.write(f"Queue length in `{class_code}`: **{qlen}** (pairs are formed automatically)")
            st.caption("Keep this page open. It will refresh while you wait.")
            time.sleep(REFRESH_SEC)
            st.rerun()
        else:
            room = get_room(room_id)
            if room is None:
                st.error("Room not found (it might have been closed). Click 'Join Queue' again.")
                if st.button("Re-Queue"):
                    enqueue_student(class_code, user_id, name)
                    st.rerun()
                return

            # Identify whether this user is P1 or P2 (by assignment time)
            role = "P1" if room.p1_id == user_id else ("P2" if room.p2_id == user_id else None)
            if role is None:
                # A third student shouldn't be in this room
                st.warning("You are not part of this room. Please re-queue.")
                if st.button("Re-Queue"):
                    enqueue_student(class_code, user_id, name)
                    st.rerun()
                return

            st.success(f"Matched! Room `{room.room_id}` — You are **{role}**")
            st.caption(f"Partner: {room.p2_name if role=='P1' else room.p1_name}")

            # If room not both ready yet, wait
            if not room.both_ready():
                st.info("Waiting for both players to be fully connected...")
                time.sleep(REFRESH_SEC)
                st.rerun()
                return

            # Round management
            st.markdown(f"**Round {room.round_index} of {room.econ.paired_rounds}**")

            # Build role-specific UI
            idxs = econ.player1_sites if role == "P1" else econ.player2_sites
            other_idxs = econ.player2_sites if role == "P1" else econ.player1_sites

            # Initialize local UI state
            if "paired_drills" not in st.session_state:
                st.session_state["paired_drills"] = {i: False for i in idxs}

            # Show toggles for the player's sites
            st.write("**Your Sites**")
            cols = st.columns(len(idxs))
            for k, i in enumerate(idxs):
                with cols[k]:
                    key = f"paired_{i}"
                    current = st.session_state["paired_drills"].get(i, False)
                    st.toggle(f"{econ.site_labels[i]} (profit {econ.site_private_profits[i]:.0f})",
                              key=key, value=current)
                    st.session_state["paired_drills"][i] = st.session_state[key]

            # Submit decision
            submitted_key = f"submitted_round_{room.room_id}_{room.round_index}"
            if st.session_state.get(submitted_key, False):
                st.info("Decision submitted. Waiting for your partner...")
            else:
                if st.button("Submit Decision"):
                    my_drills = {i: bool(st.session_state["paired_drills"].get(i, False)) for i in idxs}
                    decision = Decision(drills=my_drills, submitted_at=now_ts())
                    if role == "P1":
                        room.p1_decision = decision
                    else:
                        room.p2_decision = decision
                    save_room(room)
                    st.session_state[submitted_key] = True
                    st.rerun()

            # If both submitted, compute outcome
            room = get_room(room_id)  # refresh
            if room.p1_decision and room.p2_decision:
                # Complete decisions for missing sites as False
                p1_drills_full = {i: False for i in range(len(econ.site_labels))}
                p2_drills_full = {i: False for i in range(len(econ.site_labels))}
                p1_drills_full.update(room.p1_decision.drills if room.p1_decision else {})
                p2_drills_full.update(room.p2_decision.drills if room.p2_decision else {})

                T, p1, p2 = paired_profits(econ, p1_drills_full, p2_drills_full)

                res = RoundResult(
                    round_id=room.round_index,
                    T=T,
                    damage=econ.damage(T),
                    p1_profit=p1,
                    p2_profit=p2,
                    p1_drills=p1_drills_full,
                    p2_drills=p2_drills_full,
                    finished_at=now_ts(),
                )
                room.results.append(res)
                room.round_index += 1
                room.reset_round()
                save_room(room)

                # Show results immediately to both
                with st.container(border=True):
                    st.markdown("### Round Result")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Wells (T)", T)
                    c2.metric("Damage D(T)", f"{res.damage:.2f}")
                    c3.metric("P1 Profit", f"{res.p1_profit:.2f}")
                    c4.metric("P2 Profit", f"{res.p2_profit:.2f}")

                    st.markdown("**Decisions**")
                    dd = pd.DataFrame({
                        "Site": econ.site_labels,
                        "P1 Drill": [res.p1_drills[i] for i in range(len(econ.site_labels))],
                        "P2 Drill": [res.p2_drills[i] for i in range(len(econ.site_labels))],
                    })
                    st.dataframe(dd, hide_index=True)

                # If room done, close & archive
                room.close_if_done()
                save_room(room)
                if room.closed:
                    archive_room(room)
                    save_room(room)
                    st.success("Room finished. You can re-queue to play more rounds with a new partner.")
                    if st.button("Play Again (Re-Queue)"):
                        st.session_state.pop("room_id", None)
                        # reset local paired drills
                        st.session_state["paired_drills"] = {i: False for i in idxs}
                        enqueue_student(class_code, user_id, name)
                        st.rerun()
                else:
                    # Prepare next round UI for this user
                    st.session_state.pop(submitted_key, None)
                    # Keep paired_drills toggles as they were (students can tweak next round)

            else:
                st.info("Waiting for both players to submit...")
                time.sleep(REFRESH_SEC)
                st.rerun()

# -------------------------------
# Main App
# -------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Environmental Economics interactive — solo vs. paired drilling with shared external damages.")

    # Minimal identity for pairing
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = gen_user_id()
    if "student_name" not in st.session_state:
        st.session_state["student_name"] = ""

    # Sidebar: instructor & student inputs
    sidebar_instructor_panel()

    # Student identity & class code
    class_code = st.session_state.get("class_code", "ENVECON-101")
    econ = load_econ(class_code)

    with st.container(border=True):
        st.subheader("Join the Activity")
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.text_input("Your display name (nickname is fine)", value=st.session_state.get("student_name", ""))
            st.session_state["student_name"] = name
        with c2:
            st.text_input("Class Code (ask your instructor)", value=class_code, key="class_code")

        if not name.strip():
            st.warning("Please enter a name to proceed.")
            st.stop()

        student_header(st.session_state["class_code"], econ)

    # Stage 1: Solo
    with st.container(border=True):
        solo_stage_ui(econ)

    # Stage 2: Paired
    with st.container(border=True):
        paired_stage_ui(st.session_state["class_code"], econ, st.session_state["user_id"], st.session_state["student_name"])

    # Instructor-facing live monitor
    with st.expander("Instructor Live View (read-only for students)", expanded=False):
        store = get_store()
        rooms = [r for r in store["rooms"].values() if r.class_code == st.session_state["class_code"]]
        if rooms:
            for r in rooms:
                with st.container(border=True):
                    st.markdown(f"**Room {r.room_id}** — {'Closed' if r.closed else 'Open'} | Round {min(r.round_index, r.econ.paired_rounds)}")
                    st.write(f"P1: {r.p1_name} | P2: {r.p2_name}")
                    if r.results:
                        df = pd.DataFrame([{
                            "Round": res.round_id,
                            "T": res.T,
                            "Damage": res.damage,
                            "P1 Profit": res.p1_profit,
                            "P2 Profit": res.p2_profit,
                            "P1 Drills": json.dumps(res.p1_drills),
                            "P2 Drills": json.dumps(res.p2_drills),
                        } for res in r.results])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.caption("No results yet.")
        else:
            st.caption("No rooms yet for this Class Code.")

    st.markdown("---")
    st.markdown(
        """
        ### Notes for Instructors
        - **Match your worksheet:** Use the sidebar → *Payoff Parameters* to set site profits and the damage function so the in-app results exactly match your handout.
        - **Damage function:** The default \(D(T)=10T^2\) creates a strong externality (good for discussion). You can add a linear or constant term.
        - **Cost allocation:** In paired play, each player pays `damage_share_per_player × D(T)`. With two symmetric players, `0.5` splits the social damage evenly.
        - **Replayability:** Students can finish a 3-round match and re-queue to experience different partners/strategies.
        - **Data export:** Instructor Panel → *Export All Results (CSV)* for debrief and grading.
        """
    )

if __name__ == "__main__":
    main()
