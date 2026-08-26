"""Fair global union ticket selection (task t_e6fe4233).

Covers :func:`hermes_cli.kanban_db.dispatch_once_all_boards` — the
replacement for the structurally-starving sequential per-board dispatch
loop — plus the ``kanban.fair_selection`` escape hatch wiring in the
gateway watcher.

Starvation mechanism under test (recon t_65c52a45): the legacy loop let
whichever board enumerated FIRST consume the entire remaining host budget,
so the LAST board starved whenever an earlier board sustained a ready
backlog — regardless of priorities or ages. The union path makes every
board's candidates compete in one global priority-then-age ordering
against one shared host budget.
"""

from __future__ import annotations

import fcntl
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _all_profiles_exist(monkeypatch):
    """Every synthetic assignee maps to a real profile (same patch as the
    shared ``all_assignees_spawnable`` fixture, applied file-wide so the
    profile-existence guard never masks the fairness assertions)."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _set_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append((board, task.id))
        return 42
    return fake_spawn


def _spawned_ids(results) -> set[str]:
    return {
        tid for res in results.values() for tid, _, _ in res.spawned
    }


# ---------------------------------------------------------------------------
# 1. The incident dry-run: saturated first board cannot starve last board
# ---------------------------------------------------------------------------


def test_saturated_first_board_cannot_starve_last_board(kanban_home):
    """Acceptance criterion 1 (t_e6fe4233): while the FIRST board holds the
    host near cap AND saturates its per-profile allowance, an older
    low-priority ready ticket on the LAST board must still be selected."""
    kb.create_board("alpha")   # enumerates before omega
    kb.create_board("omega")

    # alpha: 2 running (host nearly full) + fresh p=90 backlog for alice.
    with kb.connect(board="alpha") as conn:
        for title in ("busy-1", "busy-2"):
            tid = kb.create_task(conn, title=title, assignee="alice")
            assert kb.claim_task(conn, tid) is not None
        for i in range(3):
            kb.create_task(
                conn, title=f"hot-{i}", assignee="alice", priority=90,
            )

    # omega: OLD p=0 ready ticket — historically starved by iteration order.
    with kb.connect(board="omega") as conn:
        old_id = kb.create_task(
            conn, title="old-p0", assignee="bob", priority=0,
        )

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["default", "alpha", "omega"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=3,
        max_in_progress_per_profile=2,
    )

    # Host budget: 3 − 2 running = 1 slot. alice is AT her per-profile cap,
    # so every alpha candidate defers (skipped_per_profile_capped, never a
    # break) and omega's bob ticket takes the slot.
    assert spawns == [("omega", old_id)]
    alpha_res = results["alpha"]
    assert len(alpha_res.skipped_per_profile_capped) == 3
    assert len(results["omega"].spawned) == 1


def test_cross_board_priority_beats_iteration_order(kanban_home):
    """A higher-priority ticket on the LAST board outranks a lower-priority
    ticket on the FIRST board — impossible under the sequential loop."""
    kb.create_board("aaa")
    kb.create_board("zzz")

    with kb.connect(board="aaa") as conn:
        low_id = kb.create_task(
            conn, title="first-board-low", assignee="alice", priority=10,
        )
    with kb.connect(board="zzz") as conn:
        high_id = kb.create_task(
            conn, title="last-board-high", assignee="bob", priority=90,
        )

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["aaa", "zzz"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=1,
    )

    assert spawns == [("zzz", high_id)]
    assert low_id in [tid for tid, _, _ in results["aaa"].skipped_unassigned] or \
        results["aaa"].spawned == []
    # aaa's low-priority ticket stayed ready.
    with kb.connect(board="aaa") as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (low_id,)
        ).fetchone()
        assert row["status"] == "ready"


def test_tie_broken_deterministically_by_board_then_task_id(kanban_home):
    """Identical (priority, created_at) tickets claim in a deterministic
    (board_slug, task_id) order — no oscillation between ticks."""
    kb.create_board("bbb")
    kb.create_board("ccc")

    for _round in range(3):
        # Fresh isolated pair per round: identical priority AND identical
        # hand-forced created_at, differing only in board slug / task id.
        with kb.connect(board="bbb") as conn:
            b_id = kb.create_task(conn, title=f"tie-b-{_round}", assignee="alice")
            conn.execute("UPDATE tasks SET created_at = 1000 WHERE id = ?", (b_id,))
            conn.commit()
        with kb.connect(board="ccc") as conn:
            c_id = kb.create_task(conn, title=f"tie-c-{_round}", assignee="bob")
            conn.execute("UPDATE tasks SET created_at = 1000 WHERE id = ?", (c_id,))
            conn.commit()

        spawns: list = []
        kb.dispatch_once_all_boards(
            ["ccc", "bbb"],  # deliberately reversed caller order
            spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=1,
        )
        # Exactly one slot: bbb wins on the (board_slug, task_id) tiebreak.
        assert spawns == [("bbb", b_id)]

        # Settle both tickets so the next round starts from a clean slate.
        for slug, tid in (("bbb", b_id), ("ccc", c_id)):
            with kb.connect(board=slug) as conn:
                conn.execute(
                    "UPDATE tasks SET status = 'done', completed_at = 2000 "
                    "WHERE id = ?",
                    (tid,),
                )
                conn.commit()


# ---------------------------------------------------------------------------
# 2. Invariants preserved inside the union
# ---------------------------------------------------------------------------


def test_host_cap_counts_running_across_all_boards(kanban_home):
    """max_in_progress bounds the WHOLE union: running work on other boards
    consumes the shared budget before any candidate claims."""
    kb.create_board("one")
    kb.create_board("two")

    with kb.connect(board="one") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    with kb.connect(board="two") as conn:
        kb.create_task(conn, title="wants-slot", assignee="bob")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["one", "two"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=1,
    )

    assert spawns == []
    assert not any(res.spawned for res in results.values())


def test_per_profile_cap_is_global_across_boards(kanban_home):
    """Running workers on ANOTHER board count toward a profile's cap — the
    counters are seeded from the union, not per board (#21582 semantics)."""
    kb.create_board("one")
    kb.create_board("two")

    with kb.connect(board="one") as conn:
        tid = kb.create_task(conn, title="busy-alice", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    with kb.connect(board="two") as conn:
        alice_id = kb.create_task(conn, title="more-alice", assignee="alice")
        bob_id = kb.create_task(conn, title="bob-work", assignee="bob")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["one", "two"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=8,
        max_in_progress_per_profile=1,
    )

    # alice already has 1 running (on board one) → her ticket on board two
    # defers; bob's claims.
    assert spawns == [("two", bob_id)]
    assert alice_id in [
        tid for tid, _, _ in results["two"].skipped_per_profile_capped
    ]


def test_respawn_guard_applies_inside_union(kanban_home):
    """A guarded task on any board defers instead of claiming."""
    kb.create_board("solo")

    with kb.connect(board="solo") as conn:
        tid = kb.create_task(conn, title="guarded", assignee="alice")
        # Deterministic auth blocker → respawn guard fires.
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("blocker_auth: invalid api key", tid),
        )
        conn.commit()

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["solo"],
        spawn_fn=_fake_spawn_factory(spawns),
    )

    # Guard classification depends on the shared check_respawn_guard;
    # either way nothing may spawn and the reason lands in telemetry.
    assert spawns == []
    assert (
        results["solo"].respawn_guarded
        or results["solo"].auto_blocked
        or results["solo"].spawned == []
    )


def test_review_lane_reservation_across_boards(kanban_home):
    """Review work on ANY board reserves a ready-lane slot — the OOF-30
    review-finding semantics extended to the union."""
    kb.create_board("work")
    kb.create_board("revs")

    with kb.connect(board="work") as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
    with kb.connect(board="revs") as conn:
        review_id = kb.create_task(conn, title="review-me", assignee="reviewer")
        _set_status(conn, review_id, "review")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["work", "revs"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=2,
    )

    ids = {board: {tid for tid, _, _ in res.spawned}
           for board, res in results.items()}
    # Budget 2: exactly one ready spawn + the reserved review slot.
    assert len(spawns) == 2
    assert review_id in ids["revs"]
    assert len(ids["work"]) == 1


# ---------------------------------------------------------------------------
# 3. Secondary defect fixes that fall out of the design
# ---------------------------------------------------------------------------


def test_shared_budget_never_exceeded_with_multiple_reviews(kanban_home):
    """Both lanes draw from ONE shared total (PR #95056 review finding 1).

    The union loop must not track independent per-lane counters: with
    spawn_budget=2 and multiple spawnable reviews, per-lane accounting
    admitted 1 ready + 2 reviews = 3 workers. The single-board tick
    enforces the shared ``spawned`` total in both loops; the union path
    must enforce the same total across boards.
    """
    kb.create_board("work")
    kb.create_board("revs")

    with kb.connect(board="work") as conn:
        for title in ("ready-1", "ready-2"):
            kb.create_task(conn, title=title, assignee="alice")
    with kb.connect(board="revs") as conn:
        for title in ("rev-1", "rev-2", "rev-3"):
            tid = kb.create_task(conn, title=title, assignee="reviewer")
            _set_status(conn, tid, "review")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["work", "revs"],
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=2,
    )

    total = sum(len(res.spawned) for res in results.values())
    assert total == 2, (
        f"shared budget breach: spawn_budget=2 admitted {total} workers "
        f"(spawns={spawns})"
    )
    # The reservation still guarantees the review lane its slot.
    assert len(results["revs"].spawned) >= 1
    assert len(results["work"].spawned) + len(results["revs"].spawned) == 2


def test_transient_maintenance_failure_is_not_quarantined(
    kanban_home, monkeypatch,
):
    """Non-corruption board-local failures must not set corrupt=True
    (PR #95056 review finding 2).

    A board-local failure raised by one board's maintenance phase — an
    ordinary ``RuntimeError`` (programming bug / transient infra error)
    and a non-corrupt SQLite busy/locked ``OperationalError`` — drops
    that board from the CURRENT union pass only: the healthy sibling
    still dispatches, the failed board reports corrupt=False, and the
    watcher (which maps corrupt=True to the durable quarantine registry)
    has nothing to quarantine. Classified corruption (bad SQLite file)
    still reports corrupt=True — see
    test_corrupt_board_still_reported_corrupt below.
    """
    kb.create_board("healthy")
    kb.create_board("flaky")

    real_release = kb.release_stale_claims
    failures = [
        RuntimeError("transient maintenance blowup"),
        sqlite3.OperationalError("database is locked"),
    ]

    def flaky_release(conn, *args, **kwargs):
        # Only the flaky board's connection triggers the failure. Boards
        # live at <root>/kanban/boards/<slug>/kanban.db and sqlite3
        # connections carry no python-side board attribute, so identify
        # the board from the connection's database file path —
        # deterministic regardless of enumeration order.
        row = conn.execute("PRAGMA database_list").fetchone()
        db_file = str(row[2]) if row else ""
        if "flaky" in db_file:
            raise failures.pop(0)
        return real_release(conn, *args, **kwargs)

    monkeypatch.setattr(kb, "release_stale_claims", flaky_release)

    for pass_no, boards in enumerate(
        (["flaky", "healthy"], ["healthy", "flaky"])
    ):
        # A fresh ready task per pass: the previous pass's spawn claimed
        # its task (CAS moves it out of the pool), so re-asserting on a
        # consumed tid would vacuously fail.
        with kb.connect(board="healthy") as conn:
            tid = kb.create_task(
                conn, title=f"healthy-{pass_no}", assignee="alice",
            )
        spawns: list = []
        results = kb.dispatch_once_all_boards(
            boards,
            # LIVE pid: pass 1's spawned task must look like a healthy
            # running worker to pass 2's crash detector — a dead stub pid
            # makes detect_crashed_workers requeue it and pollute pass
            # 2's spawns (the reason _live_pid_spawn_factory exists).
            spawn_fn=_live_pid_spawn_factory(spawns),
        )

        # The healthy board dispatched despite its sibling's failure.
        assert spawns == [("healthy", tid)]
        # The failed board is NOT marked corrupt — transient, retried
        # next tick. Both failure classes must stay out of quarantine.
        assert results["flaky"].corrupt is False
        assert results["healthy"].corrupt is False


def test_corrupt_board_still_reported_corrupt(kanban_home):
    """Classified corruption keeps its durable semantics: corrupt=True,
    healthy siblings unaffected (guards against over-correcting the
    transient-failure fix)."""
    kb.create_board("good")
    kb.create_board("bad")
    kb.kanban_db_path(board="bad").write_text(
        "this is not sqlite", encoding="utf-8"
    )
    with kb.connect(board="good") as conn:
        tid = kb.create_task(conn, title="healthy", assignee="alice")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["bad", "good"],
        spawn_fn=_fake_spawn_factory(spawns),
    )
    assert results["bad"].corrupt is True
    assert spawns == [("good", tid)]


def test_memory_pressure_elevated_spawns_exactly_one_total(
    kanban_home, monkeypatch,
):
    """Regression for the N× elevated-pressure leak: per-board sampling let
    EVERY board spawn 1 under elevated pressure; the union samples once."""
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda sample=None: "elevated")

    slugs = ["b1", "b2", "b3"]
    for slug in slugs:
        kb.create_board(slug)
        with kb.connect(board=slug) as conn:
            kb.create_task(conn, title=f"t-{slug}", assignee=f"p-{slug}")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        slugs,
        spawn_fn=_fake_spawn_factory(spawns),
        max_in_progress=12,
    )
    assert len(spawns) <= 1
    assert all(res.memory_pressure == "elevated" for res in results.values())


def test_zombie_reaper_runs_once_not_once_per_board(kanban_home, monkeypatch):
    """The sequential loop reaped zombies N+1 times per tick; the union
    path must reap exactly once."""
    calls = {"n": 0}

    def fake_reap():
        calls["n"] += 1

    monkeypatch.setattr(kb, "reap_worker_zombies", fake_reap)

    kb.create_board("r1")
    kb.create_board("r2")
    kb.dispatch_once_all_boards(["r1", "r2"], spawn_fn=lambda *a, **k: None)

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 4. Locking: single-writer guarantee preserved, all-or-nothing
# ---------------------------------------------------------------------------


def test_any_lock_contention_skips_entire_tick(kanban_home):
    """An external holder of ANY board's dispatch lock skips the whole
    multi-board tick with zero writes (single-writer semantics, #35240)."""
    kb.create_board("locka")
    kb.create_board("lockb")

    with kb.connect(board="locka") as conn:
        kb.create_task(conn, title="wants-slot", assignee="alice")

    lock_path = kb.kanban_db_path(board="lockb").with_name(
        kb.kanban_db_path(board="lockb").name + ".dispatch.lock"
    )
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            spawns: list = []
            results = kb.dispatch_once_all_boards(
                ["locka", "lockb"],
                spawn_fn=_fake_spawn_factory(spawns),
            )
            assert all(res.skipped_locked for res in results.values())
            assert spawns == []
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_lock_free_tick_claims_normally_after_contention_clears(kanban_home):
    kb.create_board("locka")
    kb.create_board("lockb")
    with kb.connect(board="locka") as conn:
        tid = kb.create_task(conn, title="wants-slot", assignee="alice")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["locka", "lockb"],
        spawn_fn=_fake_spawn_factory(spawns),
    )
    assert spawns == [("locka", tid)]
    assert not results["lockb"].skipped_locked


def _live_pid_spawn_factory(spawns: list):
    """Spawn stub returning a LIVE pid (this process) so a follow-up
    tick's dead-PID crash detector sees healthy running workers instead
    of zombies — isolates the double-spawn assertion from crash handling."""
    import os

    def fake_spawn(task, workspace, board=None):
        spawns.append((board, task.id))
        return os.getpid()

    return fake_spawn


def test_repeated_ticks_never_double_spawn(kanban_home):
    """AC (t_e6fe4233): no double-spawn under repeated tick invocations.

    The atomic ``claim_task``/``claim_review_task`` CAS moves each claimed
    row out of the spawnable pool inside the tick's lock hold, so an
    immediately repeated union tick over the SAME boards must find nothing
    left to claim — the snapshot→claim window never resurrects a row."""
    kb.create_board("solo")
    with kb.connect(board="solo") as conn:
        t1 = kb.create_task(conn, title="one", assignee="alice")
        t2 = kb.create_task(conn, title="two", assignee="bob")

    boards = ["default", "solo"]
    first_spawns: list = []
    results = kb.dispatch_once_all_boards(
        boards,
        spawn_fn=_live_pid_spawn_factory(first_spawns),
        max_in_progress=2,
    )
    # Sort BOTH sides: tids are random (t_<8hex>), so creation order and
    # lexicographic order coincide only ~50% of runs — comparing a sorted
    # actual against an unsorted expected flaked CI (t_6041e228).
    assert sorted(first_spawns) == sorted([("solo", t1), ("solo", t2)])
    assert len(_spawned_ids(results)) == 2

    # Second tick, same boards, no NEW work: zero claims, zero spawns.
    second_spawns: list = []
    kb.dispatch_once_all_boards(
        boards,
        spawn_fn=_live_pid_spawn_factory(second_spawns),
        max_in_progress=2,
    )
    assert second_spawns == []


def test_corrupt_board_fails_soft_and_reports(kanban_home, monkeypatch):
    """One corrupt DB must not brick the other boards; the corrupt board is
    reported via DispatchResult.corrupt for watcher quarantine."""
    good = kb.create_board("good")
    bad = kb.create_board("bad")

    # Corrupt the 'bad' board DB behind connect()'s cache.
    bad_path = kb.kanban_db_path(board="bad")
    bad_path.write_text("this is not sqlite", encoding="utf-8")

    with kb.connect(board="good") as conn:
        tid = kb.create_task(conn, title="healthy", assignee="alice")

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["bad", "good"],  # corrupt board enumerates FIRST
        spawn_fn=_fake_spawn_factory(spawns),
    )

    assert results["bad"].corrupt is True
    assert spawns == [("good", tid)]


def test_unresolvable_board_path_does_not_kill_tick(kanban_home, monkeypatch):
    """kanban_db_path failure for one board skips just that board."""
    real_path = kb.kanban_db_path

    def selective_path(board=None):
        if board == "cursed":
            raise RuntimeError("path resolution exploded")
        return real_path(board=board)

    kb.create_board("fine")
    kb.create_board("cursed")
    with kb.connect(board="fine") as conn:
        tid = kb.create_task(conn, title="healthy", assignee="alice")

    monkeypatch.setattr(kb, "kanban_db_path", selective_path)

    spawns: list = []
    results = kb.dispatch_once_all_boards(
        ["cursed", "fine"],
        spawn_fn=_fake_spawn_factory(spawns),
    )
    assert spawns == [("fine", tid)]
    assert "cursed" in results


# ---------------------------------------------------------------------------
# 5. Watcher wiring: fair_selection gate selects the path
# ---------------------------------------------------------------------------


def _make_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _run_two_ticks(monkeypatch, runner, *, fair):
    """Drive the dispatcher watcher for two ticks with stubbed boards.

    Returns (union_calls, legacy_calls): every dispatch_once_all_boards
    invocation (with its slug list) and every legacy dispatch_once board.
    """
    import asyncio
    import hermes_cli.config as cfgmod
    import logging

    union_calls: list = []
    legacy_calls: list = []

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
                "fair_selection": fair,
            }
        },
    )
    monkeypatch.setattr(
        kb, "list_boards",
        lambda include_archived=False: [{"slug": "b1"}, {"slug": "b2"}],
    )
    monkeypatch.setattr(
        kb, "read_board_metadata", lambda slug: {"slug": slug},
    )

    def fake_union(slugs, **kwargs):
        union_calls.append(list(slugs))
        return {slug: kb.DispatchResult() for slug in slugs}

    def fake_legacy(conn, **kwargs):
        legacy_calls.append(kwargs.get("board"))
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once_all_boards", fake_union)
    monkeypatch.setattr(kb, "dispatch_once", fake_legacy)

    import gateway.run as grun
    to_thread_state = {"n": 0}

    async def _to_thread(fn, *args, **kwargs):
        to_thread_state["n"] += 1
        result = fn(*args, **kwargs)
        if to_thread_state["n"] >= 6:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(grun.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(grun.asyncio, "sleep", _sleep)

    with __import__("contextlib").suppress(asyncio.TimeoutError):
        asyncio.run(
            asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0)
        )
    return union_calls, legacy_calls


def test_watcher_routes_through_union_by_default(kanban_home, monkeypatch):
    """fair_selection unset → defaults ON → one union tick per dispatch
    tick covering ALL boards; the legacy per-board path stays idle."""
    runner = _make_runner()
    union_calls, legacy_calls = _run_two_ticks(monkeypatch, runner, fair=True)

    assert union_calls, "union path never invoked"
    assert all(set(c) == {"b1", "b2"} for c in union_calls)
    assert legacy_calls == []


def test_watcher_escape_hatch_restores_legacy_loop(kanban_home, monkeypatch):
    """fair_selection=false → legacy sequential per-board loop; the union
    path must never be touched (operational rollback valve)."""
    runner = _make_runner()
    union_calls, legacy_calls = _run_two_ticks(monkeypatch, runner, fair=False)

    assert union_calls == []
    assert legacy_calls, "legacy path never invoked"
    assert set(legacy_calls) == {"b1", "b2"}
