import json

from wqb.store import ResultStore

METRICS = {"sharpe": 1.9, "fitness": 1.3, "turnover": 0.2, "returns": 0.1}


def make_store(tmp_path):
    return ResultStore(tmp_path / "alphas.db")


def test_records_and_reads_back(tmp_path):
    with make_store(tmp_path) as store:
        store.record(
            expression="rank(-1 * ts_zscore(close, 20))", template="reversion_zscore",
            field="close", window=20, status="COMPLETE", alpha_id="A1",
            metrics=METRICS, passed=True, reason="ok",
        )
        assert store.counts() == {"total": 1, "passed": 1, "submitted": 0}
        winner = store.winners()[0]
        assert winner["alpha_id"] == "A1"
        assert winner["sharpe"] == 1.9
        assert json.loads(winner["metrics_json"])["turnover"] == 0.2


def test_seen_enables_resume(tmp_path):
    with make_store(tmp_path) as store:
        store.record(expression="e1", status="COMPLETE", metrics=METRICS)
        assert store.seen() == {"e1"}


def test_record_is_idempotent_per_expression(tmp_path):
    with make_store(tmp_path) as store:
        store.record(expression="e1", status="COMPLETE", metrics={"sharpe": 0.1})
        store.record(expression="e1", status="COMPLETE", metrics={"sharpe": 2.5}, passed=True)
        assert store.counts()["total"] == 1
        assert store.top()[0]["sharpe"] == 2.5


def test_winners_appended_to_success_file(tmp_path):
    with make_store(tmp_path) as store:
        store.record(expression="e1", status="COMPLETE", metrics=METRICS, passed=True)
        store.record(expression="e2", status="COMPLETE", metrics=METRICS, passed=False)
        lines = store.success_file.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["expression"] == "e1"


def test_top_orders_by_sharpe_descending(tmp_path):
    with make_store(tmp_path) as store:
        for name, sharpe in (("low", 0.5), ("high", 3.0), ("mid", 1.5)):
            store.record(expression=name, status="COMPLETE", metrics={"sharpe": sharpe})
        assert [r["expression"] for r in store.top()] == ["high", "mid", "low"]


def test_submission_tracking(tmp_path):
    with make_store(tmp_path) as store:
        store.record(
            expression="e1", status="COMPLETE", alpha_id="A1", metrics=METRICS, passed=True
        )
        assert [r["alpha_id"] for r in store.pending_submission()] == ["A1"]
        store.mark_submitted("e1")
        assert store.pending_submission() == []
        assert store.counts()["submitted"] == 1


def test_non_numeric_metrics_are_tolerated(tmp_path):
    """A result with missing/garbage metrics must still be recorded, so the
    resume logic does not re-simulate it, even though the Sharpe leaderboard
    (which ranks on sharpe) skips it."""
    with make_store(tmp_path) as store:
        store.record(expression="e1", status="COMPLETE", metrics={"sharpe": None, "fitness": "n/a"})
        assert store.counts()["total"] == 1
        assert store.seen() == {"e1"}
        assert store.top() == []


def test_concurrent_writes_from_many_threads(tmp_path):
    """The automation loop records results from worker threads, so the store
    must tolerate writes off the creating thread."""
    from concurrent.futures import ThreadPoolExecutor

    with make_store(tmp_path) as store:
        def write(i):
            store.record(
                expression=f"expr{i}", status="COMPLETE",
                metrics={"sharpe": i / 10}, passed=(i % 2 == 0),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(60)))

        assert store.counts()["total"] == 60
        assert store.counts()["passed"] == 30
        assert len(store.success_file.read_text().strip().splitlines()) == 30
