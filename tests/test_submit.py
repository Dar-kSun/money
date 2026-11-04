import responses

from wqb.submit import check_submission, submit_alpha

BASE = "https://api.worldquantbrain.com"

PASSING = {"is": {"checks": [
    {"name": "LOW_SHARPE", "result": "PASS"},
    {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
]}}
FAILING = {"is": {"checks": [
    {"name": "LOW_SHARPE", "result": "PASS"},
    {"name": "SELF_CORRELATION", "result": "FAIL", "message": "corr 0.8 with ALPHA7"},
]}}


def auth():
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)


@responses.activate
def test_dry_run_never_posts_to_submit(client, no_sleep):
    """The default path must not perform the irreversible submission."""
    auth()
    responses.add(responses.GET, f"{BASE}/alphas/A1/check", json=PASSING, status=200)

    outcome = submit_alpha(client, "A1")  # confirm defaults to False
    assert outcome.submitted is False
    assert outcome.status == "DRY_RUN(PASS)"
    assert "--yes" in outcome.message
    assert not [c for c in responses.calls if c.request.url.endswith("/submit")]


@responses.activate
def test_confirmed_submission_posts_and_reports_success(client, no_sleep):
    auth()
    responses.add(responses.GET, f"{BASE}/alphas/A1/check", json=PASSING, status=200)
    responses.add(responses.POST, f"{BASE}/alphas/A1/submit", status=201)
    responses.add(responses.GET, f"{BASE}/alphas/A1/submit", json=PASSING, status=200)

    outcome = submit_alpha(client, "A1", confirm=True)
    assert outcome.submitted is True
    assert outcome.status == "SUBMITTED"
    assert "SUBMITTED" in outcome.describe()


@responses.activate
def test_failing_precheck_blocks_submission(client, no_sleep):
    """A submission slot must not be spent on an alpha Brain will reject."""
    auth()
    responses.add(responses.GET, f"{BASE}/alphas/A1/check", json=FAILING, status=200)

    outcome = submit_alpha(client, "A1", confirm=True)
    assert outcome.submitted is False
    assert outcome.failed_checks == ["SELF_CORRELATION"]
    assert not [c for c in responses.calls if c.request.method == "POST"
                and c.request.url.endswith("/submit")]


@responses.activate
def test_check_waits_for_pending_checks(client, no_sleep):
    auth()
    responses.add(
        responses.GET, f"{BASE}/alphas/A1/check", status=200,
        headers={"Retry-After": "4"}, body="",
    )
    responses.add(responses.GET, f"{BASE}/alphas/A1/check", json=PASSING, status=200)

    outcome = check_submission(client, "A1")
    assert outcome.status == "PASS"
    assert 4.0 in no_sleep


@responses.activate
def test_not_eligible_is_reported_cleanly(client, no_sleep):
    auth()
    responses.add(responses.GET, f"{BASE}/alphas/A1/check", status=403)
    outcome = check_submission(client, "A1")
    assert outcome.status == "NOT_ELIGIBLE"
    assert not outcome.submitted


@responses.activate
def test_check_timeout_is_not_a_submission(client, no_sleep):
    auth()
    responses.add(
        responses.GET, f"{BASE}/alphas/A1/check", status=200,
        headers={"Retry-After": "1"}, body="",
    )
    outcome = check_submission(client, "A1", timeout=-1)
    assert outcome.status == "TIMEOUT"
    assert not outcome.submitted
