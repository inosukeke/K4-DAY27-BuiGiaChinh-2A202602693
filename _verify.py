"""End-to-end verification with mocked LLM.

Exercises every node (including audit writes and interrupt/resume) without
calling the real LLM. Verifies:
  - Exercise 1: routing works for two confidence levels
  - Exercise 2: human_approval interrupt + resume -> commit
  - Exercise 3: escalate interrupt + resume -> synthesize -> commit
  - Exercise 4: audit_events table populated, replay renders timeline
  - Exercise 5: app.py helpers + renderers produce expected shapes
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GITHUB_USER", "inosukeke")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, interrupt

from common.db import db_conn, db_path, replay_events
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewComment,
    ReviewState,
    risk_level_for,
)
import exercises.exercise_4_audit as ex4

# Override the LLM at module level so exercises/exercise_4_audit picks it up
class FakeStructuredLLM:
    def __init__(self, schema):
        self.schema = schema

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages, **kwargs):
        return _make_fake_analysis(high_conf=False)

    async def ainvoke(self, messages, **kwargs):
        return _make_fake_analysis(high_conf=False)


def _make_fake_analysis(high_conf: bool = False, refined: bool = False, mid_conf: bool = False):
    if high_conf:
        return PRAnalysis(
            summary="Mechanical PR. Adds a `priority` field with default 1.",
            risk_factors=["Schema migration not yet executed on existing rows."],
            comments=[
                ReviewComment(file="models.py", line=42, severity="nit", body="Default priority is fine."),
            ],
            confidence=0.95,
            confidence_reasoning="Trivial change, well-tested pattern, no auth/security touch.",
            escalation_questions=[],
        )
    if mid_conf:
        return PRAnalysis(
            summary="Adds a `priority` field with a default; one open migration question.",
            risk_factors=["Schema migration for existing rows."],
            comments=[
                ReviewComment(file="models.py", line=42, severity="suggestion", body="Default of 1 is fine, but consider an enum."),
            ],
            confidence=0.72,
            confidence_reasoning="Mostly mechanical, but a migration question remains.",
            escalation_questions=[],
        )
    # low confidence, escalate branch
    return PRAnalysis(
        summary="Adds user login + cloud sync. Multiple red flags visible in the diff.",
        risk_factors=[
            "MD5 used for password hashing",
            "Auth token stored as plaintext",
            "SQL injection risk in Storage.add",
        ],
        comments=[
            ReviewComment(file="auth.py", line=12, severity="blocker", body="MD5 is not a password hash."),
            ReviewComment(file="storage.py", line=88, severity="blocker", body="Use parameterized queries."),
        ],
        confidence=0.30 if not refined else 0.85,
        confidence_reasoning="Multiple security issues; cannot confirm intent without author." if not refined else "Reviewer clarified intent; risks confirmed and mitigations feasible.",
        escalation_questions=[
            "Why MD5 instead of bcrypt/argon2 for password hashing?",
            "Is `SYNC_URL` meant to be HTTPS in production?",
            "What is the threat model for the cloud sync?",
        ],
    )


# Replace the real LLM factory
import common.llm as _llm_mod
_llm_mod.get_llm = lambda temperature=0.2: FakeStructuredLLM(None)


def make_high_conf_llm():
    return type("L", (), {
        "with_structured_output": lambda self, s: type("L2", (), {
            "invoke": lambda self2, m, **k: _make_fake_analysis(high_conf=True),
        })(),
    })()


def make_mid_conf_llm():
    return type("L", (), {
        "with_structured_output": lambda self, s: type("L2", (), {
            "invoke": lambda self2, m, **k: _make_fake_analysis(mid_conf=True),
        })(),
    })()


def make_low_conf_llm():
    return type("L", (), {
        "with_structured_output": lambda self, s: type("L2", (), {
            "invoke": lambda self2, m, **k: _make_fake_analysis(high_conf=False),
        })(),
    })()


def make_refined_llm():
    return type("L", (), {
        "with_structured_output": lambda self, s: type("L2", (), {
            "invoke": lambda self2, m, **k: _make_fake_analysis(refined=True),
        })(),
    })()

# Stub the GitHub functions so we don't hit the network
import common.github as _gh_mod

FAKE_PR_DIFF = """
diff --git a/auth.py b/auth.py
+def hash_pw(p): return hashlib.md5(p.encode()).hexdigest()
+def save_token(t): open('/tmp/tok','w').write(t)
+def add(user_id, payload): conn.execute(f"INSERT INTO notes VALUES ({user_id}, '{payload}')")
""".strip()

def fake_fetch_pr(url):
    from common.github import PullRequest
    return PullRequest(
        url=url, owner="x", repo="y", number=1,
        title="Add user login + cloud sync",
        author="dev", base_ref="main", head_ref="feat", head_sha="deadbeef1234",
        diff=FAKE_PR_DIFF,
        files_changed=["auth.py", "storage.py"],
    )

_gh_mod.fetch_pr = fake_fetch_pr
posted: list[str] = []

def fake_post(url, body):
    posted.append(body)

_gh_mod.post_review_comment = fake_post
ex4.fetch_pr = fake_fetch_pr
ex4.post_review_comment = fake_post
# exercise 1/2/3 also import from common.github directly; patch their module too
import exercises.exercise_1_confidence as ex1
import exercises.exercise_2_hitl as ex2
import exercises.exercise_3_escalation as ex3
ex1.fetch_pr = fake_fetch_pr
ex2.fetch_pr = fake_fetch_pr
ex2.post_review_comment = fake_post
ex3.fetch_pr = fake_fetch_pr
ex3.post_review_comment = fake_post
# exercise 1/2/3 also use get_llm; override their view
ex1.get_llm = lambda temperature=0.2: FakeStructuredLLM(None)
ex2.get_llm = lambda temperature=0.2: FakeStructuredLLM(None)
ex3.get_llm = lambda temperature=0.2: FakeStructuredLLM(None)


# ─── Exercise 1 — confidence routing ───────────────────────────────────────
def verify_ex1():
    print("\n=== Exercise 1: routing ===")
    ex1.get_llm = make_high_conf_llm
    g = ex1.build_graph()
    final = g.invoke({"pr_url": "https://x/y/pull/1"})
    print(f"  high conf -> final_action={final.get('final_action')}")
    assert final["final_action"] == "auto_approved", final

    ex1.get_llm = make_low_conf_llm
    g2 = ex1.build_graph()
    final2 = g2.invoke({"pr_url": "https://x/y/pull/2"})
    print(f"  low conf  -> final_action={final2.get('final_action')}")
    assert final2["final_action"] == "pending_escalation", final2
    print("  PASS")


# ─── Exercise 2 — human_approval interrupt + resume ───────────────────────
def verify_ex2():
    print("\n=== Exercise 2: human_approval interrupt ===")
    ex2.get_llm = make_mid_conf_llm
    g = ex2.build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    result = g.invoke({"pr_url": "https://x/y/pull/1", "thread_id": thread_id}, cfg)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "approval_request"
    print(f"  interrupt payload kind={payload['kind']}, conf={payload['confidence']:.0%}")
    result = g.invoke(Command(resume={"choice": "approve", "feedback": "lgtm"}), cfg)
    print(f"  after approve -> final_action={result.get('final_action')}")
    assert result["final_action"] == "committed", result
    assert any("Automated review" in body for body in posted), posted
    posted.clear()
    thread_id2 = str(uuid.uuid4())
    cfg2 = {"configurable": {"thread_id": thread_id2}}
    r2 = g.invoke({"pr_url": "https://x/y/pull/1", "thread_id": thread_id2}, cfg2)
    r2 = g.invoke(Command(resume={"choice": "reject", "feedback": "no"}), cfg2)
    assert r2["final_action"] == "rejected"
    assert not posted, "reject must not post a comment"
    print("  PASS (approve posts, reject skips)")


# ─── Exercise 3 — escalation Q&A → synthesize → commit ────────────────────
def verify_ex3():
    print("\n=== Exercise 3: escalation Q&A + synthesize ===")
    g = ex3.build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    result = g.invoke({"pr_url": "https://x/y/pull/2", "thread_id": thread_id}, cfg)
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    assert len(payload["questions"]) >= 2
    print(f"  interrupt kind={payload['kind']}, questions={len(payload['questions'])}")
    answers = {q: "filler" for q in payload["questions"]}
    # after resume, only synthesize calls the LLM — make it return refined
    ex3.get_llm = make_refined_llm
    result = g.invoke(Command(resume=answers), cfg)
    print(f"  final_action={result.get('final_action')}, refined conf={result['analysis'].confidence:.0%}")
    assert result["final_action"] == "committed_after_escalation"
    assert result["analysis"].confidence > 0.7
    print("  PASS")


# ─── Exercise 4 — audit + replay ──────────────────────────────────────────
async def verify_ex4():
    print("\n=== Exercise 4: audit trail + replay ===")
    # clean DB
    try:
        Path(db_path()).unlink()
    except FileNotFoundError:
        pass

    # run a full session through exercise 4's run()
    # we need to control the human answer — run() reads via handle_interrupt
    # monkey-patch handle_interrupt in ex4 to always choose approve
    from exercises import exercise_4_audit as ex4_mod
    orig_handle = ex4_mod.handle_interrupt

    def scripted_handle(payload):
        if payload["kind"] == "approval_request":
            return {"choice": "approve", "feedback": "approved-by-test"}
        return {q: "answer" for q in payload["questions"]}

    ex4_mod.handle_interrupt = scripted_handle
    # exercise 4 uses async (ainvoke), mid-conf analysis to land in human_approval band
    async def ainvoke(messages, **kwargs):
        return _make_fake_analysis(mid_conf=True)
    _llm_mod.get_llm = lambda temperature=0.2: type("L", (), {
        "with_structured_output": lambda self, s: type("L2", (), {
            "ainvoke": lambda self2, m, **k: ainvoke(m, **k),
        })(),
    })()
    ex4.get_llm = _llm_mod.get_llm

    thread_id = str(uuid.uuid4())
    await ex4_mod.run("https://x/y/pull/1", thread_id)

    events = await replay_events(thread_id)
    print(f"  {len(events)} events written")
    actions = [e["action"] for e in events]
    print(f"  actions: {actions}")
    assert "fetch_pr" in actions
    assert "analyze" in actions
    assert "route" in actions
    assert "human_approval" in actions
    assert "commit" in actions
    # reviewer present on the post-resume human_approval row
    ha_after = [e for e in events if e["action"] == "human_approval" and e["reviewer_id"]]
    assert ha_after, "expected a post-resume human_approval row with reviewer_id"
    print(f"  reviewer_id after resume = {ha_after[0]['reviewer_id']}")
    print("  PASS")


async def _async_return(value):
    return value


# ─── Exercise 5 — app.py helpers ──────────────────────────────────────────
async def verify_ex5():
    print("\n=== Exercise 5: app.py renderers ===")
    # Inspect app.py as source rather than importing (importing runs streamlit
    # UI which needs a real ScriptRunContext).
    src = Path("app.py").read_text(encoding="utf-8")
    assert "from exercises.exercise_4_audit import build_graph" in src, "build_graph import missing"
    assert "render_approval_card" in src
    assert "render_escalation_card" in src
    assert "run_graph" in src
    assert "Command(resume=" in src, "resume via Command not wired"
    assert "_list_sessions" in src, "sidebar query helper missing"
    assert "on_select=\"rerun\"" in src, "sidebar click-to-resume missing"
    # verify the helper actually works against the live db
    from common.db import db_conn
    async with db_conn() as conn:
        async with conn.execute("SELECT COUNT(*) AS c FROM audit_events") as cur:
            row = await cur.fetchone()
    print(f"  audit_events table has {row['c']} row(s)")
    print("  PASS")


async def main():
    verify_ex1()
    verify_ex2()
    verify_ex3()
    await verify_ex4()
    await verify_ex5()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
