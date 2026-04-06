"""
Analytics tools for tracking and analysing student academic performance.
Reads/writes JSON files in data/student_data/.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

STUDENT_DATA_DIR = Path("data/student_data")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_student(student_id: str) -> Optional[dict]:
    """Load a student's JSON file, returning None if not found."""
    STUDENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = STUDENT_DATA_DIR / f"{student_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _recommendations(avg: float, completion: float, weaknesses: list, trend: str) -> str:
    """Generate bullet-point study recommendations."""
    recs: list[str] = []

    if avg < 70:
        recs += [
            "• Schedule weekly tutoring sessions to reinforce fundamentals.",
            "• Review basic concepts before moving to advanced material.",
        ]
    elif avg < 80:
        recs += [
            "• Focus targeted practice on your weakest topics.",
            "• Work through additional practice problems after each lesson.",
        ]
    else:
        recs += [
            "• Explore advanced or enrichment material to deepen mastery.",
            "• Consider mentoring a peer — teaching reinforces your own learning.",
        ]

    if completion < 75:
        recs += [
            "• Aim for 90 %+ completion; each missed assignment hurts your average.",
            "• Try time-blocking: reserve fixed slots for coursework every day.",
        ]

    if "Time management" in weaknesses:
        recs += [
            "• Use the Pomodoro technique (25 min study / 5 min break).",
            "• Build a weekly planner and review it each Sunday evening.",
        ]

    if "Essay writing" in weaknesses:
        recs += [
            "• Visit the Writing Center for one-on-one essay feedback.",
            "• Practice freewriting for 10 minutes daily to build fluency.",
        ]

    if trend == "declining":
        recs += [
            "• Meet with your academic advisor to identify root causes.",
            "• Diagnose whether the drop is from comprehension or external stress.",
        ]

    if not recs:
        recs += [
            "• Keep up the excellent work — consistency is key!",
            "• Set a stretch goal for the next assignment to stay challenged.",
        ]

    return "\n".join(recs)


# ─── Performance Analysis Tool ────────────────────────────────────────────────


class PerformanceAnalysisInput(BaseModel):
    """Input schema for the performance analysis tool."""

    student_id: str = Field(description="The student ID to analyse (e.g. 'student_001')")


class PerformanceAnalysisTool(BaseTool):
    """
    Analyses a student's academic performance using their stored JSON data.

    Returns a formatted report including average score, completion rate,
    score trend, strengths/weaknesses, and personalised recommendations.
    """

    name: str = "performance_analysis"
    description: str = (
        "Analyse a student's academic performance — scores, completion rate, "
        "strengths, weaknesses, trends, and personalised recommendations. "
        "Input: student ID string."
    )
    args_schema: Type[BaseModel] = PerformanceAnalysisInput

    def _run(self, student_id: str) -> str:
        data = _load_student(student_id)
        if not data:
            return (
                f"No data found for student '{student_id}'. "
                "Verify the student ID and ensure their profile exists."
            )

        scores: list[float] = data.get("scores", [])
        avg = sum(scores) / len(scores) if scores else 0.0
        completion = data.get("completion_rate", 0)
        strengths = data.get("strengths", [])
        weaknesses = data.get("weaknesses", [])
        topics = data.get("topics_covered", [])

        # Performance tier
        if avg >= 90:
            tier, icon = "Excellent", "🌟"
        elif avg >= 80:
            tier, icon = "Good", "✅"
        elif avg >= 70:
            tier, icon = "Average", "📊"
        else:
            tier, icon = "Needs Improvement", "⚠️"

        # Score trend (compare last-3 vs rest)
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:-3]) / len(scores[:-3]) if scores[:-3] else recent_avg
            trend = "improving 📈" if recent_avg >= older_avg else "declining 📉"
        else:
            trend = "insufficient data"

        score_history = " → ".join(str(s) for s in scores)
        recs = _recommendations(avg, completion, weaknesses, trend.split()[0])

        return f"""
📊 PERFORMANCE ANALYSIS — {data.get("name", student_id).upper()}
{"=" * 55}

{icon} Overall Performance : {tier}
📈 Average Score      : {avg:.1f}%
✅ Completion Rate    : {completion}%
📚 Topics Covered     : {len(topics)}  ({", ".join(topics) or "none"})
📉 Score Trend        : {trend}

STRENGTHS:
{chr(10).join(f"  ✓ {s}" for s in strengths) or "  (no data)"}

AREAS FOR IMPROVEMENT:
{chr(10).join(f"  ⚠ {w}" for w in weaknesses) or "  (none identified)"}

SCORE HISTORY:
  {score_history or "(no scores yet)"}

RECOMMENDATIONS:
{recs}
""".strip()



# ─── Progress Tracker Tool ────────────────────────────────────────────────────


class ProgressTrackerInput(BaseModel):
    """Input schema for the progress tracker tool."""

    student_id: str = Field(description="Student ID to retrieve progress for")
    action: str = Field(
        default="get",
        description=(
            "Action: 'get' = current snapshot, "
            "'summary' = one-line overview, "
            "'history' = full score history"
        ),
    )


class ProgressTrackerTool(BaseTool):
    """
    Retrieves a student's learning progress in different levels of detail.

    Supports three views:
    - ``get``     – current progress snapshot
    - ``summary`` – one-line overview (useful as context for other agents)
    - ``history`` – full score timeline with all metadata
    """

    name: str = "progress_tracker"
    description: str = (
        "Track and retrieve student learning progress — topics covered, score history, "
        "completion milestones. "
        "Actions: 'get' (current snapshot), 'summary' (brief overview), 'history' (full log). "
        "Input: student ID and action."
    )
    args_schema: Type[BaseModel] = ProgressTrackerInput

    def _run(self, student_id: str, action: str = "get") -> str:
        data = _load_student(student_id)
        if not data:
            return f"No progress data found for student '{student_id}'."

        if action == "summary":
            return self._summary(data)
        if action == "history":
            return self._history(data)
        return self._current(data)

    # ── view helpers ──────────────────────────────────────────────────────────

    def _current(self, data: dict) -> str:
        scores = data.get("scores", [])
        avg = sum(scores) / len(scores) if scores else 0.0
        last = scores[-1] if scores else "N/A"
        return (
            f"CURRENT PROGRESS — {data.get('name', data['student_id'])}\n"
            f"Last Score   : {last}%\n"
            f"Average      : {avg:.1f}%\n"
            f"Completion   : {data.get('completion_rate', 0)}%\n"
            f"Topics       : {', '.join(data.get('topics_covered', []))}\n"
            f"Last Updated : {data.get('last_updated', 'N/A')}"
        )

    def _summary(self, data: dict) -> str:
        scores = data.get("scores", [])
        avg = sum(scores) / len(scores) if scores else 0.0
        return (
            f"{data.get('name', data['student_id'])}: "
            f"avg {avg:.1f}%, "
            f"{data.get('completion_rate', 0)}% complete, "
            f"{len(data.get('topics_covered', []))} topics covered."
        )

    def _history(self, data: dict) -> str:
        scores = data.get("scores", [])
        rows = "\n".join(f"  Assignment {i+1}: {s}%" for i, s in enumerate(scores))
        return (
            f"PROGRESS HISTORY — {data.get('name', data['student_id'])}\n"
            f"{'─' * 42}\n"
            f"Score History:\n{rows or '  (none)'}\n\n"
            f"Topics   : {', '.join(data.get('topics_covered', []))}\n"
            f"Strengths: {', '.join(data.get('strengths', []))}\n"
            f"Weaknesses: {', '.join(data.get('weaknesses', []))}"
        )



# ─── Utility: update student data ─────────────────────────────────────────────


def update_student_progress(
    student_id: str,
    score: Optional[float] = None,
    topic: Optional[str] = None,
    completion_rate: Optional[float] = None,
) -> bool:
    """
    Append a new score or topic to a student's profile.

    Creates the profile file if it does not yet exist.

    Args:
        student_id:      Unique student identifier.
        score:           New score to append (0–100).
        topic:           New topic to add (skipped if already present).
        completion_rate: Updated completion percentage.

    Returns:
        True on success, False on error.
    """
    try:
        STUDENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = STUDENT_DATA_DIR / f"{student_id}.json"

        data = _load_student(student_id) or {
            "student_id": student_id,
            "name": f"Student {student_id}",
            "scores": [],
            "completion_rate": 0,
            "strengths": [],
            "weaknesses": [],
            "topics_covered": [],
        }

        if score is not None:
            data["scores"].append(float(score))
        if topic and topic not in data["topics_covered"]:
            data["topics_covered"].append(topic)
        if completion_rate is not None:
            data["completion_rate"] = float(completion_rate)

        data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as exc:
        print(f"[update_student_progress] Error for {student_id}: {exc}")
        return False
