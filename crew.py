"""
Crew orchestration for the Multi-Agent AI Academic Assistant.

AcademicAssistantCrew wires together the Tutor, Analytics, and Support agents
into three sequential workflows:

  1. handle_tutoring_request   – personalised Q&A with RAG
  2. handle_support_request    – administrative policy lookup
  3. generate_progress_report  – full analytics + learning plan

Each method returns (result: str, agent_log: str) so the Streamlit UI can
display both the final answer and the raw agent activity log.
"""

import io
import sys
import time
from typing import Tuple

from crewai import Crew, Process, Task

from agents.analytics_agent import create_analytics_agent
from agents.support_agent import create_support_agent
from agents.tutor_agent import create_tutor_agent


def _run_with_retry(crew: Crew, max_retries: int = 3) -> object:
    """Run a crew, retrying on 429 rate-limit errors with exponential back-off."""
    for attempt in range(max_retries):
        try:
            return crew.kickoff()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 45 * (attempt + 1)
                print(f"[Rate limit] Waiting {wait}s before retry {attempt + 1}/{max_retries}…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting.")


class AcademicAssistantCrew:
    """
    Orchestrates the multi-agent collaboration for academic assistance.

    Agents are created once at initialisation and reused across requests
    to avoid the overhead of re-instantiating LLM clients.
    """

    def __init__(self) -> None:
        """Instantiate all three agents."""
        self.tutor = create_tutor_agent()
        self.analytics = create_analytics_agent()
        self.support = create_support_agent()

    # ──────────────────────────────────────────────────────────────────────────
    # Workflow 1 — Personalised Tutoring
    # ──────────────────────────────────────────────────────────────────────────

    def handle_tutoring_request(
        self, student_id: str, question: str
    ) -> Tuple[str, str]:
        """
        Answer a student's academic question with full multi-agent collaboration.

        Workflow (sequential):
          Step 1 — Analytics Agent analyses the student's performance profile.
          Step 2 — Tutor Agent searches course materials and crafts a personalised
                   explanation informed by the analytics output.

        Args:
            student_id: Unique student identifier (e.g. "student_001").
            question:   The academic question to answer.

        Returns:
            Tuple of (answer, agent_log) where agent_log is the raw verbose output
            captured from the CrewAI execution.
        """
        log_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buf

        try:
            # ── Task 1: Analyse the student ──────────────────────────────────
            analyse_task = Task(
                description=(
                    f"Analyse the learning profile for student ID: {student_id}.\n\n"
                    f"Use the performance_analysis tool to retrieve their scores, "
                    f"completion rate, strengths, and weaknesses.\n"
                    f"Then identify:\n"
                    f"  • Their current performance tier (excellent / good / average / struggling)\n"
                    f"  • Which strengths the tutor should build on\n"
                    f"  • Which weaknesses need extra care\n"
                    f"  • The recommended explanation depth for this question: '{question}'"
                ),
                expected_output=(
                    "A concise student profile summary including performance tier, "
                    "relevant strengths/weaknesses, and a recommended explanation depth "
                    "(beginner / intermediate / advanced)."
                ),
                agent=self.analytics,
            )

            # ── Task 2: Search materials + teach ─────────────────────────────
            tutor_task = Task(
                description=(
                    f"Using the student profile from the analytics agent, provide a "
                    f"comprehensive, personalised answer to:\n\n"
                    f"  \"{question}\"\n\n"
                    f"Steps:\n"
                    f"  1. Search uploaded course materials with document_search.\n"
                    f"  2. Craft an explanation at the depth recommended by the analytics agent.\n"
                    f"  3. Use clear analogies, worked examples, and step-by-step breakdowns.\n"
                    f"  4. Cite specific page numbers / source files when quoting materials.\n"
                    f"  5. Close with exactly 3 practice questions that test understanding."
                ),
                expected_output=(
                    "A well-structured, student-level-appropriate explanation that: "
                    "addresses the question thoroughly, references course materials where "
                    "available (with citations), uses relatable analogies, and ends with "
                    "3 targeted practice questions."
                ),
                agent=self.tutor,
                context=[analyse_task],
            )

            crew = Crew(
                agents=[self.analytics, self.tutor],
                tasks=[analyse_task, tutor_task],
                process=Process.sequential,
                verbose=True,
            )
            result = _run_with_retry(crew)

        finally:
            sys.stdout = old_stdout

        text = result.raw if hasattr(result, "raw") else str(result)
        return text, log_buf.getvalue()

    # ──────────────────────────────────────────────────────────────────────────
    # Workflow 2 — Administrative Support
    # ──────────────────────────────────────────────────────────────────────────

    def handle_support_request(self, question: str) -> Tuple[str, str]:
        """
        Answer an administrative or policy question.

        Workflow (sequential):
          Single step — Support Agent searches the policy knowledge base and
          returns a complete, actionable answer.

        Args:
            question: The administrative / policy question.

        Returns:
            Tuple of (answer, agent_log).
        """
        log_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buf

        try:
            support_task = Task(
                description=(
                    f"Answer the following student question:\n\n"
                    f"  \"{question}\"\n\n"
                    f"Steps:\n"
                    f"  1. Use policy_search to retrieve relevant policy information.\n"
                    f"  2. Give a clear, complete answer with specific details.\n"
                    f"  3. Highlight any important deadlines or requirements.\n"
                    f"  4. Mention 1–2 related pieces of information the student might "
                    f"     find useful (even if not asked).\n"
                    f"  5. If further help is needed, direct them to the right office/contact."
                ),
                expected_output=(
                    "A complete and friendly policy answer that directly addresses the "
                    "question, includes specific details and deadlines, surfaces related "
                    "helpful information, and provides a clear next step."
                ),
                agent=self.support,
            )

            crew = Crew(
                agents=[self.support],
                tasks=[support_task],
                process=Process.sequential,
                verbose=True,
            )
            result = _run_with_retry(crew)

        finally:
            sys.stdout = old_stdout

        text = result.raw if hasattr(result, "raw") else str(result)
        return text, log_buf.getvalue()

    # ──────────────────────────────────────────────────────────────────────────
    # Workflow 3 — Progress Report
    # ──────────────────────────────────────────────────────────────────────────

    def generate_progress_report(self, student_id: str) -> Tuple[str, str]:
        """
        Generate a comprehensive progress report with personalised learning plan.

        Workflow (sequential):
          Step 1 — Analytics Agent performs a deep-dive analysis of all metrics.
          Step 2 — Tutor Agent converts the analytics into a concrete study plan.

        Args:
            student_id: Unique student identifier.

        Returns:
            Tuple of (report, agent_log).
        """
        log_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buf

        try:
            # ── Task 1: Deep analytics ────────────────────────────────────────
            analytics_task = Task(
                description=(
                    f"Produce a comprehensive performance report for student: {student_id}.\n\n"
                    f"Use both performance_analysis and progress_tracker tools:\n"
                    f"  • Full score history and trend direction\n"
                    f"  • Completion rate and engagement level\n"
                    f"  • Ranked strengths (top 3) and weaknesses (top 3)\n"
                    f"  • Topics already mastered vs topics needing review\n"
                    f"  • Overall academic standing (at-risk / on-track / excelling)\n"
                    f"  • Predicted end-of-semester outcome if trend continues"
                ),
                expected_output=(
                    "A detailed analytics report with: full performance metrics, "
                    "trend analysis, ranked strengths/weaknesses, mastery map, "
                    "academic standing assessment, and outcome prediction."
                ),
                agent=self.analytics,
            )

            # ── Task 2: Personalised learning plan ───────────────────────────
            plan_task = Task(
                description=(
                    f"Using the analytics report for {student_id}, build a personalised "
                    f"learning plan covering the next 4 weeks.\n\n"
                    f"Include:\n"
                    f"  1. Top 3 priority topics to address immediately (with rationale)\n"
                    f"  2. Specific study strategies for each weak area\n"
                    f"  3. A weekly study schedule template (Mon–Sun)\n"
                    f"  4. Short-term SMART goals (2-week horizon)\n"
                    f"  5. Long-term SMART goals (end-of-semester)\n"
                    f"  6. Motivational insights that connect existing strengths to "
                    f"     the areas that need growth"
                ),
                expected_output=(
                    "A student-ready 4-week learning plan with: prioritised topics, "
                    "study strategies, weekly schedule template, SMART goals "
                    "(short- and long-term), and motivational framing."
                ),
                agent=self.tutor,
                context=[analytics_task],
            )

            crew = Crew(
                agents=[self.analytics, self.tutor],
                tasks=[analytics_task, plan_task],
                process=Process.sequential,
                verbose=True,
            )
            result = _run_with_retry(crew)

        finally:
            sys.stdout = old_stdout

        text = result.raw if hasattr(result, "raw") else str(result)
        return text, log_buf.getvalue()
