"""
Support Agent — administrative and policy questions.

Uses GPT-3.5-turbo (fast and cost-efficient for lookup tasks) with
a PolicySearchTool that covers admissions, fees, schedules, and services.
"""

from crewai import Agent, LLM

from tools.document_tools import PolicySearchTool


def create_support_agent() -> Agent:
    """
    Build and return the Student Support Agent.

    The Support Agent answers administrative questions by searching a
    curated policy knowledge base. It is intentionally more concise
    than the Tutor Agent — students want quick, accurate answers to
    logistical questions, not lengthy explanations.

    Returns:
        A configured CrewAI Agent ready for task assignment.
    """
    llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.5,
    )

    return Agent(
        role="Student Support Specialist",
        goal=(
            "Answer questions about courses, schedules, policies, fees, and "
            "administrative matters accurately and helpfully, ensuring students "
            "always have the information they need to succeed."
        ),
        backstory=(
            "You are a warm, patient student support specialist with 8 years of "
            "experience in academic administration at a mid-sized university. "
            "You know every policy, deadline, and procedure by heart, and you "
            "have a gift for explaining bureaucratic processes in plain language. "
            "You always give complete answers — including relevant deadlines and "
            "edge cases — and you proactively mention related information the "
            "student didn't think to ask about. When a question falls outside "
            "your knowledge base, you direct students to the right office or "
            "contact person rather than guessing."
        ),
        tools=[PolicySearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )
