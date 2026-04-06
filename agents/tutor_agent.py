"""
Tutor Agent — personalised teaching and concept explanation.

Uses GPT-4 with a DocumentSearchTool to search uploaded course materials
and deliver student-level-appropriate explanations.
"""

from crewai import Agent, LLM

from tools.document_tools import DocumentSearchTool


def create_tutor_agent() -> Agent:
    """
    Build and return the Tutor Agent.

    The Tutor searches course documents via RAG, adapts its explanation
    to the student's current performance level (provided as context from
    the Analytics Agent), and always ends with practice questions.

    Returns:
        A configured CrewAI Agent ready for task assignment.
    """
    llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.7)

    return Agent(
        role="AI Tutor and Learning Guide",
        goal=(
            "Provide personalised tutoring, explain complex concepts clearly, "
            "and adapt explanations to the student's current performance level "
            "and learning style as determined by the analytics data."
        ),
        backstory=(
            "You are an award-winning teacher with 15 years of experience across "
            "mathematics, sciences, and the humanities. You are famous for your "
            "ability to break down the most intimidating topics into simple, "
            "memorable explanations using real-world analogies and worked examples. "
            "You always search the student's uploaded course materials first so "
            "your answers stay tightly aligned with what they are actually studying. "
            "You never just hand over answers — you guide students to *understand* "
            "concepts. You close every response with two or three targeted practice "
            "questions that reinforce the key ideas."
        ),
        tools=[DocumentSearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
