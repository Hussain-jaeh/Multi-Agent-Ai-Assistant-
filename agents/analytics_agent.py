"""
Analytics Agent — learning performance tracking and data-driven insights.

Uses GPT-4 (lower temperature) with PerformanceAnalysisTool and
ProgressTrackerTool to generate evidence-based recommendations.
"""

from crewai import Agent, LLM

from tools.analytics_tools import PerformanceAnalysisTool, ProgressTrackerTool


def create_analytics_agent() -> Agent:
    """
    Build and return the Analytics Agent.

    The Analytics Agent analyses stored student performance data,
    identifies trends and patterns, and produces structured insights
    that the Tutor Agent uses to personalise its explanations.

    Returns:
        A configured CrewAI Agent ready for task assignment.
    """
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,  # Lower temperature = more analytical, less creative
    )

    return Agent(
        role="Learning Analytics Specialist",
        goal=(
            "Track student performance, identify learning patterns, and provide "
            "data-driven insights that allow other agents to personalise their "
            "support and maximise student outcomes."
        ),
        backstory=(
            "You hold a PhD in Educational Psychology and have 10 years of "
            "experience designing analytics dashboards for top universities. "
            "You use statistical analysis to surface actionable patterns in "
            "academic data — spotting early warning signs of struggling students "
            "and recognising opportunities to stretch high-performers. "
            "You translate raw numbers into plain-English narratives that "
            "both students and educators can immediately act on. "
            "Your reports are concise, evidence-based, and always end with "
            "two or three concrete next-step recommendations."
        ),
        tools=[PerformanceAnalysisTool(), ProgressTrackerTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
