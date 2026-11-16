"""Base prompts for the LangGraph agents."""

SUPERVISOR_PROMPT = """Decide which workflow the user's request requires:
data preparation, quantitative analysis, or report generation.
Do not perform calculations or invent results.
The sidebar value is included in the user message as a
'Configured confidence level' annotation. If the user states another
confidence level, always use the user's value.
"""

DATA_AGENT_PROMPT = """Check and prepare the historical data before calculation.
Report formatting or data-quality errors explicitly.
"""

QUANTITATIVE_AGENT_PROMPT = """Run the requested calculation with the available
tools and report the parameters used.
"""

REPORT_AGENT_PROMPT = """Write a clear report from the numerical results and
indicate whether a plot is available.
"""
