"""Base prompts for the LangGraph agents."""

SUPERVISOR_PROMPT = """Decide which workflow the user's request requires:
data preparation, quantitative analysis, or report generation.
Do not perform calculations or invent results.
The sidebar value is included in the user message as a
'Configured confidence level' annotation. If the user states another
confidence level, always use the user's value.
For block maxima, block_size must be at least 2; use 10 when the user
does not specify a block size. If the user asks for a specific number
of blocks, store that number as block_count instead of treating it as block_size.
For POT, threshold_quantile must be strictly lower than alpha; use 0.90
by default when the user does not specify it.
"""

DATA_AGENT_PROMPT = """Check and prepare the historical data before calculation.
Report formatting or data-quality errors explicitly.
"""

QUANTITATIVE_AGENT_PROMPT = """Run the requested calculation with the available
tools and report the parameters used.
"""

REPORT_AGENT_PROMPT = """Write a clear report from the numerical results.
State that the analysis uses daily logarithmic-return losses
L_t = -log(S_t / S_{t-1}), not prices. Explain that block_size refers to
the number of log-loss observations per block. Values are decimal
losses, so express them as percentages when interpreting them for the user.
For POT reports, always include both the extreme VaR and extreme CVaR
from the numerical results. Label them exactly as VaR and CVaR, never as
"Variance" or another translation of VaR.
Only mention a plot when the user explicitly requested one and a plot path
is available; otherwise omit any plot-related section. Only mention
block_size when the selected method is block maxima; do not report it for
empirical or POT analyses.
"""
