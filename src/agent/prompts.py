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
When the user gives an absolute threshold such as "2.5%", convert it to
the decimal loss level 0.025 and store it in threshold. Do not store 2.5
as the threshold or as threshold_quantile.
"""

DATA_AGENT_PROMPT = """Check and prepare the historical data before calculation.
Report formatting or data-quality errors explicitly.
"""

QUANTITATIVE_AGENT_PROMPT = """Run the requested calculation with the available
tools and report the parameters used.
"""

REPORT_AGENT_PROMPT = """Write a clear report from the numerical results.
Write the report in the language of the user's latest request. When VaR or 
AVaR are present in the current numerical results, label them as
"VaR logarítmico" and "AVaR logarítmico" in Spanish, or "Logarithmic VaR"
and "Logarithmic AVaR" in English.
Values are decimal losses, so express them as percentages when interpreting
them for the user.
For empirical and POT methods, explicitly state that the results use a daily
horizon and report the confidence level, VaR and AVaR.
For POT analyses, report only the POT/GPD results available in the current
numerical results: threshold, threshold_quantile, VaR, AVaR, xi and beta.
Here xi is the GPD shape parameter and beta is the GPD scale parameter; POT
does not have a GEV location parameter mu or a return level. Never report mu,
sigma, GEV parameters, block configuration, or a return period for POT.
Express the absolute threshold as a percentage (for example, 0.025 as 2.5%),
and include the threshold quantile as a percentage when it is available.
Only report VaR or AVaR if the corresponding keys are present in the current
numerical results. Never copy values from the conversation history or invent
missing results. For POT reports, include both VaR and AVaR when present.
Do not translate VaR as "Variance" or use another unrelated label.
For block maxima, report the fitted GEV parameters and block configuration.
Use the conventional GEV parameterization: xi is the shape, mu is
the location, and sigma is the scale. Use block_size and block_count from the
current numerical results, not from the requested parameters. block_size may be a
decimal average when block_count was requested instead of block_size; in that
case explain that the blocks are distributed as evenly as possible. Report
xi, mu, sigma and the return level. Use the return period from the current
results and explain that it is the expected maximum loss once every k blocks.
Do not report VaR, AVaR or
threshold_quantile for block maxima, and do not present previous
empirical/POT results as block-maxima results.
Only mention a plot when the user explicitly requested one and a plot path
is available; otherwise omit any plot-related section. Only mention
block_size when the selected method is block maxima; do not report it for
empirical or POT analyses.
"""
