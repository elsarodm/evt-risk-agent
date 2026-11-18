# EVT Risk Agent

EVT Risk Agent is an interactive market-risk application built with Python,
Streamlit, LangGraph and OpenAI. It loads historical market prices, converts
them into daily logarithmic losses and estimates tail risk using three
Extreme Value Theory (EVT) approaches:

- Historical or empirical daily VaR and AVaR
- Block maxima with a generalized extreme value distribution
- Peaks Over Threshold (POT) with a generalized Pareto distribution

The application also uses a supervisor agent to interpret natural-language
requests and a report agent to present the numerical results in Spanish or
English.

## Features

- CSV upload with automatic date and price column detection
- Conversational interface that keeps the latest 10 messages
- Configurable confidence level from the sidebar
- Empirical daily VaR and AVaR calculation
- Block-maxima analysis with configurable block size or block count
- POT threshold selection by quantile or absolute loss level
- Optional POT tail-diagnostics plot
- Structured LLM output for model selection and parameter extraction
- Temporary storage of uploaded files and generated plots

## Project Structure

```text
evt-risk-agent/
├── src/
│   ├── app.py                 # Streamlit user interface
│   ├── agent/
│   │   ├── graph.py           # LangGraph workflow and agents
│   │   ├── prompts.py         # Supervisor and report prompts
│   │   └── tools.py           # Data cleaning and risk calculations
│   └── data/                  # Uploaded CSV files and generated plots
├── .env.example
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- Internet access for the language-model requests

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/elsarodm/evt-risk-agent.git
cd evt-risk-agent
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, activate it with:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create an `.env` file from the example:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

You can also configure the model used for the agents through
`OPENAI_MODEL`. If it is not set, the project uses `gpt-4o-mini`.

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Keep the `.env` file private. It is excluded by `.gitignore`.

## Running the Application

Start Streamlit from the repository root:

```bash
streamlit run src/app.py
```

Streamlit will usually open the application at
`http://localhost:8501`.

## Input CSV Format

Upload a CSV containing at least two columns:

- A date column
- A price column

Recognized date names include `date`, `datetime`, `timestamp`, `time` and
`fecha`. Recognized price names include `adjusted close`, `close`, `price`,
`cierre`, `precio`, `value` and `valor`.

If a price name is not recognized, the application selects the unique numeric
column that is not the date column.

Example:

```csv
date,close
2025-01-02,100.50
2025-01-03,101.20
2025-01-06,99.80
```

The data loader:

1. Parses dates and numeric prices.
2. Removes missing values and non-positive prices.
3. Removes duplicate dates, keeping the last record.
4. Sorts observations chronologically.
5. Calculates logarithmic losses as:
   `L[t] = -log(P[t] / P[t-1])`.

At least two valid positive prices are required.

## Using the Agent

After uploading a CSV and selecting the confidence level in the sidebar,
request an analysis in the chat. Example prompts include:

```text
Calculate empirical daily VaR and AVaR at 99% confidence.
```

```text
Apply the POT method with a threshold of 2.5% and request a tail plot.
```

```text
Use block maxima with 60 blocks.
```

```text
Change the POT confidence level to 99%.
```

The supervisor extracts the requested method and parameters. The data and
quantitative agents prepare the losses and execute the selected calculation.
The report agent then generates a current-results summary.

## LangGraph Workflow

The application follows this workflow:

```text
START
  -> supervisor
  -> data agent
  -> quantitative agent
  -> report agent
  -> END
```

The supervisor uses structured Pydantic output to select one of:

- `empirical`
- `pot`
- `block`

The graph state retains the uploaded CSV path, the latest chat messages,
processed losses, quantitative results and the generated plot path.

## Risk Methods

### Empirical Method

The historical method calculates quantiles directly from the loss
distribution:

- VaR is the loss at the configured confidence quantile.
- AVaR is the average daily loss at or beyond that VaR threshold.

### Peaks Over Threshold

The POT method:

1. Converts the threshold quantile or absolute threshold into one threshold
   level.
2. Calculates exceedances over that threshold.
3. Fits a generalized Pareto distribution.
4. Calculates extreme VaR and AVaR.

The threshold must be strictly below the confidence level.

### Block Maxima

The block-maxima method divides the losses into non-overlapping blocks and
fits a generalized extreme value distribution to the maximum loss in each
block. It reports the GEV parameters (`xi`, `mu`, and `sigma`) and a return 
level for the standard return period:

```text
k = 1 / (1 - alpha)
```

For example, `alpha=0.99` corresponds to a return period of 100 blocks. The
return level is the expected maximum loss exceeded approximately once every
`k` blocks.

The user can specify either:

- `block_size`, defaulting to 10 observations
- `block_count`, which must be at least 3

## Generated Files

Uploaded CSV files are stored temporarily under `src/data/`.
POT plots are generated at:

```text
src/data/plot_temp.png
```

Files matching `src/data/*.csv` and `src/data/plot_temp.png` are ignored by
Git.

## Troubleshooting

### The application reports that no CSV is loaded

Upload a valid CSV in the sidebar before sending a chat request.

### The application cannot identify the price column

Rename the price column to `price`, `close`, `adjusted close`, or one of the
recognized names, or ensure that only one other numeric column is present.

### The OpenAI request fails

Verify that:

- `OPENAI_API_KEY` is defined in `.env`
- The API key is valid
- The selected model is available to the account

### The POT fit fails

POT fitting requires at least three observations above the threshold and a
valid GPD fit. Try a lower threshold or verify that the uploaded data contain
enough extreme observations.

### The block-maxima fit fails

Block maxima requires at least three complete blocks and at least two
observations per block.
