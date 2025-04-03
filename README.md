# OpenEPA
A repository for bridging the gap between CEJST dataset and user-initiated, model-driven analysis of various demographic questions.

# OpenEPA

An open-source tool designed to bridge the gap between the complex Climate and Economic Justice Screening Tool (CEJST) dataset and users seeking clear answers to demographic and environmental justice questions. OpenEPA allows users to query CEJST data using natural language and receive concise, AI-driven analyses.

## The Problem

The official U.S. Climate and Economic Justice Screening Tool (CEJST) provides critical, nationwide data identifying communities facing significant environmental burdens, climate risks, and socioeconomic challenges due to marginalization and underinvestment. This dataset is vital for initiatives like Justice40, aiming to direct federal resources equitably.

However, navigating and interpreting the raw CEJST dataset, often distributed as large CSV files, typically requires specialized tools (like GIS software) or data analysis expertise, creating a barrier for many potential users like community groups, grant writers, local planners, and concerned citizens.

## The Solution: OpenEPA

OpenEPA democratizes access to insights within the CEJST data by providing a user-friendly web application. Users can:

1.  **Specify a Location:** Filter data down to a specific State, County, or even an individual Census Tract ID.
2.  **Ask Questions Naturally:** Input queries in plain English (e.g., "Summarize the key demographics," "How many tracts are considered disadvantaged?", "What's the average poverty level?").
3.  **Get AI-Powered Summaries:** The application processes the query, retrieves relevant summary statistics for the selected area, and uses an AI model (via the OpenAI API) to generate a direct, concise answer, translating complex data into understandable insights.

This approach avoids overwhelming users with raw data tables or requiring complex analysis steps.

## Core Dataset: CEJST

This tool is designed to work with the **Climate and Economic Justice Screening Tool (CEJST) dataset (version 2.0 structure)**. Key aspects of the CEJST data include:

* **Disadvantaged Communities:** Highlights census tracts burdened by pollution, facing underinvestment, and marginalized by society.
* **Tribal Lands:** Federally Recognized Tribes and Alaska Native Villages are generally considered disadvantaged communities within the CEJST framework.
* **Multiple Burdens:** Uses nationally consistent data across various categories (climate change, energy, health, housing, legacy pollution, transportation, water, workforce development) to determine disadvantage.

*(Based on archived information from Unofficial CEJST / Atlas Public Policy, dated Feb 6, 2025)*

## Features

* Simple web interface for filtering by State, County, or Tract ID.
* Natural language query input for flexible analysis requests.
* Two-step AI process:
    * Query analysis to understand intent and map to available data points.
    * Concise summary generation based on relevant statistics for the filtered area.
* Avoids row-by-row data dumps, providing synthesized answers.
* Built with Python (Flask, Pandas) and standard web technologies (HTML, CSS, JS).

## How It Works (Technical Overview)

The application uses a Python Flask backend to handle web requests and data processing. When a user submits a query:
1.  The backend filters a local copy of the CEJST `2.0-communities.csv` dataset using Pandas based on the provided location filters.
2.  Summary statistics are calculated for the filtered data subset.
3.  A **first call** is made to the OpenAI API to analyze the user's natural language query, identify the core intent (action, subject), and map it to the available summary statistics keys. This step also checks if the query requires data unavailable in the summary.
4.  If the query is feasible, a **second call** is made to the OpenAI API. This prompt includes the user's query, specific instructions for concise summarization, and only the relevant summary statistics identified in the first step.
5.  The final, summarized analysis from the second AI call is returned to the user's browser.

## Getting Started

1.  **Clone Repository:** `git clone <your-repo-url>`
2.  **Install Dependencies:** `pip install Flask pandas openai python-dotenv`
3.  **Obtain Data:** Download the CEJST `2.0-communities.csv` file. Ensure it is placed in the same directory as `app.py`. *(The application expects this specific filename)*.
    * *Note:* Links from the archived site text (potentially outdated):
        * [CEQ CEJST Instructions](<link_if_available>)
        * [CEJST Technical Support Document](<link_if_available>)
        * [2.0 Communities Excel File](<link_if_available>)
        * [2.0 Codebook](<link_if_available>)
        * *(Consider finding current official CEQ/Geoplatform links if possible)*
4.  **Set API Key:** Create a `.env` file in the project root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY='your_actual_openai_key_here'
    ```
    Alternatively, set it as an environment variable.
5.  **Run Server:** `python app.py`
6.  **Access App:** Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address provided in the terminal).

## Contributing

*(Add guidelines here if you want others to contribute - e.g., reporting issues, suggesting features, submitting pull requests).*

## License

*(Specify a license, e.g., MIT, Apache 2.0, or leave blank if undecided).*
