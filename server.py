import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback
import datetime  # For timestamping logs

# Logging function for queries and responses
def log_query_response(query_data, query_analysis, response, num_records, error=None):
    """Log query and response data to a file for training purposes"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory at {logs_dir}")
    
    # Use current date in filename for easier management
    log_file = os.path.join(logs_dir, f'query_logs_{datetime.datetime.now().strftime("%Y%m%d")}.jsonl')
    
    # Create a log entry as JSON
    log_entry = {
        "timestamp": timestamp,
        "state": query_data.get('state', ''),
        "county": query_data.get('county', ''),
        "tract_id": query_data.get('tract_id', ''),
        "user_query": query_data.get('query', ''),
        "records_found": num_records,
        "query_analysis": query_analysis,
        "response": response,
        "error": error
    }
    
    # Append to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"Query logged to {log_file}")
    except Exception as e:
        print(f"ERROR: Failed to write to log file: {e}")

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Data Loading ---
df = None
try:
    csv_path = os.path.join(os.path.dirname(__file__), '2.0-communities.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        # Data Cleaning
        str_cols = ['State/Territory', 'County Name', 'Census tract 2010 ID']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        print(f"DataFrame loaded successfully with {len(df)} records from {csv_path}.")
    else:
        print(f"ERROR: CSV file not found at {csv_path}. Application cannot function without data.")
        df = pd.DataFrame()
except Exception as e:
    print(f"CRITICAL Error loading DataFrame: {e}")
    df = pd.DataFrame()

# --- OpenAI Client Initialization ---
client = None
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"CRITICAL Error initializing OpenAI client: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the analysis request:
    1. Filters data based on user input.
    2. Calculates summary statistics for the filtered data.
    3. Uses Step 1 LLM call to analyze the user query intent and map to available stats.
    4. Uses Step 2 LLM call to generate a concise analysis based on query intent and stats.
    """
    # Initial error checking
    if df is None or df.empty:
        error_msg = "Dataset not loaded on server."
        log_query_response(request.get_json() or {}, {}, error_msg, 0, error=error_msg)
        return jsonify({"error": error_msg}), 500
    
    if not client:
        error_msg = "OpenAI client not configured on server."
        log_query_response(request.get_json() or {}, {}, error_msg, 0, error=error_msg)
        return jsonify({"error": error_msg}), 500

    try:
        request_data = request.get_json()
        print(f"\n--- New Request ---")
        print(f"Received request data: {request_data}")

        state = request_data.get('state')
        county = request_data.get('county')
        tract_id = request_data.get('tract_id')
        user_query = request_data.get('query')

        if not user_query:
            error_msg = "User query cannot be empty."
            log_query_response(request_data, {}, error_msg, 0, error=error_msg)
            return jsonify({"error": error_msg}), 400

        # --- Filtering Logic ---
        print(f"Starting filtering with state='{state}', county='{county}', tract_id='{tract_id}'")
        filtered_df = df.copy()

        available_states = df['State/Territory'].unique().tolist()
        if state:
            state_strip = state.lower().strip()
            filtered_df = filtered_df[filtered_df['State/Territory'].str.lower() == state_strip]
            print(f"After state filter: {len(filtered_df)} records")
            if filtered_df.empty: 
                msg = f"No data found for state: '{state}'. Available states might include: {', '.join(available_states[:10])}..."
                log_query_response(request_data, {}, msg, 0)
                return jsonify({"analysis": msg})

        available_counties = filtered_df['County Name'].unique().tolist() if not filtered_df.empty else []
        if county:
            county_strip = county.lower().strip()
            filtered_df = filtered_df[filtered_df['County Name'].str.lower() == county_strip]
            print(f"After county filter: {len(filtered_df)} records")
            if filtered_df.empty:
                state_name = state if state else "the dataset"
                county_list_str = f" Available counties in {state_name} might include: {', '.join(available_counties[:10])}..." if available_counties else ""
                msg = f"No data found for county: '{county}' in {state_name}.{county_list_str}"
                log_query_response(request_data, {}, msg, 0)
                return jsonify({"analysis": msg})

        if tract_id:
            tract_strip = tract_id.strip()
            filtered_df = filtered_df[filtered_df['Census tract 2010 ID'] == tract_strip]
            print(f"After tract filter: {len(filtered_df)} records")

        if filtered_df.empty:
            msg = "No data found matching the specified filters."
            log_query_response(request_data, {}, msg, 0)
            return jsonify({"analysis": msg})

        # --- Data Summarization ---
        num_tracts_found = len(filtered_df)
        print(f"Calculating summary stats for {num_tracts_found} tracts.")
        summary_stats = {
            "Number of Census Tracts Found": num_tracts_found,
            "Total Population (Sum)": None,
            "Average Percent Hispanic or Latino": None,
            "Average Percent Black or African American alone": None,
            "Average Percent White": None,
            "Number of Disadvantaged Tracts (Identified)": None,
            "Number of Low Income Tracts": None,
            "Average Total Threshold Criteria Exceeded": None,
            "Average Percent Below 200% FPL": None,
            "Average Percent With Asthma": None,
            "Average Percent With Diabetes": None,
        }

        if num_tracts_found > 0:
            # Define column names from dataset
            pop_col = 'Total population'; hisp_col = 'Percent Hispanic or Latino'; black_col = 'Percent Black or African American alone'; white_col = 'Percent White'; disadv_col = 'Identified as disadvantaged'; lowinc_col = 'Is low income?'; thresh_col = 'Total threshold criteria exceeded'; pov_col = 'Adjusted percent of individuals below 200% Federal Poverty Line'; asthma_col = 'Current asthma among adults aged greater than or equal to 18 years'; diabetes_col = 'Diagnosed diabetes among adults aged greater than or equal to 18 years'

            # Sums / Counts (Handle NaNs)
            if pop_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pop_col]):
                summary_stats["Total Population (Sum)"] = int(filtered_df[pop_col].sum())
            if disadv_col in filtered_df.columns:
                 summary_stats["Number of Disadvantaged Tracts (Identified)"] = int(filtered_df[disadv_col].fillna(False).astype(bool).sum())
            if lowinc_col in filtered_df.columns:
                 summary_stats["Number of Low Income Tracts"] = int(filtered_df[lowinc_col].fillna(False).astype(bool).sum())

            # Averages
            stats_to_average = [
                (hisp_col, "Average Percent Hispanic or Latino"), (black_col, "Average Percent Black or African American alone"),
                (white_col, "Average Percent White"), (thresh_col, "Average Total Threshold Criteria Exceeded"),
                (pov_col, "Average Percent Below 200% FPL"), (asthma_col, "Average Percent With Asthma"),
                (diabetes_col, "Average Percent With Diabetes"),]

            for col, key in stats_to_average:
                if col in filtered_df.columns:
                    if pd.api.types.is_numeric_dtype(filtered_df[col]):
                        valid_data = filtered_df[col].dropna()
                        if not valid_data.empty:
                            multiplier = 1 if 'Threshold' in key else 100
                            avg = valid_data.mean()
                            summary_stats[key] = round(avg * multiplier, 1) if multiplier == 1 or avg <= 1 else round(avg, 1)
                    else: print(f"Warning: Column '{col}' not numeric.")
                else: print(f"Warning: Column '{col}' not found.")

        # Filter out stats that couldn't be calculated (are None)
        calculated_summary_stats = {k: v for k, v in summary_stats.items() if v is not None}
        if not calculated_summary_stats:
             msg = "Could not calculate summary statistics for the selected area."
             log_query_response(request_data, {}, msg, 0)
             return jsonify({"analysis": msg})

        # --- STEP 1: Analyze the User Query with an LLM ---
        print("--- Starting Step 1: Query Analysis ---")
        available_data_points = list(calculated_summary_stats.keys())

        step1_system_prompt = f"""You are an expert query analysis assistant for US census and environmental justice data. Analyze the user query to understand the core intent (action) and identify EXACTLY which of the strictly defined 'Available data points' are relevant.
Available data points: {', '.join(available_data_points)}

Mapping Guidelines & Examples:
- **Demographics:** Map 'population'/'how many people'/'amount of people' to 'Total Population (Sum)'. Map ethnicity/race terms ('hispanic', 'latino', 'black', 'african american', 'white', etc.) to the corresponding 'Average Percent...' key.
- **Disadvantage/Income:** Map 'disadvantaged'/'burdened' to 'Number of Disadvantaged Tracts (Identified)'. Map 'low income'/'poverty'/'Federal Poverty Line'/'FPL' to 'Number of Low Income Tracts' and/or 'Average Percent Below 200% FPL'. Map 'thresholds'/'criteria' to 'Average Total Threshold Criteria Exceeded'.
- **Health:** Map 'asthma' to 'Average Percent With Asthma'. Map 'diabetes' to 'Average Percent With Diabetes'.
- **General:** Map 'summarize'/'describe'/'overview'/'composition'/'makeup'/'size' to the action 'summarize'. If no specific field is mentioned, 'relevant_fields' can include primary fields like population and demographics.

Output Instructions:
- Output ONLY a valid JSON object. Do not include any text before or after the JSON.
- Use the following JSON structure:
{{
  "action": "<identified action e.g., summarize, list_value, compare, check_feasibility, unknown>",
  "relevant_fields": ["<list of EXACT key names from Available data points that directly match the query's core concepts>"],
  "unmatched_query_parts": ["<list ONLY substantive concepts from the query that have NO corresponding Available data point>"]
}}
- **Crucially:** Do NOT put common words (like 'how many', 'percentage of', 'people', 'rate', 'level', 'amount', 'of', 'in', 'as', 'opposed', 'to', 'have') in 'unmatched_query_parts' if the main data concept *was* successfully mapped to a 'relevant_field'. Only list concepts representing *data types* that are genuinely unavailable in the provided list.
- If the query asks for something clearly unavailable (e.g., 'education levels', 'crime rates', 'rent vs own'), set action to 'check_feasibility' and list the concept in 'unmatched_query_parts'.
"""
        step1_user_content = f"Analyze this user query: \"{user_query}\""

        query_analysis = {"action": "summarize", "relevant_fields": [], "unmatched_query_parts": ["Query analysis failed (default)"]}
        try:
            step1_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": step1_system_prompt},
                    {"role": "user", "content": step1_user_content}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            step1_result_str = step1_completion.choices[0].message.content
            print(f"Step 1 Raw Result: {step1_result_str}")
            query_analysis = json.loads(step1_result_str)
            print(f"Step 1 Parsed Analysis: {query_analysis}")

        except Exception as e:
            print(f"Error during Step 1 (Query Analysis) API call: {e}")

        # --- Check Feasibility / Prepare for Step 2 ---
        action = query_analysis.get("action", "unknown")
        unmatched = query_analysis.get("unmatched_query_parts", [])
        relevant = query_analysis.get("relevant_fields", [])

        if action == "check_feasibility" or (action == "unknown" and unmatched):
            missing_parts = unmatched if unmatched else ["unknown concepts"]
            analysis_result = f"Specific data for '{', '.join(missing_parts)}' needed for your query is not available in the calculated summary statistics."
            print("Step 1 indicated query requires unavailable data OR action unknown with unmatched parts. Returning message.")
            log_query_response(request_data, query_analysis, analysis_result, num_tracts_found)
            return jsonify({"analysis": analysis_result})

        # --- STEP 2: Perform Data Analysis ---
        print("--- Starting Step 2: Data Analysis ---")

        # Decide which stats to send: relevant ones if found, otherwise all calculated stats
        final_summary_stats_to_send = calculated_summary_stats
        if relevant:
             filtered_relevant = [f for f in relevant if f in calculated_summary_stats]
             if filtered_relevant:
                 final_summary_stats_to_send = {k: v for k, v in calculated_summary_stats.items() if k in filtered_relevant or k == "Number of Census Tracts Found"}
             elif not calculated_summary_stats:
                  msg = "Could not calculate any summary statistics for the selected area, unable to proceed."
                  log_query_response(request_data, query_analysis, msg, num_tracts_found)
                  return jsonify({"analysis": msg})

        final_summary_data_string = json.dumps(final_summary_stats_to_send, indent=2)

        # Step 2 System Prompt
        step2_system_prompt = """You are an AI assistant analyzing US Census tract data summaries.
        - Provide a concise summary addressing the user's query based *only* on the provided summary statistics for the specified location.
        - Do NOT list individual census tracts.
        - State results directly. Do not use introductory phrases like "Based on the data..." or "The data shows...".
        - If the query asks about something not present in the summary statistics, state "Specific data for that query is not available in the summary."
        - Keep the response brief and focused on the overall area described by the statistics.
        - Format the output as a short paragraph or bullet points.
        """

        location_name = tract_id if tract_id else (county if county else (state if state else "Selected Area"))
        action_verb = query_analysis.get("action", "summarize")

        # Step 2 User Content
        step2_user_content = f"""
        Analysis Request:
        Location: {location_name} ({state if state else ''}{', '+county if county else ''})
        User Query: "{user_query}"
        Identified Action: {action_verb}
        Identified Relevant Fields by Step 1: {json.dumps(relevant)}

        Summary Statistics Provided for Analysis:
        {final_summary_data_string}

        Provide a concise response answering the User Query for the Location based on the statistics provided and the Identified Action. Do not list individual tracts. If the statistics don't directly support the query, explain briefly based *only* on the provided stats.
        """

        print(f"Sending summarized prompt for Step 2 to OpenAI (User Content Length: {len(step2_user_content)} chars)")

        step2_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": step2_system_prompt},
                {"role": "user", "content": step2_user_content}
            ],
            temperature=0.2,
            max_tokens=350
        )

        analysis_result = step2_completion.choices[0].message.content
        print("Received analysis from OpenAI (Step 2).")
        
        # Log the successful interaction
        log_query_response(
            request_data, 
            query_analysis, 
            analysis_result, 
            num_tracts_found
        )
        
        return jsonify({"analysis": analysis_result})

    except Exception as e:
        print(f"--- ERROR in /analyze route ---")
        traceback.print_exc()
        error_msg = f"An unexpected server error occurred: {str(e)}"
        log_query_response(request.get_json() or {}, {}, "", 0, error=error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)