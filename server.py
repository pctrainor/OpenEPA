import os
import pandas as pd
import io
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback # Import traceback for better error logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
# --- Data Loading ---
# Initialize df as None globally or load it here
df = None
try:
    # Use relative path for robustness
    csv_path = os.path.join(os.path.dirname(__file__), '2.0-communities.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False) # low_memory=False can help with mixed types
        # Basic data cleaning - apply consistently
        str_cols = ['State/Territory', 'County Name', 'Census tract 2010 ID']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        print(f"DataFrame loaded successfully with {len(df)} records from {csv_path}.")
        # print(f"Columns: {df.columns.tolist()}") # Keep for debugging if needed
    else:
        print(f"ERROR: CSV file not found at {csv_path}. Application cannot function without data.")
        # Consider exiting or using placeholder data ONLY for basic testing
        df = pd.DataFrame() # Set to empty if file not found

except Exception as e:
    print(f"CRITICAL Error loading DataFrame: {e}")
    df = pd.DataFrame() # Ensure df is defined even on error

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
    # client remains None


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Initial checks moved here for clarity
    if df is None or df.empty:
        return jsonify({"error": "Dataset not loaded or empty on server. Check server logs."}), 500
    if not client:
        return jsonify({"error": "OpenAI client not configured on server. Check server logs."}), 500

    try:
        request_data = request.get_json()
        print(f"Received request data: {request_data}")

        state = request_data.get('state')
        county = request_data.get('county')
        tract_id = request_data.get('tract_id')
        user_query = request_data.get('query')

        if not user_query:
            return jsonify({"error": "User query cannot be empty."}), 400

        # --- Filtering Logic (Same as before) ---
        print(f"Starting filtering with state={state}, county={county}, tract_id={tract_id}")
        filtered_df = df.copy()
        # ... (Keep the improved filtering logic from the previous version here) ...
        available_states = df['State/Territory'].unique().tolist()
        if state:
            state_strip = state.lower().strip()
            filtered_df = filtered_df[filtered_df['State/Territory'].str.lower() == state_strip]
            print(f"After state filter: {len(filtered_df)} records")
            if filtered_df.empty: return jsonify({"analysis": f"No data found for state: {state}..."})
        available_counties = filtered_df['County Name'].unique().tolist() if not filtered_df.empty else []
        if county:
            county_strip = county.lower().strip()
            filtered_df = filtered_df[filtered_df['County Name'].str.lower() == county_strip]
            print(f"After county filter: {len(filtered_df)} records")
            if filtered_df.empty: return jsonify({"analysis": f"No data found for county: {county}..."})
        if tract_id:
            tract_strip = tract_id.strip()
            filtered_df = filtered_df[filtered_df['Census tract 2010 ID'] == tract_strip] # Assumes ID is already string
            print(f"After tract filter: {len(filtered_df)} records")
        if filtered_df.empty: return jsonify({"analysis": "No data found matching the specified filters."})
        # --- End Filtering ---


        # --- Data Summarization (Same as before) ---
        num_tracts_found = len(filtered_df)
        summary_stats = { # Define structure first
            "Number of Census Tracts Found": num_tracts_found, "Total Population (Sum)": None,
            "Average Percent Hispanic or Latino": None, "Average Percent Black or African American alone": None,
            "Average Percent White": None, "Number of Disadvantaged Tracts (Identified)": None,
            "Number of Low Income Tracts": None, "Average Total Threshold Criteria Exceeded": None,
        }
        # ... (Keep the calculation logic for summary_stats here) ...
        if num_tracts_found > 0:
            pop_col='Total population'; hisp_col='Percent Hispanic or Latino'; black_col='Percent Black or African American alone'; white_col='Percent White'; disadv_col='Identified as disadvantaged'; lowinc_col='Is low income?'; thresh_col='Total threshold criteria exceeded'
            if pop_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pop_col]): summary_stats["Total Population (Sum)"] = int(filtered_df[pop_col].sum())
            for col, key in [(hisp_col, "Average Percent Hispanic or Latino"), (black_col, "Average Percent Black or African American alone"), (white_col, "Average Percent White"), (thresh_col, "Average Total Threshold Criteria Exceeded")]:
                if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
                   valid_data = filtered_df[col].dropna(); summary_stats[key] = round(valid_data.mean() * 100, 1) if not valid_data.empty else None
            if disadv_col in filtered_df.columns: summary_stats["Number of Disadvantaged Tracts (Identified)"] = int(filtered_df[disadv_col].fillna(False).sum()) # Handle NaN before sum
            if lowinc_col in filtered_df.columns: summary_stats["Number of Low Income Tracts"] = int(filtered_df[lowinc_col].fillna(False).sum()) # Handle NaN before sum
        # --- End Summarization ---

        # --- STEP 1: Analyze the User Query with an LLM ---
        print("--- Starting Step 1: Query Analysis ---")
        available_data_points = list(summary_stats.keys()) # Use the keys from our calculated stats

        step1_system_prompt = f"""You are a query analysis assistant. Analyze the user query to understand the intent (action) and identify which of the available data points are relevant.
Available data points: {', '.join(available_data_points)}

Common query mappings:
- "size", "how big", "how large" → "Number of Census Tracts Found", "Total Population (Sum)"
- "demographics", "composition", "ethnicity" → demographic percentages
- "disadvantaged" → "Number of Disadvantaged Tracts (Identified)"
- "income" → "Number of Low Income Tracts"

Output ONLY a JSON object with the following structure:
{{
  "action": "describe", "compare", "summarize", "list_value", "unknown"}},
  "relevant_fields": ["list", "of", "relevant", "data", "point", "keys", "from", "the", "list", "above"],
  "unmatched_query_parts": ["list", "of", "concepts", "in", "the", "query", "that", "don't", "match", "available", "data"]
}}
Map simple terms to appropriate fields using common sense. Only use "unmatched_query_parts" when truly nothing matches.
"""
        step1_user_content = f"Analyze this user query: \"{user_query}\""

        try:
            step1_completion = client.chat.completions.create(
                model="gpt-3.5-turbo", # Can use a faster/cheaper model potentially
                messages=[
                    {"role": "system", "content": step1_system_prompt},
                    {"role": "user", "content": step1_user_content}
                ],
                temperature=0.0, # Low temperature for consistent JSON output
                max_tokens=150,
                response_format={"type": "json_object"} # Request JSON output if model supports
            )
            step1_result_str = step1_completion.choices[0].message.content
            print(f"Step 1 Raw Result: {step1_result_str}")
            query_analysis = json.loads(step1_result_str) # Parse the JSON response
            print(f"Step 1 Parsed Analysis: {query_analysis}")

        except Exception as e:
            print(f"Error during Step 1 (Query Analysis) API call: {e}")
            # Fallback: If query analysis fails, proceed with the original single-step approach
            query_analysis = {"action": "summarize", "relevant_fields": [], "unmatched_query_parts": ["Query analysis failed"]}


        # --- Check Feasibility Based on Step 1 ---
        action = query_analysis.get("action")
        unmatched = query_analysis.get("unmatched_query_parts")
        relevant = query_analysis.get("relevant_fields")
        
        # Only stop if truly nothing matches
        if action == "check_feasibility" or (unmatched and not relevant):
            missing_parts = unmatched if unmatched else ["unknown concepts"]
            analysis_result = f"Specific data for '{', '.join(missing_parts)}' needed for your query is not available in the summary statistics."
            print("Step 1 indicated query requires unavailable data. Returning message.")
            return jsonify({"analysis": analysis_result})

        # --- STEP 2: Perform Data Analysis using Summarized Data and Query Analysis ---
        print("--- Starting Step 2: Data Analysis ---")

        # Filter summary_stats to include only relevant fields if identified, otherwise use all
        final_summary_stats = summary_stats
        relevant_fields = query_analysis.get("relevant_fields")
        if relevant_fields:
             final_summary_stats = {k: v for k, v in summary_stats.items() if k in relevant_fields or k == "Number of Census Tracts Found"} # Always include count
             # If filtering results in empty dict (besides count), revert to all stats
             if len(final_summary_stats) <= 1 and summary_stats:
                 final_summary_stats = summary_stats


        final_summary_data_string = json.dumps(final_summary_stats, indent=2)

        # Use the same directive system prompt as before
        step2_system_prompt = """You are an AI assistant analyzing US Census tract data summaries.
        - Provide a concise summary addressing the user's query based *only* on the provided summary statistics for the specified location.
        - Do NOT list individual census tracts.
        - State results directly. Do not use introductory phrases like "Based on the data..." or "The data shows...".
        - If the query asks about something not present in the summary statistics, state "Specific data for that query is not available in the summary."
        - Keep the response brief and focused on the overall area described by the statistics.
        - Format the output as a short paragraph or bullet points.
        """

        location_name = tract_id if tract_id else (county if county else (state if state else "Selected Area"))
        action_verb = query_analysis.get("action", "summarize") # Use identified action

        # Modify user content slightly based on identified action
        step2_user_content = f"""
        Analysis Request:
        Location: {location_name}
        Action Requested: {action_verb} (based on user query: "{user_query}")

        Summary Statistics for the Filtered Area:
        {final_summary_data_string}

        Provide a concise response for the location based on the action requested and the statistics provided. Do not list individual tracts.
        """

        print(f"Sending summarized prompt for Step 2 to OpenAI (User Content Length: {len(step2_user_content)} chars)")

        step2_completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4" etc.
            messages=[
                {"role": "system", "content": step2_system_prompt},
                {"role": "user", "content": step2_user_content}
            ],
            temperature=0.2,
            max_tokens=300
        )

        analysis_result = step2_completion.choices[0].message.content
        print("Received analysis from OpenAI (Step 2).")
        return jsonify({"analysis": analysis_result})

    except Exception as e:
        print(f"Error during analysis function: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An server error occurred during analysis: {str(e)}"}), 500


# Keep the if __name__ == '__main__': block
if __name__ == '__main__':
    # Make sure templates and static folders exist
    # if not os.path.exists('templates'): os.makedirs('templates') # Not needed if using Flask conventions
    # if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True)