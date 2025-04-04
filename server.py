import os
import pandas as pd
import io
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback # Import traceback for better error logging
import re # Import regex - might be useful later

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

# --- Helper Function for Safe Averaging (WITH DEBUG PRINTS) ---
def calculate_safe_average(dataframe, column_name, is_percentage=True):
    """Calculates mean for a numeric column, handles missing data/types."""
    if column_name in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column_name]):
            valid_data = dataframe[column_name].dropna()
            if not valid_data.empty:
                avg = valid_data.mean()
                multiplier = 100 if is_percentage and avg <= 1 and avg >= 0 else 1 # Added avg >= 0 check

                # --- DEBUG PRINTS ---
                print(f"DEBUG avg_calc: Col='{column_name}', Raw Avg={avg:.4f}, is_percentage={is_percentage}, Multiplier={multiplier}")
                # --- END DEBUG ---

                result = round(avg * multiplier, 1)
                return result
            else:
                print(f"DEBUG avg_calc: Col='{column_name}' - No valid data after dropna.")
        else:
             print(f"Warning: Column '{column_name}' not numeric, cannot average.")
    else:
         print(f"Warning: Column '{column_name}' not found.")
    return None # Return None if calculation fails
# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the analysis request with two-step LLM process."""
    if df is None or df.empty:
        return jsonify({"error": "Dataset not loaded on server."}), 500
    if not client:
        return jsonify({"error": "OpenAI client not configured on server."}), 500

    try:
        request_data = request.get_json()
        print(f"\n--- New Request ---")
        print(f"Received request data: {request_data}")

        state = request_data.get('state')
        county = request_data.get('county')
        tract_id = request_data.get('tract_id')
        user_query = request_data.get('query')

        if not user_query:
            return jsonify({"error": "User query cannot be empty."}), 400

        # --- Filtering Logic ---
        print(f"Starting filtering with state='{state}', county='{county}', tract_id='{tract_id}'")
        filtered_df = df.copy()
        # (Filtering logic remains the same - ensure it's robust)
        available_states = df['State/Territory'].unique().tolist()
        if state:
            state_strip = state.lower().strip(); filtered_df = filtered_df[filtered_df['State/Territory'].str.lower() == state_strip]
            if filtered_df.empty: return jsonify({"analysis": f"No data found for state: '{state}'."})
        available_counties = filtered_df['County Name'].unique().tolist() if not filtered_df.empty else []
        if county:
            county_strip = county.lower().strip(); filtered_df = filtered_df[filtered_df['County Name'].str.lower() == county_strip]
            if filtered_df.empty: return jsonify({"analysis": f"No data found for county: '{county}' in state '{state}'. Available: {', '.join(available_counties[:5])}..."})
        if tract_id:
            tract_strip = tract_id.strip(); filtered_df = filtered_df[filtered_df['Census tract 2010 ID'] == tract_strip]
            if filtered_df.empty: return jsonify({"analysis": f"No data found for Tract ID: '{tract_id}'."})
        if filtered_df.empty: return jsonify({"analysis": "No data found matching filters."})
        # --- End Filtering ---

        # --- Data Summarization (Expanded) ---
        num_tracts_found = len(filtered_df)
        print(f"Calculating summary stats for {num_tracts_found} tracts.")

        # Define source column names from codebook
        cols = {
            "pop": 'Total population', "hisp": 'Percent Hispanic or Latino', "black": 'Percent Black or African American alone',
            "white": 'Percent White', "disadv": 'Identified as disadvantaged', "lowinc_flag": 'Is low income?',
            "thresh": 'Total threshold criteria exceeded', "pov200": 'Adjusted percent of individuals below 200% Federal Poverty Line',
            "asthma": 'Current asthma among adults aged greater than or equal to 18 years',
            "diabetes": 'Diagnosed diabetes among adults aged greater than or equal to 18 years',
            "unemp": 'Unemployment (percent)', "energy": 'Energy burden', "housing_burden": 'Housing burden (percent)',
            "pm25": 'PM2.5 in the air', "diesel": 'Diesel particulate matter exposure', "traffic": 'Traffic proximity and volume',
            "hazwaste": 'Proximity to hazardous waste sites', "superfund": 'Proximity to NPL (Superfund) sites',
            "rmp": 'Proximity to Risk Management Plan (RMP) facilities', "lead_paint": 'Percent pre-1960s housing (lead paint indicator)',
            "wastewater": 'Wastewater discharge', "life_exp": 'Life expectancy (years)'
        }

        summary_stats = {"Number of Census Tracts Found": num_tracts_found}

        # Calculate stats, using helper function for averages
        if num_tracts_found > 0:
            # Sums / Counts
            if cols["pop"] in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[cols["pop"]]):
                summary_stats["Total Population (Sum)"] = int(filtered_df[cols["pop"]].sum())
            if cols["disadv"] in filtered_df.columns:
                summary_stats["Number of Disadvantaged Tracts"] = int(filtered_df[cols["disadv"]].fillna(False).astype(bool).sum())
            if cols["lowinc_flag"] in filtered_df.columns:
                summary_stats["Number of Low Income Tracts"] = int(filtered_df[cols["lowinc_flag"]].fillna(False).astype(bool).sum())

            # Averages (Percentages converted to 0-100, others kept as is)
            summary_stats["Avg Percent Hispanic or Latino"] = calculate_safe_average(filtered_df, cols["hisp"])
            summary_stats["Avg Percent Black or African American"] = calculate_safe_average(filtered_df, cols["black"])
            summary_stats["Avg Percent White"] = calculate_safe_average(filtered_df, cols["white"])
            summary_stats["Avg Percent Below 200% FPL"] = calculate_safe_average(filtered_df, cols["pov200"])
            summary_stats["Avg Unemployment Rate (%)"] = calculate_safe_average(filtered_df, cols["unemp"])
            summary_stats["Avg Energy Burden (%)"] = calculate_safe_average(filtered_df, cols["energy"])
            summary_stats["Avg Housing Burden (%)"] = calculate_safe_average(filtered_df, cols["housing_burden"])
            summary_stats["Avg Percent Pre-1960s Housing (Lead Paint Proxy)"] = calculate_safe_average(filtered_df, cols["lead_paint"])
            summary_stats["Avg PM2.5 Exposure (ug/m3)"] = calculate_safe_average(filtered_df, cols["pm25"], is_percentage=False)
            summary_stats["Avg Diesel PM Exposure"] = calculate_safe_average(filtered_df, cols["diesel"], is_percentage=False)
            summary_stats["Avg Traffic Proximity Score"] = calculate_safe_average(filtered_df, cols["traffic"], is_percentage=False)
            summary_stats["Avg Proximity to HazWaste Sites Score"] = calculate_safe_average(filtered_df, cols["hazwaste"], is_percentage=False)
            summary_stats["Avg Proximity to Superfund Sites Score"] = calculate_safe_average(filtered_df, cols["superfund"], is_percentage=False)
            summary_stats["Avg Proximity to RMP Facilities Score"] = calculate_safe_average(filtered_df, cols["rmp"], is_percentage=False)
            summary_stats["Avg Wastewater Discharge Score"] = calculate_safe_average(filtered_df, cols["wastewater"], is_percentage=False)
            summary_stats["Avg Percent With Asthma"] = calculate_safe_average(filtered_df, cols["asthma"])
            summary_stats["Avg Percent With Diabetes"] = calculate_safe_average(filtered_df, cols["diabetes"])
            summary_stats["Avg Life Expectancy (Years)"] = calculate_safe_average(filtered_df, cols["life_exp"], is_percentage=False)
            summary_stats["Avg Total Threshold Criteria Exceeded"] = calculate_safe_average(filtered_df, cols["thresh"], is_percentage=False)

        # Filter out stats that are None
        calculated_summary_stats = {k: v for k, v in summary_stats.items() if v is not None}
        if not calculated_summary_stats or calculated_summary_stats.get("Number of Census Tracts Found", 0) == 0 :
            return jsonify({"analysis": "Could not calculate valid summary statistics for the selected area."})
        # --- End Summarization ---


        # --- STEP 1: Analyze the User Query ---
        print("--- Starting Step 1: Query Analysis ---")
        available_data_concepts_str = f"""
        - Basic Info: Number of Census Tracts Found, Total Population (Sum)
        - Demographics: Avg Percent Hispanic or Latino, Avg Percent Black or African American, Avg Percent White
        - Economic Status: Number of Disadvantaged Tracts, Number of Low Income Tracts, Avg Percent Below 200% FPL, Avg Unemployment Rate (%), Avg Energy Burden (%), Avg Housing Burden (%)
        - Environmental Exposure: Avg PM2.5 Exposure (ug/m3), Avg Diesel PM Exposure, Avg Traffic Proximity Score
        - Environmental/Legacy Sites: Avg Proximity to HazWaste Sites Score, Avg Proximity to Superfund Sites Score, Avg Proximity to RMP Facilities Score, Avg Wastewater Discharge Score, Avg Percent Pre-1960s Housing (Lead Paint Proxy)
        - Health Outcomes: Avg Percent With Asthma, Avg Percent With Diabetes, Avg Life Expectancy (Years)
        - Burden Score: Avg Total Threshold Criteria Exceeded
        (Note: Proximity scores may need context; lower might mean closer/worse)
        """

        step1_system_prompt = f"""You are an expert query analysis assistant for US CEJST data. Your goal is to map the user query to the specific calculated summary statistics provided.
Available Summary Statistics Concepts & (Exact Keys):
{available_data_concepts_str}

Mapping Guidelines:
- Map common terms to the corresponding Exact Key (e.g., 'poverty' -> 'Avg Percent Below 200% FPL', 'asthma' -> 'Avg Percent With Asthma', 'air quality'/'pm2.5' -> 'Avg PM2.5 Exposure (ug/m3)', 'waste sites' -> 'Avg Proximity to HazWaste Sites Score', 'disadvantaged' -> 'Number of Disadvantaged Tracts').
- If a query is general ('summarize', 'describe'), map to 'summarize' action and potentially include primary stats like population and demographics in relevant_fields.
- If a query asks for data NOT reflected in the concepts above (e.g., 'education', 'crime', 'rent vs own'), set action to 'check_feasibility' and list the missing concept in 'unmatched_query_parts'.

Output Instructions:
- Output ONLY a valid JSON object: {{"action": "<action>", "relevant_fields": ["<Exact Key 1>", "<Exact Key 2>", ...], "unmatched_query_parts": ["<Missing Concept 1>", ...]}}
- Use 'relevant_fields' for the EXACT keys corresponding to the query's core concepts.
- Use 'unmatched_query_parts' ONLY for substantive data concepts that are genuinely unavailable in the list. Do NOT include common words (how, many, what, is, the, average, percent, people, of, rate, level, etc.) in unmatched_query_parts if the core concept *was* mapped.
"""
        step1_user_content = f"Analyze this user query: \"{user_query}\""

        query_analysis = {"action": "summarize", "relevant_fields": [], "unmatched_query_parts": ["Query analysis failed (default)"]}
        try:
            step1_completion = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[ {"role": "system", "content": step1_system_prompt}, {"role": "user", "content": step1_user_content}],
                temperature=0.0, max_tokens=200, response_format={"type": "json_object"} )
            step1_result_str = step1_completion.choices[0].message.content
            print(f"Step 1 Raw Result: {step1_result_str}")
            query_analysis = json.loads(step1_result_str)
            print(f"Step 1 Parsed Analysis: {query_analysis}")
        except Exception as e: print(f"Error during Step 1 API call: {e}")

        # --- Check Feasibility / Prepare for Step 2 ---
        action = query_analysis.get("action", "unknown")
        unmatched = query_analysis.get("unmatched_query_parts", [])
        relevant = query_analysis.get("relevant_fields", [])

        # Refined check: Stop only if action is clearly 'check_feasibility' or if the LLM truly couldn't map anything relevant.
        if action == "check_feasibility" or (unmatched and not relevant and action == "unknown"):
            missing_parts = unmatched if unmatched else ["unknown concepts"]
            analysis_result = f"Specific data for '{', '.join(missing_parts)}' needed for your query is not available in the calculated summary statistics."
            print("Step 1 indicated query requires unavailable data or failed mapping. Returning message.")
            return jsonify({"analysis": analysis_result})

        # --- STEP 2: Perform Data Analysis ---
        print("--- Starting Step 2: Data Analysis ---")

        # Select stats to send: relevant ones if identified, otherwise all calculated stats
        final_summary_stats_to_send = calculated_summary_stats
        if relevant:
            filtered_relevant_keys = [f for f in relevant if f in calculated_summary_stats]
            if filtered_relevant_keys:
                final_summary_stats_to_send = {k: calculated_summary_stats[k] for k in filtered_relevant_keys}
                # Always add tract count and population sum if available and not already included
                if "Number of Census Tracts Found" in calculated_summary_stats: final_summary_stats_to_send["Number of Census Tracts Found"] = calculated_summary_stats["Number of Census Tracts Found"]
                if "Total Population (Sum)" in calculated_summary_stats: final_summary_stats_to_send["Total Population (Sum)"] = calculated_summary_stats["Total Population (Sum)"]

        final_summary_data_string = json.dumps(final_summary_stats_to_send, indent=2)

        # Step 2 System Prompt (remains the same)
        step2_system_prompt = """You are an AI assistant analyzing US Census tract data summaries. Provide a concise summary addressing the user's query based *only* on the provided summary statistics. Do NOT list individual census tracts. State results directly without introductory phrases. If the statistics don't directly answer, state that based *only* on the provided stats. Format as a short paragraph or bullet points."""

        location_name = tract_id if tract_id else (county if county else (state if state else "Selected Area"))
        action_verb = query_analysis.get("action", "summarize")

        # Step 2 User Content (Include original query for context)
        step2_user_content = f"""Location Context: {location_name} ({state if state else ''}{', '+county if county else ''})
User Query: "{user_query}"
Identified Action: {action_verb}
Identified Relevant Fields by Step 1: {json.dumps(relevant)}

Available Summary Statistics for Analysis:
{final_summary_data_string}

Task: Provide a concise response answering the User Query for the Location Context using ONLY the Available Summary Statistics. Follow system prompt instructions strictly."""

        print(f"Sending summarized prompt for Step 2 to OpenAI (User Content Length: {len(step2_user_content)} chars)")

        step2_completion = client.chat.completions.create(
            model="gpt-4o-mini", # Using a potentially slightly more capable/recent model
            messages=[ {"role": "system", "content": step2_system_prompt}, {"role": "user", "content": step2_user_content}],
            temperature=0.2, max_tokens=400 )

        analysis_result = step2_completion.choices[0].message.content
        print("Received analysis from OpenAI (Step 2).")
        return jsonify({"analysis": analysis_result})

    # Main exception handler
    except Exception as e:
        print(f"--- ERROR in /analyze route ---")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

# --- Run App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)