import requests
import time
import random
import csv
import os
from datetime import datetime, timedelta
import atexit # Import atexit to save on exit
import json # Import json for safe response parsing
import sys

# --- Try importing us library, install if missing ---
try:
    import us
except ImportError:
    print("Installing required 'us' package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "us"])
    import us

# --- Configuration ---
SERVER_URL = "http://localhost:5000/analyze"  # CORRECTED endpoint
# Assumes trainingqueries.csv is inside a 'trainingdata' subdirectory
# If it's in the same directory as the script, change to: "trainingqueries.csv"
QUERY_CSV_PATH = os.path.join(os.path.dirname(__file__), "trainingdata", "trainingqueries.csv")

# --- Configure Log Directory and File ---
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
    print(f"Created logs directory at {LOGS_DIR}")

# Create log filename with current date (YYYYMMDD format)
current_date = datetime.now().strftime("%Y%m%d")
RESULTS_LOG_FILE = os.path.join(LOGS_DIR, f"client_results_{current_date}.csv")

RUNTIME_HOURS = 3 # Changed to 3 hours as per your earlier question, adjust if needed
REQUEST_INTERVAL_SEC = 3  # Time between requests
SAVE_INTERVAL_REQUESTS = 50 # Save results every 50 requests

# --- Generate realistic state/county pairs ---
def generate_state_county_pairs():
    """Generate valid state and county pairs using the us library"""
    try:
        # Get all states
        states = us.states.STATES
        
        # Create list of (state, county) pairs
        pairs = []
        state_count = min(len(states), 25)  # Limit to 25 states
        
        # Select a subset of states
        selected_states = random.sample(list(states), state_count)
        
        for state in selected_states:
            # The state.fips is the Federal Information Processing Standard state code
            # Use this to fetch counties when available
            state_name = state.name
            
            # For now, add a placeholder county for each state
            # In a real implementation, you would fetch actual counties by state
            pairs.append((state_name, ""))
            
        return pairs
        
    except Exception as e:
        print(f"Error generating state/county pairs: {e}")
        # Return the original hardcoded list as fallback
        return SAMPLE_LOCATIONS

# --- Use us library to get state/county pairs ---
STATE_COUNTY_PAIRS = generate_state_county_pairs()

# --- Global list to buffer results ---
results_buffer = []

# --- CSV Headers for Result Log ---
RESULTS_LOG_HEADERS = [
    'timestamp', 'request_num', 'input_state', 'input_county', 'input_query',
    'request_status', 'response_status_code', 'response_content'
]

def load_training_queries(filepath):
    """Load training queries from the CSV file"""
    queries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            if 'query_text' not in csv_reader.fieldnames:
                 print(f"ERROR: CSV file '{filepath}' must contain a 'query_text' header.")
                 return []
            for row in csv_reader:
                query = row['query_text'].strip('"').strip()
                if query: queries.append(query)
        print(f"Loaded {len(queries)} training queries from '{filepath}'")
    except FileNotFoundError:
        print(f"ERROR: Query file not found at '{filepath}'. Check path/directory structure.")
        queries = [] # Return empty, main will exit
    except Exception as e:
        print(f"Error loading training data from '{filepath}': {e}")
        queries = []
    return queries

def save_results_to_csv(results_to_save, filepath):
    """Appends buffered results to the CSV log file and clears buffer."""
    if not results_to_save:
        return # Nothing to save

    print(f"\n--- Saving {len(results_to_save)} results to {filepath} ---")
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=RESULTS_LOG_HEADERS)
            if not file_exists:
                writer.writeheader()
            for row_data in results_to_save:
                # Ensure all headers exist, default missing to empty string
                row_to_write = {header: row_data.get(header, '') for header in RESULTS_LOG_HEADERS}
                writer.writerow(row_to_write)
        results_to_save.clear() # Clear buffer after successful save
        print("--- Results saved successfully ---")
    except Exception as e:
        print(f"--- ERROR writing results to log file {filepath}: {e} ---")
        print("--- Results remain in buffer ---")

# Register the save function to run automatically on script exit
# This handles normal exit and Ctrl+C (KeyboardInterrupt)
def on_exit():
    """Function to run when script exits - save remaining results and print summary"""
    save_results_to_csv(results_buffer, RESULTS_LOG_FILE)
    print(f"\nScript terminated. Final results saved to {RESULTS_LOG_FILE}")

atexit.register(on_exit)

def main():
    """Main function to run the dummy client"""
    # Load training queries
    training_queries = load_training_queries(QUERY_CSV_PATH)
    if not training_queries:
        print("No training queries found. Exiting.")
        return
        
    print(f"Using {len(STATE_COUNTY_PAIRS)} different state/county combinations")
    
    # Set end time
    end_time = datetime.now() + timedelta(hours=RUNTIME_HOURS)
    total_requests = 0
    successful_requests = 0
    
    print(f"Starting dummy client at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will run until {end_time.strftime('%Y-%m-%d %H:%M:%S')} (approximately {RUNTIME_HOURS} hours)")
    
    try:
        while datetime.now() < end_time:
            # Select a random query and location
            query = random.choice(training_queries)
            state, county = random.choice(STATE_COUNTY_PAIRS)
            
            # Format state and county names
            state_formatted = state.lower() if state else ""
            county_formatted = county.lower() if county else ""
            
            # Prepare request data
            request_data = {
                "state": state_formatted,
                "county": county_formatted,
                "tract_id": "",
                "query": query
            }
            
            # Prepare log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_requests += 1
            log_entry = {
                'timestamp': timestamp,
                'request_num': total_requests,
                'input_state': state_formatted,
                'input_county': county_formatted,
                'input_query': query
            }
            
            try:
                # Send request to server
                response = requests.post(
                    SERVER_URL,
                    json=request_data,
                    timeout=30
                )
                
                # Log request result
                log_entry['request_status'] = 'success' if response.status_code == 200 else 'failed'
                log_entry['response_status_code'] = response.status_code
                
                # Try to parse and log response content safely
                try:
                    if response.status_code == 200:
                        successful_requests += 1
                        resp_content = response.json()
                        log_entry['response_content'] = str(resp_content)[:500]  # Truncate if too long
                    else:
                        log_entry['response_content'] = response.text[:500]  # Truncate if too long
                except:
                    log_entry['response_content'] = "Error parsing response"
                
                # Print status message
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {total_requests}: '{query[:40]}...' - {'Success' if response.status_code == 200 else f'Failed ({response.status_code})'}")
                
            except requests.exceptions.RequestException as e:
                # Handle request errors
                log_entry['request_status'] = 'error'
                log_entry['response_status_code'] = 'N/A'
                log_entry['response_content'] = str(e)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request error: {e}")
            
            # Add entry to buffer
            results_buffer.append(log_entry)
            
            # Periodically save results
            if len(results_buffer) >= SAVE_INTERVAL_REQUESTS:
                save_results_to_csv(results_buffer, RESULTS_LOG_FILE)
            
            # Wait before next request
            time.sleep(REQUEST_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    
    # Print summary stats
    runtime = datetime.now() - (end_time - timedelta(hours=RUNTIME_HOURS))
    hours, remainder = divmod(runtime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n--- Training Session Summary ---")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {successful_requests/total_requests*100:.2f}% (if not 100%, check server logs)")
    print(f"Average requests per minute: {total_requests/(runtime.total_seconds()/60):.2f}")

if __name__ == "__main__":
    main()