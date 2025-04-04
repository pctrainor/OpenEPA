import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from collections import Counter
import glob
import re
from datetime import datetime

def load_logs(logs_dir='logs'):
    """Load all query logs into a pandas DataFrame"""
    log_files = glob.glob(os.path.join(logs_dir, 'query_logs_*.jsonl'))
    
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return None
    
    all_logs = []
    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        all_logs.append(log_entry)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_logs:
        print("No valid log entries found")
        return None
    
    df = pd.DataFrame(all_logs)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} log entries from {len(log_files)} files")
    return df

def analyze_queries(df):
    """Analyze user queries and responses"""
    if df is None or df.empty:
        print("No data to analyze")
        return
    
    # Basic statistics
    total_queries = len(df)
    errors = df[df['error'].notnull()].shape[0]
    unique_states = df['state'].nunique()
    unique_counties = df['county'].nunique()
    
    print(f"\n===== Query Log Analysis =====")
    print(f"Total queries: {total_queries}")
    print(f"Error rate: {errors/total_queries:.2%}")
    print(f"Unique states: {unique_states}")
    print(f"Unique counties: {unique_counties}")
    
    # Query complexity analysis (word count as a simple proxy)
    df['query_length'] = df['user_query'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    print(f"\n===== Query Complexity =====")
    print(f"Average words per query: {df['query_length'].mean():.2f}")
    print(f"Median words per query: {df['query_length'].median()}")
    print(f"Max words: {df['query_length'].max()}")
    
    # Topic analysis based on query
    print("\n===== Query Topics =====")
    # Extract common themes
    demographic_pattern = re.compile(r'population|hispanic|latino|black|african|white|race|racial|ethnic', re.IGNORECASE)
    economic_pattern = re.compile(r'poor|poverty|income|low-income|disadvantaged|fpl', re.IGNORECASE)
    health_pattern = re.compile(r'asthma|diabetes|health', re.IGNORECASE)
    
    df['has_demographic'] = df['user_query'].apply(lambda x: bool(demographic_pattern.search(str(x))) if pd.notnull(x) else False)
    df['has_economic'] = df['user_query'].apply(lambda x: bool(economic_pattern.search(str(x))) if pd.notnull(x) else False)
    df['has_health'] = df['user_query'].apply(lambda x: bool(health_pattern.search(str(x))) if pd.notnull(x) else False)
    
    print(f"Demographic queries: {df['has_demographic'].sum()} ({df['has_demographic'].sum()/total_queries:.2%})")
    print(f"Economic queries: {df['has_economic'].sum()} ({df['has_economic'].sum()/total_queries:.2%})")
    print(f"Health queries: {df['has_health'].sum()} ({df['has_health'].sum()/total_queries:.2%})")
    
    # Success rate of query interpretation
    has_relevant_fields = df['query_analysis'].apply(
        lambda x: len(x.get('relevant_fields', [])) > 0 if isinstance(x, dict) else False
    )
    print(f"\n===== Query Understanding =====")
    print(f"Queries with relevant fields matched: {has_relevant_fields.sum()} ({has_relevant_fields.sum()/total_queries:.2%})")
    
    # Plot query lengths
    plt.figure(figsize=(10, 6))
    plt.hist(df['query_length'], bins=10, alpha=0.7, color='blue')
    plt.title('Distribution of Query Lengths')
    plt.xlabel('Number of Words in Query')
    plt.ylabel('Frequency')
    plt.axvline(df['query_length'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.savefig('query_length_distribution.png')
    print("Query length distribution saved as 'query_length_distribution.png'")
    
    # Records found analysis
    df['records_found'] = pd.to_numeric(df['records_found'], errors='coerce').fillna(0)
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['records_found'], bins=20, alpha=0.7, color='green')
    plt.title('Distribution of Records Found')
    plt.xlabel('Number of Records')
    plt.ylabel('Frequency')
    plt.axvline(df['records_found'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.savefig('records_distribution.png')
    print("Records found distribution saved as 'records_distribution.png'")
    
    # Most common states and counties
    top_states = df['state'].value_counts().head(10)
    top_counties = df['county'].value_counts().head(10)
    
    print("\n===== Most Queried Locations =====")
    print("Top States:")
    for state, count in top_states.items():
        print(f"  {state}: {count}")
        
    print("\nTop Counties:")
    for county, count in top_counties.items():
        print(f"  {county}: {count}")

def perform_anova_on_logs(df):
    """Perform ANOVA analysis on log data"""
    if df is None or df.empty:
        print("No data to analyze with ANOVA")
        return
    
    print("\n===== ANOVA Analysis =====")
    
    # Convert records_found to numeric
    df['records_found'] = pd.to_numeric(df['records_found'], errors='coerce').fillna(0)
    
    # Example 1: ANOVA on query complexity by query type
    query_types = []
    for i, row in df.iterrows():
        if row['has_demographic']:
            query_types.append('demographic')
        elif row['has_economic']:
            query_types.append('economic')
        elif row['has_health']:
            query_types.append('health')
        else:
            query_types.append('other')
    
    df['query_type'] = query_types
    
    # ANOVA: query complexity by query type
    groups = [df[df['query_type'] == t]['query_length'] for t in df['query_type'].unique()]
    group_names = df['query_type'].unique()
    
    if len(groups) > 1:  # Need at least two groups for ANOVA
        f_val, p_val = stats.f_oneway(*groups)
        print(f"ANOVA: Query length by query type")
        print(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")
        print(f"Statistically significant: {'Yes' if p_val < 0.05 else 'No'}")
        
        # Group means
        print("\nQuery length by type:")
        for name, group in zip(group_names, groups):
            print(f"  {name}: {group.mean():.2f} words")
        
        # Visualize group differences
        plt.figure(figsize=(10, 6))
        plt.boxplot([g for g in groups if len(g) > 0], 
                   labels=[n for n, g in zip(group_names, groups) if len(g) > 0])
        plt.title('Query Length by Query Type')
        plt.ylabel('Number of Words')
        plt.grid(True, alpha=0.3)
        plt.savefig('query_length_by_type.png')
        print("Query length by type saved as 'query_length_by_type.png'")
    else:
        print("Not enough different query types for ANOVA")

    # Example 2: Compare records found between different states (if enough data)
    top_states = df['state'].value_counts().head(5).index.tolist()
    if len(top_states) > 1:
        state_groups = [df[df['state'] == state]['records_found'] for state in top_states]
        state_groups = [g for g in state_groups if len(g) >= 3]  # Need some minimum sample size
        
        if len(state_groups) > 1:
            f_val, p_val = stats.f_oneway(*state_groups)
            print(f"\nANOVA: Records found by state")
            print(f"F-value: {f_val:.4f}, p-value: {p_val:.4f}")
            print(f"Statistically significant: {'Yes' if p_val < 0.05 else 'No'}")
            
            # State means
            print("\nRecords found by state:")
            valid_states = [state for state, g in zip(top_states, state_groups) if len(g) >= 3]
            for state, group in zip(valid_states, state_groups):
                print(f"  {state}: {group.mean():.2f} records")
        else:
            print("\nNot enough data per state for ANOVA")

def extract_response_patterns(df):
    """Analyze patterns in AI responses"""
    if df is None or df.empty:
        return
    
    print("\n===== Response Analysis =====")
    
    # Response length analysis
    df['response_length'] = df['response'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    print(f"Average response length: {df['response_length'].mean():.2f} words")
    
    # Check for bullet points vs paragraphs
    bullet_pattern = re.compile(r'^[-•*]', re.MULTILINE)
    df['has_bullets'] = df['response'].apply(
        lambda x: bool(bullet_pattern.search(str(x))) if pd.notnull(x) else False
    )
    print(f"Responses with bullet points: {df['has_bullets'].sum()} ({df['has_bullets'].sum()/len(df):.2%})")
    
    # Correlate query complexity with response length
    correlation = np.corrcoef(df['query_length'], df['response_length'])[0, 1]
    print(f"Correlation between query length and response length: {correlation:.4f}")

if __name__ == "__main__":
    print("OpenEPA Query Log Analyzer")
    print("=========================")
    
    # Load log data
    logs_df = load_logs()
    
    if logs_df is not None:
        analyze_queries(logs_df)
        perform_anova_on_logs(logs_df)
        extract_response_patterns(logs_df)
        
        # Save the processed DataFrame as CSV for further analysis
        logs_df.to_csv('processed_logs.csv', index=False)
        print("\nProcessed logs saved to 'processed_logs.csv'")
    else:
        print("\nNo data to analyze. Make sure you have query log files in the 'logs' directory.")