import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="SCORE Mentor Matcher",
    page_icon="🤝",
    layout="wide"
)

st.title("🤝 SCORE Mentor Matching Tool")
st.write("Enter the client's request or skill requirements below to find the best matching mentors.")

# --- Helper Function for Medal Tiers ---
def get_medal_tier(score):
    """
    Converts similarity score (0 to 1 scale) into a medal rating tier.
    """
    if score >= 0.50:
        return "🥇 Gold Match"
    elif score >= 0.25:
        return "🥈 Silver Match"
    elif score >= 0.10:
        return "🥉 Bronze Match"
    else:
        return "No Medal Tier"

# --- Data Loading ---
@st.cache_data
def load_data():
    # Load the Excel file from the repository root
    file_path = "Chapter 24 Skill Sets.xlsx"
    df = pd.read_excel(file_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading 'Chapter 24 Skill Sets.xlsx': {e}")
    st.stop()

# --- Search Input ---
user_query = st.text_area(
    "Describe the client's needs or key business topics:",
    placeholder="e.g., Needs help with digital marketing, quickbooks accounting, and business plan formation",
    height=100
)

# --- Matching Logic ---
if st.button("Find Matching Mentors", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query or description before searching.")
    else:
        # Combine text columns to build search corpus
        # Adjust column names below if your Excel headers differ slightly
        search_corpus = df.astype(str).agg(' '.join, axis=1)

        # Vectorize and calculate similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(search_corpus)
        query_vec = vectorizer.transform([user_query])
        
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Add match scores and medal tiers to dataframe
        results_df = df.copy()
        results_df['Match Score'] = scores
        results_df['Match Tier'] = results_df['Match Score'].apply(get_medal_tier)

        # Filter out non-matching results and sort by score descending
        filtered_results = results_df[results_df['Match Score'] > 0.05].sort_values(
            by='Match Score', ascending=False
        )

        if filtered_results.empty:
            st.info("No close mentor matches found for this specific query. Try using broader keywords.")
        else:
            # Re-order columns so 'Match Tier' appears first
            cols = ['Match Tier'] + [c for c in filtered_results.columns if c not in ['Match Score', 'Match Tier']]
            display_df = filtered_results[cols]

            st.success(f"Found {len(display_df)} potential mentor matches!")
            
            # Display results table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
