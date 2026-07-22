import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="SCORE Chapter 24 - Mentor Matcher",
    page_icon="🤝",
    layout="wide",
)

st.title("🤝 SCORE Chapter 24 Client Intake & Mentor Matcher")
st.markdown(
    "Enter the client's business background, industry, or specific challenge below to find the top 3 best-suited mentors."
)


@st.cache_data
def load_data():
    df = pd.read_excel("Chapter 24 Skill Sets.xlsx")
    # Combine relevant text fields for similarity matching
    df["Combined_Text"] = (
        df["Skill"].fillna("")
        + " "
        + df["Experience"].fillna("")
        + " "
        + df["Profile"].fillna("")
    )
    return df


df = load_data()

# User Input
client_needs = st.text_area(
    "Client Request / Business Background:",
    height=120,
    placeholder="Example: Client wants to launch an artisan bakery. Needs assistance with business planning, local marketing, lease negotiation, and financial forecasting.",
)

if st.button("Find Top Mentor Matches", type="primary"):
    if not client_needs.strip():
        st.warning(
            "Please enter a client request or business description to match."
        )
    else:
        # TF-IDF Matching Engine
        corpus = df["Combined_Text"].tolist()
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)

        query_vec = vectorizer.transform([client_needs])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Get top 3 indices
        top_indices = scores.argsort()[-3:][::-1]

        st.subheader("🎯 Recommended Mentor Matches")

        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            match_score = round(scores[idx] * 100, 1)

            results.append(
                {
                    "Mentor Last Name": row["Last_Name"],
                    "Match Relevancy": f"{match_score}%",
                    "Core Skills": row["Skill"],
                    "Industry Experience": row["Experience"],
                    "Background Summary": row["Profile"],
                }
            )

        results_df = pd.DataFrame(results)

        # Display formatted table
        st.dataframe(results_df, use_container_width=True)
