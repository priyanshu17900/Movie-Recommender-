import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ------------------ Custom styling ------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0f172a;
    }
    .main {
        background: linear-gradient(135deg, #020617, #020617);
        color: white;
    }
    .card {
        background-color: #020617;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load data (CACHED) ------------------
@st.cache_resource
def load_data():
    movies = pickle.load(open("movie_list.pkl", "rb"))
    vectors = pickle.load(open("vectors.pkl", "rb"))
    movie_to_index = {m: i for i, m in enumerate(movies)}
    return movies, vectors, movie_to_index

movies, vectors, movie_to_index = load_data()

# ------------------ Title ------------------
st.markdown("<h1 style='text-align:center;'>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>"
    "Find movies with similar tone, genre, and themes — not random noise."
    "</p>",
    unsafe_allow_html=True
)

st.write("")

# ------------------ Card UI ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

movie_name = st.selectbox(
    "Select a movie",
    movies,
    help="Start typing to search"
)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Recommendation logic (OPTIMIZED) ------------------
def recommend(movie, top_n=5):
    idx = movie_to_index[movie]
    scores = cosine_similarity(vectors[idx], vectors).ravel()
    top_indices = np.argpartition(scores, -top_n-1)[-top_n-1:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    top_indices = top_indices[1:top_n+1]
    return [movies[i] for i in top_indices]

# ------------------ Button ------------------
if st.button("🔍 Recommend", use_container_width=True):
    with st.spinner("Analyzing movies..."):
        results = recommend(movie_name)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recommended Movies")

    for m in results:
        st.markdown(f"• **{m}**")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Footer ------------------
st.write("")
st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:0.8rem;'>"
    "Powered by TF-IDF similarity • Python-only frontend"
    "</p>",
    unsafe_allow_html=True
)
