import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation JSON from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Movie-themed Lottie animation URL (you can change this)
lottie_animation = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_sSF6EG.json")

# Inject custom CSS styles
st.markdown(
    """
    <style>
    /* Background color for the whole app */
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Style the title */
    .css-18e3th9 {
        color: #2a9d8f;
        font-weight: 700;
        font-size: 40px;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Style buttons */
    button[kind="primary"] {
        background-color: #264653 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #2a9d8f !important;
    }

    /* Style dropdowns */
    .stSelectbox > div {
        background-color: #e9ecef;
        border-radius: 6px;
        padding: 5px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

# Display animation at the top
if lottie_animation:
    st_lottie(lottie_animation, speed=1, width=200, height=200, key="movie")

st.title("🎬 Movie Rating Predictor with ML")

@st.cache_data
def load_data():
    df = pd.read_csv("imdb_top_1000.csv")
    df['Genre'] = df['Genre'].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown')
    df.dropna(subset=['Genre', 'Director', 'Actor 1', 'Rating'], inplace=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Load data and model
df = load_data()
model = load_model()

st.markdown("## Select movie features to predict rating")

col1, col2, col3 = st.columns(3)

with col1:
    selected_genre = st.selectbox("Select Genre", sorted(df['Genre'].unique()))
with col2:
    selected_director = st.selectbox("Select Director", sorted(df['Director'].unique()))
with col3:
    selected_actor = st.selectbox("Select Lead Actor (Actor 1)", sorted(df['Actor 1'].unique()))

# Prepare input for prediction
input_df = pd.DataFrame({
    'Genre': [selected_genre],
    'Director': [selected_director],
    'Actor 1': [selected_actor]
})

if st.button("Predict Rating"):
    predicted_rating = model.predict(input_df)[0]
    st.success(f"🎯 Predicted IMDb Rating: {predicted_rating:.2f}")
else:
    st.info("Select features and click 'Predict Rating'")
