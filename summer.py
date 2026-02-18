import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

st.set_page_config(page_title="Hybrid Movie Recommendation System", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System + EDA Dashboard")

# -------------------- Load & Clean Data --------------------
try:
    df = pd.read_csv("imdb_movie_data_2023.csv")
except FileNotFoundError:
    st.error("Dataset file 'imdb_movie_data_2023.csv' not found.")
    st.stop()

df = df.rename(columns={"Moive Name": "title", "Genre": "genres"})
df['movieId'] = df.index
df = df.drop_duplicates()
df['genres'] = df['genres'].fillna('Unknown')

# Simulate numerical fields
df['duration'] = np.random.randint(80, 180, size=len(df))
df['imdb_rating'] = np.round(np.random.uniform(5.0, 9.5, size=len(df)), 1)

# Extract decade
df['decade'] = df['Year'].fillna(0).astype(int)
df['decade'] = (df['decade'] // 10) * 10
df['decade'] = df['decade'].replace(0, np.nan)

# -------------------- EDA Section --------------------
st.header("📊 Exploratory Data Analysis")

st.subheader("Top 10 Genres")
genre_counts = df['genres'].str.split('|').explode().value_counts().head(10)
fig1, ax1 = plt.subplots()
sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax1)
st.pyplot(fig1)

st.subheader("Duration vs IMDb Rating")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='duration', y='imdb_rating', ax=ax2)
st.pyplot(fig2)

st.subheader("IMDb Rating Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(df['imdb_rating'], bins=20, kde=True, ax=ax3)
st.pyplot(fig3)

st.subheader("Duration Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(df['duration'], bins=20, kde=True, ax=ax4)
st.pyplot(fig4)

st.subheader("Correlation Heatmap")
fig5, ax5 = plt.subplots()
corr = df[['duration', 'imdb_rating']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
st.pyplot(fig5)

st.subheader("Movies Per Decade")
fig6, ax6 = plt.subplots()
sns.countplot(data=df[df['decade'].notna()], x='decade', ax=ax6)
plt.xticks(rotation=45)
st.pyplot(fig6)

# -------------------- Recommendation System --------------------

st.header("🤖 Hybrid Recommendation System")

users = [f"user_{i}" for i in range(1, 11)]
ratings_data = []
np.random.seed(42)

for user in users:
    rated_movies = np.random.choice(df['movieId'], size=15, replace=False)
    for mid in rated_movies:
        rating = np.random.uniform(2.5, 5.0)
        ratings_data.append([user, mid, round(rating, 1)])

ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating'])

user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_collab_recs(user_id, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    scores = item_similarity_df[rated.index].dot(rated).div(item_similarity_df[rated.index].sum(axis=1))
    scores = scores.sort_values(ascending=False)
    recommendations = scores[~scores.index.isin(rated.index)].head(n)
    return df.loc[recommendations.index][['title', 'genres']].assign(predicted_score=np.round(recommendations.values, 2))

# Content-based filtering
all_genres = list(set('|'.join(df['genres'].dropna()).replace(',', '|').split('|')))
all_genres = [g.strip() for g in all_genres if g.strip() != '']

for genre in all_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in str(x) else 0)

genre_features = df[all_genres]

def get_content_recs(user_id, n=5):
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    liked = user_movies[user_movies['rating'] >= 4.0]
    if liked.empty:
        return pd.DataFrame(columns=['title', 'genres', 'content_similarity'])
    liked_genres_matrix = genre_features.loc[liked['movieId']]
    user_profile = liked_genres_matrix.mean().values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_profile, genre_features)[0]
    df['content_similarity'] = similarity_scores
    seen = liked['movieId'].tolist()
    recommendations = df[~df['movieId'].isin(seen)].sort_values(by='content_similarity', ascending=False).head(n)
    return recommendations[['title', 'genres', 'content_similarity']]

# User Selection
selected_user = st.selectbox("Select User", users)

if st.button("Generate Recommendations"):
    collab_recs = get_collab_recs(selected_user)
    content_recs = get_content_recs(selected_user)

    st.subheader("📌 Collaborative Filtering Recommendations")
    st.dataframe(collab_recs)

    st.subheader("📌 Content-Based Filtering Recommendations")
    st.dataframe(content_recs)

    # CSV Download
    collab_csv = collab_recs.to_csv(index=False)
    content_csv = content_recs.to_csv(index=False)

    st.download_button("Download Collaborative CSV", collab_csv, file_name="collaborative.csv")
    st.download_button("Download Content-Based CSV", content_csv, file_name="content_based.csv")

st.success("✅ System Ready")
