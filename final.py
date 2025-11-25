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

# -------------------- Load & Clean Data --------------------
df = pd.read_csv(r"C:\Users\91798\Downloads\imdb_movie_data_2023.csv")
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

# -------------------- EDA PDF --------------------
def save_plot(path, plot_func):
    plot_func()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

os.makedirs("plots", exist_ok=True)

def plot_top_genres():
    genre_counts = df['genres'].str.split('|').explode().value_counts().head(10)
    genre_labels = genre_counts.index
    sns.barplot(
        x=genre_counts.values,
        y=genre_labels,
        hue=genre_labels,
        dodge=False,
        palette="viridis",
        legend=False
    )
    plt.title("Top 10 Genres by Frequency")
    plt.xlabel("Count")
    plt.ylabel("Genre")
save_plot("plots/top_genres.png", plot_top_genres)

def plot_scatter():
    sns.scatterplot(data=df, x='duration', y='imdb_rating', alpha=0.7)
    plt.title("Movie Duration vs IMDb Rating")
    plt.xlabel("Duration (min)")
    plt.ylabel("IMDb Rating")
save_plot("plots/scatter_duration_rating.png", plot_scatter)

def plot_hist_rating():
    sns.histplot(df['imdb_rating'], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of IMDb Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
save_plot("plots/hist_rating.png", plot_hist_rating)

def plot_hist_duration():
    sns.histplot(df['duration'], bins=20, kde=True, color='salmon')
    plt.title("Distribution of Duration")
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
save_plot("plots/hist_duration.png", plot_hist_duration)

def plot_corr():
    corr = df[['duration', 'imdb_rating']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
save_plot("plots/heatmap.png", plot_corr)

def plot_decade():
    data = df[df['decade'].notna()]
    sns.countplot(data=data, x='decade', hue='decade', palette='cubehelix', legend=False)
    plt.title("Movies Per Decade")
    plt.xlabel("Decade")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
save_plot("plots/movies_decade.png", plot_decade)

def generate_eda_pdf(filename="EDA_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("\U0001F3AC IMDb Movie Data - EDA Report", styles["Title"]), Spacer(1, 12)]

    summary_text = f"""
    <b>Total Movies:</b> {len(df)}<br/>
    <b>Unique Genres:</b> {len(set('|'.join(df['genres']).replace(',', '|').split('|')))}<br/>
    """
    elements.append(Paragraph(summary_text.replace("\n", "<br/>"), styles["Normal"]))
    elements.append(Spacer(1, 12))

    plots = [
        ("Top 10 Genres", "plots/top_genres.png"),
        ("Duration vs IMDb Rating", "plots/scatter_duration_rating.png"),
        ("Histogram of IMDb Ratings", "plots/hist_rating.png"),
        ("Histogram of Movie Duration", "plots/hist_duration.png"),
        ("Correlation Heatmap", "plots/heatmap.png"),
        ("Movies per Decade", "plots/movies_decade.png")
    ]

    for title, img in plots:
        elements.append(Paragraph(f"\U0001F4CA {title}", styles["Heading2"]))
        elements.append(Image(img, width=400, height=250))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"\n✅ EDA PDF Report saved: {filename}")

# -------------------- Recommendation System --------------------
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

def generate_recommendation_pdf(user_id, collab_df, content_df, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph(f"\U0001F3AC Movie Recommendation Report for {user_id}", styles["Title"]), Spacer(1, 12)]

    elements.append(Paragraph("\U0001F4CC Collaborative Filtering Recommendations:", styles["Heading2"]))
    collab_data = [collab_df.columns.tolist()] + collab_df.values.tolist()
    collab_table = Table(collab_data, repeatRows=1)
    collab_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(collab_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("\U0001F4CC Content-Based Filtering Recommendations:", styles["Heading2"]))
    content_data = [content_df.columns.tolist()] + content_df.values.tolist()
    content_table = Table(content_data, repeatRows=1)
    content_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(content_table)

    doc.build(elements)
    print(f"✅ PDF report saved: {filename}")

# ---------- Run ----------
example_user = 'user_1'
collab_recs = get_collab_recs(example_user)
content_recs = get_content_recs(example_user)

collab_recs.to_csv("collaborative_recommendations.csv", index=False)
content_recs.to_csv("content_based_recommendations.csv", index=False)
print("✅ CSV files exported successfully.")

generate_recommendation_pdf(example_user, collab_recs, content_recs, filename=f"Recommendation_Report_{example_user}.pdf")
generate_eda_pdf()
