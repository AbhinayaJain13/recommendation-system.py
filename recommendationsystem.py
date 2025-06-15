import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = pd.DataFrame({
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Prestige', 'The Dark Knight'],
    'genre': ['Sci-Fi, Action', 'Sci-Fi, Action', 'Sci-Fi, Drama', 'Drama, Mystery', 'Action, Crime']
})

# TF-IDF vectorization of genre
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genre'])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies based on title
def recommend(title, movies, cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]  # Get top 2 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example usage
print(recommend('Inception', movies, cosine_sim))
