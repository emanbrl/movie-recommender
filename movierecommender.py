"""
Movie Recommendation System
============================
Hybrid recommendation system combining:
- Neural Collaborative Filtering (Keras embeddings)
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Fast Similarity Search (FAISS)

Dataset: MovieLens (place rating.csv and movie.csv in data/ folder)
Author: Imene Bouretal
GitHub: https://github.com/emanbrl
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import faiss

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
EMBEDDING_DIM = 50
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
TOP_K_CONTENT = 10
TOP_K_FAISS = 5
RATINGS_PATH = 'data/rating.csv'
MOVIES_PATH = 'data/movie.csv'


# ─────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────
def load_data():
    """
    Loads and preprocesses the MovieLens dataset.

    Returns:
        ratings (DataFrame): User-movie interactions with encoded IDs
        movies (DataFrame): Movie metadata with cleaned genre strings
    
    Raises:
        FileNotFoundError: If CSV files are not found in data/ folder
    """
    try:
        ratings = pd.read_csv(RATINGS_PATH)
        movies = pd.read_csv(MOVIES_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset file not found: {e}. "
            "Please download MovieLens data and place rating.csv and movie.csv in the data/ folder."
        )

    # Encode user and movie IDs as integer categories
    ratings['userId'] = ratings['userId'].astype('category').cat.codes
    ratings['movieId'] = ratings['movieId'].astype('category').cat.codes

    # Clean genre strings — replace pipe separator with space for TF-IDF
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['genres'] = movies['genres'].replace('(no genres listed)', '')

    print(f"Loaded {len(ratings)} ratings from {ratings['userId'].nunique()} users "
          f"and {ratings['movieId'].nunique()} movies.")
    return ratings, movies


# ─────────────────────────────────────────
# 2. COLLABORATIVE FILTERING MODEL
# ─────────────────────────────────────────
def build_collaborative_model(n_users, n_movies):
    """
    Builds a neural collaborative filtering model using user and movie embeddings.

    Architecture:
        - User and movie IDs → Embedding layers (50-dim)
        - Embeddings concatenated → Dense(128, ReLU) → Dense(1, linear)

    Args:
        n_users (int): Total number of unique users
        n_movies (int): Total number of unique movies

    Returns:
        model (keras.Model): Compiled Keras model
    """
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(
        input_dim=n_users,
        output_dim=EMBEDDING_DIM,
        name='user_embedding'
    )(user_input)

    movie_embedding = Embedding(
        input_dim=n_movies,
        output_dim=EMBEDDING_DIM,
        name='movie_embedding'
    )(movie_input)

    user_flatten = Flatten()(user_embedding)
    movie_flatten = Flatten()(movie_embedding)

    concatenated = Concatenate()([user_flatten, movie_flatten])
    dense = Dense(128, activation='relu')(concatenated)
    output = Dense(1, activation='linear', name='output')(dense)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# ─────────────────────────────────────────
# 3. TRAIN COLLABORATIVE MODEL
# ─────────────────────────────────────────
def train_collaborative_model(model, ratings):
    """
    Trains the collaborative filtering model on user-movie rating data.

    Args:
        model (keras.Model): Compiled Keras model
        ratings (DataFrame): DataFrame with userId, movieId, rating columns

    Returns:
        model (keras.Model): Trained model
    """
    X = [ratings['userId'].values, ratings['movieId'].values]
    y = ratings['rating'].values

    print("\nTraining collaborative filtering model...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # Evaluate on validation predictions
    val_predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, val_predictions))
    print(f"\nTraining RMSE: {rmse:.4f}")

    return model


# ─────────────────────────────────────────
# 4. CONTENT-BASED FILTERING
# ─────────────────────────────────────────
def content_based_recommendations(movie_title, movies, top_k=TOP_K_CONTENT):
    """
    Returns movies similar to the given title based on genre profile.

    Uses TF-IDF vectorization on genre strings and cosine similarity
    to find the most similar movies.

    Args:
        movie_title (str): Title of the reference movie (must match dataset exactly)
        movies (DataFrame): Movies DataFrame with 'title' and 'genres' columns
        top_k (int): Number of recommendations to return (default: 10)

    Returns:
        recommendations (Series): Top-k similar movie titles

    Raises:
        ValueError: If movie_title is not found in the dataset
    """
    # Check movie exists
    matches = movies[movies['title'] == movie_title]
    if matches.empty:
        raise ValueError(
            f"Movie '{movie_title}' not found in dataset. "
            "Check the title matches exactly including the year e.g. 'Toy Story (1995)'."
        )

    # TF-IDF vectorization on genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    # Compute cosine similarity for the reference movie
    idx = matches.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort by similarity score, exclude the movie itself
    similar_indices = cosine_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_k]

    return movies['title'].iloc[similar_indices]


# ─────────────────────────────────────────
# 5. FAISS SIMILARITY SEARCH
# ─────────────────────────────────────────
def setup_faiss_index(model):
    """
    Builds a FAISS index from movie embeddings extracted from the trained model.

    FAISS enables fast approximate nearest-neighbour search across
    large embedding spaces — much faster than brute-force cosine similarity.

    Args:
        model (keras.Model): Trained collaborative filtering model

    Returns:
        index (faiss.Index): FAISS index populated with movie embeddings
    """
    movie_embeddings = model.get_layer('movie_embedding').get_weights()[0]
    movie_embeddings = np.array(movie_embeddings, dtype='float32')

    dimension = movie_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(movie_embeddings)

    print(f"\nFAISS index built with {index.ntotal} movie embeddings (dim={dimension}).")
    return index


def query_similar_movies(index, movie_id, movies, top_k=TOP_K_FAISS):
    """
    Finds movies most similar to the given movie ID using FAISS.

    Args:
        index (faiss.Index): Populated FAISS index
        movie_id (int): Encoded movie ID to query
        movies (DataFrame): Movies DataFrame
        top_k (int): Number of similar movies to return (default: 5)

    Returns:
        similar_movies (Series): Titles of top-k similar movies
    """
    query_vector = index.reconstruct(movie_id).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k + 1)

    # Exclude the query movie itself
    similar_indices = [i for i in indices[0] if i != movie_id][:top_k]
    return movies.iloc[similar_indices]['title']


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    ratings, movies = load_data()

    # Build and train collaborative filtering model
    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()
    collab_model = build_collaborative_model(n_users, n_movies)
    trained_model = train_collaborative_model(collab_model, ratings)

    # Content-based recommendations
    movie_title = "Toy Story (1995)"
    print(f"\nContent-based recommendations for '{movie_title}':")
    try:
        recommendations = content_based_recommendations(movie_title, movies)
        print(recommendations.to_string(index=False))
    except ValueError as e:
        print(f"Error: {e}")

    # FAISS similarity search
    faiss_index = setup_faiss_index(trained_model)
    movie_id = 1
    print(f"\nMovies similar to movie ID {movie_id} (FAISS):")
    print(query_similar_movies(faiss_index, movie_id, movies).to_string(index=False))
