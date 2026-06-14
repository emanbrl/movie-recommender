"""
demo.py — Quick Demo with Synthetic Data
==========================================
Tests the collaborative filtering architecture
without requiring the MovieLens dataset.

Useful for verifying the model builds and trains correctly
before running on real data.

Author: Imene Bouretal
"""

import numpy as np
import pandas as pd
from movierecommender import build_collaborative_model, train_collaborative_model

# ─────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────
np.random.seed(42)

N_SAMPLES = 1000
N_USERS = 100
N_MOVIES = 200

ratings = pd.DataFrame({
    'userId': np.random.randint(0, N_USERS, N_SAMPLES),
    'movieId': np.random.randint(0, N_MOVIES, N_SAMPLES),
    'rating': np.random.randint(1, 6, N_SAMPLES).astype(float)
})

print(f"Synthetic dataset: {N_SAMPLES} ratings, {N_USERS} users, {N_MOVIES} movies")
print(ratings.head())

# ─────────────────────────────────────────
# BUILD AND TRAIN
# ─────────────────────────────────────────
model = build_collaborative_model(n_users=N_USERS, n_movies=N_MOVIES)
model.summary()

trained_model = train_collaborative_model(model, ratings)

# ─────────────────────────────────────────
# SAMPLE PREDICTIONS
# ─────────────────────────────────────────
sample_users = np.array([0, 1, 2])
sample_movies = np.array([50, 100, 150])

predictions = trained_model.predict([sample_users, sample_movies])

print("\nSample predictions (user, movie → predicted rating):")
for u, m, p in zip(sample_users, sample_movies, predictions.flatten()):
    print(f"  User {u:>3} | Movie {m:>3} → {p:.2f}")

# ─────────────────────────────────────────
# BASIC CHECKS
# ─────────────────────────────────────────
assert predictions.shape == (3, 1), f"Unexpected output shape: {predictions.shape}"
print("\nAll checks passed. Model architecture is working correctly.")
