# 🎬 Movie Recommendation System

A hybrid movie recommendation system combining **Collaborative Filtering** (Neural Network-based) and **Content-Based Filtering** (TF-IDF + Cosine Similarity), with **FAISS** for fast approximate nearest-neighbour search on movie embeddings.

Built on the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

---

## 📌 Project Overview

This project implements three complementary recommendation approaches:

| Approach | Method | Description |
|---|---|---|
| Collaborative Filtering | Neural Network (Keras) | Learns user-movie interaction patterns via embedding layers |
| Content-Based Filtering | TF-IDF + Cosine Similarity | Recommends movies with similar genre profiles |
| Fast Similarity Search | FAISS (Facebook AI) | Efficient nearest-neighbour retrieval on learned embeddings |

---

## 🗂️ Project Structure

```
movie-recommender/
│
├── movierecommender.py   # Main system: data loading, model building, recommendations
├── demo.py               # Lightweight collaborative filtering test with synthetic data
├── requirements.txt      # Python dependencies
├── data/                 # Place MovieLens CSV files here (not included)
│   ├── rating.csv
│   └── movie.csv
└── README.md
```

---

## ⚙️ How It Works

### 1. Collaborative Filtering (Neural Network)
- User and movie IDs are encoded as embeddings (50-dimensional vectors)
- Embeddings are concatenated and passed through dense layers
- Model learns to predict ratings from user-movie interaction patterns
- Trained using Adam optimizer with MSE loss

### 2. Content-Based Filtering
- Movie genres are vectorized using TF-IDF
- Cosine similarity matrix computed across all movies
- Given a movie title, returns the top 10 most similar movies by genre profile

### 3. FAISS Similarity Search
- Movie embeddings extracted from the trained neural network
- Indexed using FAISS IndexFlatL2 for fast L2-distance search
- Enables scalable nearest-neighbour retrieval across large movie catalogs

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/emanbrl/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the MovieLens dataset
Download from [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/) and place `rating.csv` and `movie.csv` inside the `data/` folder.

### 4. Run the main system
```bash
python movierecommender.py
```

### 5. Run the test script
```bash
python test.py
```

---

## 📊 Example Output

```
Content-based recommendations for 'Toy Story (1995)':
- Antz (1998)
- Toy Story 2 (1999)
- Adventures of Rocky and Bullwinkle, The (2000)
- Emperor's New Groove, The (2000)
- Monsters, Inc. (2001)
...

Movies similar to ID 1 (FAISS):
- Movie A
- Movie B
- Movie C
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** — neural collaborative filtering
- **Scikit-learn** — TF-IDF vectorization, cosine similarity
- **FAISS** — fast embedding similarity search
- **Pandas / NumPy** — data processing

---

## 📁 Dataset

This project uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset by GroupLens Research. Dataset files are not included in this repository due to size. Download and place in the `data/` folder before running.

---

## 👩‍💻 Author

**Imene Bouretal**
MSc AI Engineering — Istinye University Istanbul
[GitHub](https://github.com/emanbrl) | [LinkedIn](https://www.linkedin.com/in/imene-b-52b364197/) | [Medium](https://medium.com/@imeneb)

---

## 📄 License

This project is licensed under the MIT License.
