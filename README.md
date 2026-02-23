# plot-gpt

A Flask-based movie recommendation system powered by cosine similarity, genre vectorization, and TF-IDF weighted user profiling. Built on the MovieLens dataset with a retro CRT-style terminal UI.

---

## What it does

- Takes movies you've watched and figures out what else you'd probably like
- Uses real math (cosine similarity on genre vectors) rather than simple tag matching
- Visualizes your genre taste profile, genre distribution, movies-per-year trends, and genre co-occurrence relationships
- Supports optional genre filtering, diversity tuning, and year-proximity weighting

---

## How the recommendation algorithm works

### 1. Genre Vectors

Each movie is represented as a binary vector in genre-space. If there are 20 unique genres, every movie gets a 20-dimensional vector where `1` means it belongs to that genre and `0` means it doesn't.

```
Genres = [Action, Comedy, Drama, Thriller, ...]
"Die Hard"   → [1, 0, 0, 1, ...]
"Toy Story"  → [0, 1, 0, 0, ...]
```

The full dataset is stored as a normalized matrix so batch similarity over thousands of movies is a single matrix multiplication.

---

### 2. User Genre Profile (TF-IDF Weighted)

Your watch history is aggregated into a single genre profile vector:

- **TF (Term Frequency):** How often a genre appeared across your watched movies
- **IDF-like weighting:** Genres that are rare in the full dataset get boosted — so if you watch a lot of obscure Film-Noir, that's treated as a stronger signal than watching a lot of Action (which is everywhere)

```python
tf = count / total_movies
idf = log(total_movies / (1 + total_with_genre))
weight = tf * (1 + idf * 0.1)
```

The result is a weighted vector that captures the *strength* of your genre preferences, not just a binary yes/no.

---

### 3. Cosine Similarity

Cosine similarity measures the angle between your profile vector and each movie's genre vector:

```
similarity = (A · B) / (||A|| × ||B||)
```

- **1.0** → identical genre direction
- **0.0** → completely orthogonal (no shared genre orientation)

**Why cosine over Jaccard?**

Jaccard asks: *"Do they share genres?"* (yes/no)  
Cosine asks: *"How closely does this movie's genre mix match the direction of your preferences?"*

Cosine respects the *magnitude* of your preferences. If you mostly watch Drama with some Thriller, a Drama-heavy Thriller scores higher than a pure Thriller.

Batch computation is done via matrix multiplication against the pre-normalized genre matrix — `O(n_movies × n_genres)` but fully vectorized in NumPy, so it's fast even across tens of thousands of movies.

---

### 4. Final Score

The final recommendation score combines:

```
final_score = (cosine_score × (1 - year_weight) + year_score × year_weight) × diversity_bonus
```

| Component | Description |
|---|---|
| `cosine_score` | Genre vector similarity (primary signal) |
| `year_score` | Proximity to the average year of your watch history |
| `diversity_bonus` | Penalizes movies too similar to each other, reducing the echo chamber effect |

`diversity_factor=0.0` → pure relevance ranking  
`diversity_factor=1.0` → maximum spread across genres/styles

---

### 5. Genre Co-occurrence Matrix

Tracks how often genre pairs appear together across the dataset. Stored as a matrix where `cooccurrence[i][j]` = number of movies with both genre `i` and genre `j`. The heatmap normalizes this to conditional probabilities: `P(genre_j | genre_i)`.

---

## Project structure

```
plot-gpt/
├── app.py              # Flask app, database, recommendation engine, visualizations
├── movies.csv          # MovieLens dataset (~58k movies)
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # Retro terminal-style frontend
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `GET` | `/api/movies` | List all movies (supports search & genre filter) |
| `GET` | `/api/genres` | List all unique genres |
| `POST` | `/api/recommendations` | Get recommendations from watch history |
| `GET` | `/api/similar/<id>` | Get movies similar to a specific movie |
| `GET` | `/api/trending` | Get trending movies (recency + genre breadth) |
| `GET` | `/api/viz/genre-distribution` | Bar chart of genre distribution |
| `GET` | `/api/viz/movies-per-year` | Line chart of movies released per year |
| `GET` | `/api/viz/genre-cooccurrence` | Genre co-occurrence heatmap |
| `POST` | `/api/viz/genre-profile` | Your personal genre preference chart |

### Recommendation request body

```json
{
  "watched_ids": [1, 2, 3],
  "preferred_genres": ["Drama", "Thriller"],
  "n": 15,
  "diversity": 0.3,
  "year_weight": 0.2
}
```

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/Sukanth19/plot-gpt.git
cd plot-gpt
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Web server and routing |
| pandas | Data loading and cleaning |
| numpy | Vector math and matrix operations |
| matplotlib | Chart rendering |
| seaborn | Heatmaps and styled plots |

---

## Dataset

Uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (`movies.csv`). Movies without any listed genre are excluded at load time.
