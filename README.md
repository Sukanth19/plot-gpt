# plot-gpt

A Flask-based movie recommendation system with a retro CRT-style terminal UI. It starts as a strong content-based recommender (cosine similarity over genre vectors + TF‑IDF-like user profiling) and is designed to evolve into a hybrid system (content + collaborative filtering) using MovieLens ratings.

---

## Features

- Interactive web UI to:
  - search movies and build a watched list
  - optionally filter by genres
  - tune diversity and result count
- Content-based recommendations using:
  - genre vectorization (movie → genre-space vector)
  - TF‑IDF-like weighted user profiles (watch history → preference vector)
  - cosine similarity computed efficiently with NumPy
- Dataset analytics + debugging visualizations (Matplotlib/Seaborn):
  - genre distribution
  - movies per year
  - genre co-occurrence heatmap
  - personal genre profile chart

---

## How it works (high level)

This project uses a content-based approach (and prepares for hybridization later):

1. Convert each movie's genres into a vector (one dimension per genre).
2. Build a user preference profile from watched movies using weighted genre frequency.
3. Rank all unseen movies by cosine similarity to the user profile.
4. Optionally blend in release-year proximity and a diversity factor.

This approach is fast, explainable, and works even when you don’t have ratings for the current user.

---

## Recommendation algorithm (details)

### 1) Genre vectors

Each movie is represented as a binary vector in “genre space”. If there are `G` unique genres in the dataset, then every movie becomes a `G`-dimensional vector where:

- `1` = movie has that genre
- `0` = movie does not

Example:

```
Genres = [Action, Comedy, Drama, Thriller, ...]
"Die Hard"  -> [1, 0, 0, 1, ...]
"Toy Story" -> [0, 1, 0, 0, ...]
```

All movie vectors are stored in a matrix `M` (shape: `n_movies x n_genres`) and pre-normalized so that cosine similarity can be computed quickly using matrix multiplication.

---

### 2) User genre profile (TF‑IDF-like weighting)

A user profile is built from watched movies. Genres that appear frequently in your watched list get a higher weight (TF). Genres that are extremely common across the whole dataset are slightly down-weighted via an IDF-like penalty, so rare but meaningful preferences (e.g., Film-Noir) stand out more.

Core idea:

```python
tf = count / total_watched
idf = log(total_movies / (1 + movies_with_genre))
weight = tf * (1 + idf * 0.1)
```

The result is a weighted preference vector representing the “direction” of your taste in genre space.

---

### 3) Cosine similarity

Cosine similarity measures how aligned two vectors are:

```
similarity = (A · B) / (||A|| * ||B||)
```

Interpretation:

- `1.0` -> very similar genre direction
- `0.0` -> unrelated genre direction

Why cosine instead of Jaccard:

- Jaccard is set-based: “do they overlap?”
- Cosine is vector-based: “how strongly do they align with your preference distribution?”

In practice, cosine is smoother and usually ranks better when your profile has different preference strengths across multiple genres.

---

### 4) Final scoring (relevance + year + diversity)

The final score combines:

- `cosine_score`: primary relevance signal
- `year_score`: mild preference for movies close to the average year of watched movies
- `diversity_bonus`: reduces “echo chamber” recommendations when you want a wider spread

One version of the scoring logic:

```
final_score = (cosine_score * (1 - year_weight) + year_score * year_weight) * diversity_bonus
```

Tuning:

- `diversity_factor = 0.0` means “very similar recommendations”
- `diversity_factor = 1.0` means “more variety”
- `year_weight` controls how much release-year proximity affects ranking

---

### 5) Genre co-occurrence matrix (analysis)

A co-occurrence matrix counts how often two genres appear together in the dataset:

- `cooccurrence[i][j]` = number of movies containing both genres `i` and `j`

The heatmap visualization often reveals clusters like:
- Action <-> Thriller
- Romance <-> Drama
- Animation <-> Children

This helps you understand the dataset and can guide future algorithm improvements.

---

## Project structure

```
plot-gpt/
├── app.py              # Flask app + recommendation engine + analytics routes
├── movies.csv          # MovieLens movies dataset
├── ratings.csv         # MovieLens ratings dataset (optional but recommended)
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # CRT-style frontend
```

Notes:
- If you don’t want to commit large CSVs, add them to `.gitignore` and document how to download them.
- Keeping `ratings.csv` available enables collaborative filtering upgrades.

---

## API endpoints

### Core

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend |
| GET | `/api/movies/search?q=...&limit=...` | Search movies by title |
| GET | `/api/movies/<movie_id>` | Get a movie by ID |
| GET | `/api/genres` | List all unique genres |
| POST | `/api/recommendations` | Get recommendations |
| GET | `/api/similar/<movie_id>` | Recommendations seeded by one movie |
| GET | `/api/trending?genre=...&limit=...` | Simple trending list |
| GET | `/api/stats` | Dataset statistics |

### Visualizations (PNG-as-base64 JSON payload)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/viz/genre-distribution` | Bar chart of genre counts |
| GET | `/api/viz/movies-per-year` | Movies released per year |
| GET | `/api/viz/genre-cooccurrence` | Genre co-occurrence heatmap |
| POST | `/api/viz/genre-profile` | User genre profile chart |

---

## Recommendation request format

Example request body:

```json
{
  "watched_ids": [1, 2, 3],
  "preferred_genres": ["Drama", "Thriller"],
  "n": 15,
  "diversity": 0.3,
  "year_weight": 0.2
}
```

- `watched_ids` (required): list of movieIds
- `preferred_genres` (optional): if provided, overrides inferred profile
- `n`: number of recommendations
- `diversity`: 0 to 1
- `year_weight`: 0 to 1

---

## Setup

Requirements: Python 3.9+ recommended

```bash
git clone https://github.com/Sukanth19/plot-gpt.git
cd plot-gpt

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open:

```
http://localhost:5000
```

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Web server + routing |
| pandas | CSV loading + cleaning |
| numpy | Vector math + fast matrix operations |
| matplotlib | Chart rendering (server-side) |
| seaborn | Styled plots + heatmaps |

Optional / upcoming (for collaborative filtering):
- scikit-learn / scipy (SVD + sparse matrices)

---

## Dataset

This project uses the MovieLens dataset format:

- `movies.csv`: `movieId,title,genres`
- `ratings.csv`: `userId,movieId,rating,timestamp`

Movies with `(no genres listed)` are excluded when building genre-based features.

---

## Roadmap (planned upgrades)

Text-only “emojis” included for style:

- [ ] Hybrid recommendations: content + collaborative filtering (SVD)
- [ ] Evaluation metrics: Precision@K, Recall@K, NDCG, coverage, diversity
- [ ] Explainability: show "why recommended" (genre overlap + CF similarity)
- [ ] Cache + performance: avoid repeated computations during requests
- [ ] UI: charts panel + model diagnostics inside the app

---

## Notes on reproducibility

If you install new packages during development:

```bash
pip freeze > requirements.txt
```

Then commit the updated `requirements.txt` so the environment can be reproduced on another machine.

---

## License

Add a license if you plan to share/extend this project publicly.
