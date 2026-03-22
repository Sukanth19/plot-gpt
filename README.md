```
 _____ _       _      ____ ____ _____ 
|  __ \ |     | |    / ___|  _ \_   _|
| |__) | | ___ | |_  | |  _| |_) || |  
|  ___/| |/ _ \| __| | |_| |  __/ | |  
| |    | | (_) | |_   \____|_|    | |  
|_|    |_|\___/ \__|                   
```

<div align="center">

**A retro-styled movie recommendation engine with intelligent genre-based matching**

`[ Content-Based Filtering ]` · `[ TF-IDF Profiling ]` · `[ CRT Aesthetics ]`

---

</div>

## .:[ OVERVIEW ]:.

A Flask-based movie recommendation system featuring a nostalgic CRT terminal interface. Built on cosine similarity over genre vectors with TF-IDF user profiling, designed to evolve into a hybrid recommendation system combining content-based and collaborative filtering approaches.

```
┌─────────────────────────────────────────────────────────────┐
│  > Search movies                                            │
│  > Build your watch history                                 │
│  > Get personalized recommendations                         │
│  > Visualize your genre preferences                         │
│  > Tune diversity and year weighting                        │
└─────────────────────────────────────────────────────────────┘
```

---

## .:[ FEATURES ]:.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  [*] Interactive Web UI                                          │
│      └─> Search, watch history, personalized recommendations     │
│                                                                  │
│  [*] Content-Based Recommendations                               │
│      └─> Genre vectorization with cosine similarity              │
│                                                                  │
│  [*] TF-IDF Weighting                                            │
│      └─> Intelligent user profile generation                     │
│                                                                  │
│  [*] Data Visualizations                                         │
│      └─> Genre distribution, heatmaps, personal profiles         │
│                                                                  │
│  [*] Retro Aesthetic                                             │
│      └─> CRT terminal styling with scanlines and glow            │
│                                                                  │
│  [*] Configurable Parameters                                     │
│      └─> Diversity, year weighting, result count                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## .:[ QUICK START ]:.

```bash
# Clone the repository
git clone https://github.com/Sukanth19/plot-gpt.git
cd plot-gpt

# Download the dataset from Kaggle
# https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system
# Extract movies.csv and ratings.csv to the data/ directory

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

```
┌─────────────────────────────────────┐
│  Server running on:                 │
│  http://localhost:5000              │
└─────────────────────────────────────┘
```

---

## .:[ PROJECT STRUCTURE ]:.

```
plot-gpt/
│
├── src/                          # Source code
│   ├── app.py                    # Flask routes & API endpoints
│   ├── models.py                 # SQLAlchemy database models
│   ├── database.py               # Database manager
│   ├── migration.py              # CSV to database migration
│   └── engine/                   # Recommendation engine
│       ├── movie_db.py           # Movie database & indexing
│       ├── recommender.py        # Recommendation algorithms
│       └── visualizations.py     # Chart generation
│
├── data/                         # Data files
│   ├── movies.csv                # MovieLens movies dataset
│   ├── ratings.csv               # MovieLens ratings dataset
│   └── movies.db                 # SQLite database (generated)
│
├── tests/                        # Test suite
│   └── unit/                     # Unit & property-based tests
│
├── templates/                    # HTML templates
│   └── index.html                # Single-page frontend
│
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## .:[ HOW IT WORKS ]:.

### >> Content-Based Recommendation Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  STEP 1: Genre Vectorization                                    │
│  ────────────────────────────                                   │
│  Each movie → vector in genre-space                             │
│  Binary encoding: [1, 0, 1, 0, ...]                             │
│  Pre-normalized for fast cosine similarity                      │
│                                                                 │
│  STEP 2: User Profile Generation                                │
│  ─────────────────────────────                                  │
│  TF  (Term Frequency): Genre frequency in watched movies        │
│  IDF (Inverse Document Frequency): Down-weight common genres    │
│  Result: Personalized genre preference vector                   │
│                                                                 │
│  STEP 3: Cosine Similarity                                      │
│  ───────────────────────                                        │
│  Measure alignment: user profile ↔ candidate movies             │
│  Range: 0.0 (unrelated) to 1.0 (perfect match)                  │
│  Respects strength of genre preferences                         │
│                                                                 │
│  STEP 4: Final Scoring                                          │
│  ────────────────────                                           │
│  • Genre similarity (primary signal)                            │
│  • Year proximity (secondary signal)                            │
│  • Diversity factor (avoid echo chamber)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### >> Why Cosine Over Jaccard?

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  JACCARD SIMILARITY                                      │
│  └─> Binary set overlap                                  │
│  └─> "Do they share genres?"                             │
│  └─> Ignores preference strength                         │
│                                                          │
│  COSINE SIMILARITY                                       │
│  └─> Vector alignment                                    │
│  └─> "How strongly do preferences align?"                │
│  └─> Respects distribution & strength                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## .:[ API ENDPOINTS ]:.

### >> Core Endpoints

```
┌────────┬──────────────────────────────┬─────────────────────────────┐
│ Method │ Endpoint                     │ Description                 │
├────────┼──────────────────────────────┼─────────────────────────────┤
│ GET    │ /                            │ Serve frontend              │
│ GET    │ /api/movies/search?q=...     │ Search movies by title      │
│ GET    │ /api/movies/<id>             │ Get movie by ID             │
│ GET    │ /api/genres                  │ List all genres             │
│ POST   │ /api/recommendations         │ Get recommendations         │
│ GET    │ /api/similar/<id>            │ Similar movies              │
│ GET    │ /api/trending                │ Trending movies             │
│ GET    │ /api/stats                   │ Dataset statistics          │
└────────┴──────────────────────────────┴─────────────────────────────┘
```

### >> Visualization Endpoints

```
┌────────┬──────────────────────────────┬─────────────────────────────┐
│ Method │ Endpoint                     │ Description                 │
├────────┼──────────────────────────────┼─────────────────────────────┤
│ GET    │ /api/viz/genre-distribution  │ Genre count bar chart       │
│ GET    │ /api/viz/movies-per-year     │ Release timeline            │
│ GET    │ /api/viz/genre-cooccurrence  │ Genre relationship heatmap  │
│ POST   │ /api/viz/genre-profile       │ User preference profile     │
└────────┴──────────────────────────────┴─────────────────────────────┘
```

`Note: All visualization endpoints return base64-encoded PNG images in JSON`

---

## .:[ RECOMMENDATION REQUEST ]:.

```json
{
  "watched_ids": [1, 2, 3],
  "preferred_genres": ["Drama", "Thriller"],
  "n": 15,
  "diversity": 0.3,
  "year_weight": 0.2
}
```

```
┌─────────────────────────────────────────────────────────────┐
│  PARAMETERS                                                 │
│  ──────────                                                 │
│                                                             │
│  watched_ids      [required]  List of movie IDs watched     │
│  preferred_genres [optional]  Override inferred profile     │
│  n                [optional]  Number of results (def: 15)   │
│  diversity        [optional]  0.0-1.0 range (def: 0.3)      │
│  year_weight      [optional]  Year influence (def: 0.2)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## .:[ TECH STACK ]:.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  BACKEND                                                    │
│  ───────                                                    │
│  • Flask 3.1.3          Web framework                       │
│  • pandas & numpy       Data processing                     │
│  • SQLAlchemy           Database ORM                        │
│  • matplotlib & seaborn Visualizations                      │
│  • Hypothesis           Property-based testing              │
│                                                             │
│  FRONTEND                                                   │
│  ────────                                                   │
│  • Vanilla JavaScript   No framework                        │
│  • HTML/CSS             Retro terminal styling              │
│  • Google Fonts         VT323, Share Tech Mono              │
│                                                             │
│  DATA                                                       │
│  ────                                                       │
│  • MovieLens dataset    movies.csv, ratings.csv             │
│  • SQLite database      Future hybrid features              │
│  • numpy matrices       In-memory performance               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## .:[ DEVELOPMENT ]:.

### >> Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/unit/test_migration_data_preservation.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### >> Database Migration

```bash
# Migrate CSV data to SQLite database
python -c "from src.database import DatabaseManager; \
           db = DatabaseManager('sqlite:///data/movies.db'); \
           db.create_schema(); \
           db.migrate_from_csv('data/movies.csv', 'data/ratings.csv')"
```

### >> Adding Dependencies

```bash
pip install <package>
pip freeze > requirements.txt
```

---

## .:[ ROADMAP ]:.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [x] Content-based recommendations with cosine similarity   │
│  [x] TF-IDF user profiling                                  │
│  [x] Interactive visualizations                             │
│  [x] Database infrastructure (SQLAlchemy + SQLite)          │
│                                                             │
│  [ ] Collaborative filtering (user-based CF)                │
│  [ ] Hybrid recommendation algorithm                        │
│  [ ] Personalized learning from interactions                │
│  [ ] Star rating system                                     │
│  [ ] Watchlist feature                                      │
│  [ ] Recommendation explanations                            │
│  [ ] Interactive Plotly charts                              │
│  [ ] Lazy loading for performance                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## .:[ DATASET ]:.

This project uses the **MovieLens dataset** available on Kaggle:

```
movies.csv   → movieId, title, genres (pipe-separated)
ratings.csv  → userId, movieId, rating, timestamp
```

**Download:** https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

After downloading, extract `movies.csv` and `ratings.csv` to the `data/` directory.

`Note: Movies with "(no genres listed)" are excluded from recommendations`

---

## .:[ LICENSE ]:.

```
MIT License - See LICENSE file for details
```

---

## .:[ CONTRIBUTING ]:.

Contributions welcome! Please follow these steps:

```
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
```

---

## .:[ ACKNOWLEDGMENTS ]:.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  • MovieLens dataset by GroupLens Research                  │
│  • Inspired by classic terminal UIs                         │
│  • Retro computing aesthetics                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

<div align="center">

**Made with <3 for movie lovers and retro computing enthusiasts**

`[ Flask ]` · `[ Python ]` · `[ Machine Learning ]` · `[ Retro UI ]`

</div>
