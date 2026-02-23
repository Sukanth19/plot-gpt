from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import Counter
import re
import io
import base64

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (important for Flask!)
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Movie:
    """Movie data class"""
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None


# ─────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────

class MovieDatabase:
    """Enhanced movie database with advanced features"""

    def __init__(self, filepath: str):
        self.df = self._load_and_clean(filepath)
        self.movies = self._create_movie_objects()
        self._build_genre_index()
        self._build_genre_vectors()      # NEW: for cosine similarity
        self._build_cooccurrence_matrix() # NEW: genre relationships

    def _load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Load and clean the CSV data"""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        df['title'] = df['title'].str.strip()
        df['genres'] = df['genres'].str.strip()
        df['genre_list'] = df['genres'].str.split('|')
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')[0].astype('Int64')
        # Remove movies with no genres listed
        df = df[df['genres'] != '(no genres listed)']
        return df

    def _create_movie_objects(self) -> dict:
        """Create Movie objects from dataframe"""
        movies = {}
        for _, row in self.df.iterrows():
            movies[row['movieId']] = Movie(
                movie_id=row['movieId'],
                title=row['title'],
                genres=row['genre_list'],
                year=int(row['year']) if pd.notna(row['year']) else None
            )
        return movies

    def _build_genre_index(self):
        """Build index for fast genre lookup: {genre -> [movie_ids]}"""
        self.genre_index = {}
        for movie in self.movies.values():
            for genre in movie.genres:
                if genre not in self.genre_index:
                    self.genre_index[genre] = []
                self.genre_index[genre].append(movie.movie_id)

    def _build_genre_vectors(self):
        """
        Build a binary genre vector for each movie.
        
        THEORY: We represent each movie as a vector in genre-space.
        If there are 20 genres, each movie gets a 20-dimensional vector
        where 1 means the movie has that genre, 0 means it doesn't.
        
        Example: Genres = [Action, Comedy, Drama]
        "Die Hard" → [1, 0, 0]
        "Toy Story" → [0, 1, 0]
        "The Pursuit of Happyness" → [0, 0, 1]
        
        This enables cosine similarity calculations.
        """
        all_genres = self.get_all_genres()
        self.genre_list_ordered = all_genres  # fixed ordering
        self.genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        # Build numpy matrix: shape (n_movies, n_genres)
        movie_ids = list(self.movies.keys())
        self.movie_id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
        self.row_to_movie_id = {i: mid for i, mid in enumerate(movie_ids)}

        n = len(movie_ids)
        g = len(all_genres)
        self.genre_matrix = np.zeros((n, g), dtype=np.float32)

        for movie_id, movie in self.movies.items():
            row = self.movie_id_to_row[movie_id]
            for genre in movie.genres:
                if genre in self.genre_to_idx:
                    col = self.genre_to_idx[genre]
                    self.genre_matrix[row, col] = 1.0

        # Pre-normalize rows for fast cosine similarity later
        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        # If rows are pre-normalized (||row|| = 1), then cos_sim = dot(A, B)
        norms = np.linalg.norm(self.genre_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self.genre_matrix_normalized = self.genre_matrix / norms

    def _build_cooccurrence_matrix(self):
        """
        Build genre co-occurrence matrix.
        
        THEORY: Co-occurrence counts how often two genres appear together.
        If Action and Thriller always appear together, they're strongly related.
        This helps the recommender understand genre relationships.
        
        Result: cooccurrence[i][j] = number of movies with both genre i and genre j
        """
        genres = self.get_all_genres()
        n = len(genres)
        self.cooccurrence = np.zeros((n, n), dtype=np.int32)

        for movie in self.movies.values():
            indices = [self.genre_to_idx[g] for g in movie.genres if g in self.genre_to_idx]
            for i in indices:
                for j in indices:
                    self.cooccurrence[i][j] += 1

    @lru_cache(maxsize=1)
    def get_all_genres(self) -> List[str]:
        """Get all unique genres, cached"""
        genres = set()
        for movie in self.movies.values():
            genres.update(movie.genres)
        return sorted(list(genres))

    def search_movies(self, query: str, limit: int = 20) -> List[Movie]:
        """Enhanced search with scoring"""
        query_lower = query.lower()
        query_parts = query_lower.split()
        matches = []

        for movie in self.movies.values():
            title_lower = movie.title.lower()
            score = 0
            if query_lower in title_lower:
                score += 100
            for part in query_parts:
                if part in title_lower:
                    score += 10
            if query.isdigit() and str(movie.year) == query:
                score += 50
            if score > 0:
                matches.append((movie, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:limit]]

    def get_movie(self, movie_id: int) -> Optional[Movie]:
        return self.movies.get(movie_id)

    def get_movies_by_year(self, year: int) -> List[Movie]:
        return [m for m in self.movies.values() if m.year == year]

    def get_year_range(self) -> Tuple[int, int]:
        years = [m.year for m in self.movies.values() if m.year]
        return (min(years), max(years)) if years else (0, 0)


# ─────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────

class RecommendationEngine:
    """Advanced recommendation engine with multiple algorithms"""

    def __init__(self, database: MovieDatabase):
        self.db = database

    # ── Similarity Methods ──────────────────────────────────

    def cosine_similarity_vector(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        
        THEORY: Measures the cosine of the angle between two vectors.
        - Result of 1.0 → identical direction (same genres)
        - Result of 0.0 → completely different genres
        - Unlike Jaccard, this respects the magnitude of genre preferences
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def batch_cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity of query_vec against ALL movies at once.
        
        THEORY: Using matrix multiplication for efficiency.
        Since genre_matrix_normalized has pre-normalized rows,
        dot product with a normalized query = cosine similarity.
        
        This is O(n_movies * n_genres) but vectorized in numpy — very fast.
        """
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(self.db.movies))
        query_normalized = query_vec / query_norm
        # Matrix multiply: (n_movies, n_genres) @ (n_genres,) → (n_movies,)
        return self.db.genre_matrix_normalized @ query_normalized

    def calculate_genre_similarity(self, genres1: Set[str], genres2: Set[str]) -> float:
        """Jaccard similarity — kept for reference"""
        if not genres1 or not genres2:
            return 0.0
        intersection = len(genres1.intersection(genres2))
        union = len(genres1.union(genres2))
        return intersection / union if union > 0 else 0.0

    def get_genre_profile(self, watched_ids: List[int]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Build a weighted genre profile vector from watch history.
        
        THEORY: 
        - We sum the genre vectors of all watched movies
        - Popular genres across your history get higher values
        - TF-IDF weighting reduces influence of overly common genres
        
        Returns both a numpy vector (for cosine sim) and a dict (for display)
        """
        genre_counts = Counter()
        total_movies = len(watched_ids)

        for movie_id in watched_ids:
            movie = self.db.get_movie(movie_id)
            if movie:
                for genre in movie.genres:
                    genre_counts[genre] += 1

        if not genre_counts:
            return np.zeros(len(self.db.get_all_genres())), {}

        # Build weighted vector
        profile_vec = np.zeros(len(self.db.get_all_genres()), dtype=np.float32)
        genre_weights = {}

        for genre, count in genre_counts.items():
            if genre not in self.db.genre_to_idx:
                continue
            idx = self.db.genre_to_idx[genre]

            # TF: how frequently this genre appeared in your history
            tf = count / total_movies

            # IDF-like: rare genres in the full dataset get higher weight
            total_with_genre = len(self.db.genre_index.get(genre, []))
            idf = np.log(len(self.db.movies) / (1 + total_with_genre))

            weight = tf * (1 + idf * 0.1)
            profile_vec[idx] = weight
            genre_weights[genre] = float(weight)

        # Normalize weights for display
        max_w = max(genre_weights.values()) if genre_weights else 1
        genre_weights = {g: w / max_w for g, w in genre_weights.items()}

        return profile_vec, genre_weights

    def calculate_year_similarity(self, year1: Optional[int], year2: Optional[int], max_diff: int = 10) -> float:
        """Year proximity score — closer years score higher"""
        if year1 is None or year2 is None:
            return 0.5
        diff = abs(year1 - year2)
        if diff == 0:
            return 1.0
        return max(0.0, 1.0 - (diff / max_diff))

    # ── Main Recommendation Method ───────────────────────────

    def recommend(
        self,
        watched_ids: List[int],
        preferred_genres: List[str] = None,
        n: int = 15,
        diversity_factor: float = 0.3,
        year_weight: float = 0.2
    ) -> List[Tuple[dict, float]]:
        """
        Recommendation using cosine similarity on genre vectors.
        
        ALGORITHM OVERVIEW:
        1. Build a user profile vector from watched movies (weighted genre vector)
        2. Compute cosine similarity between user profile and every unwatched movie
        3. Apply year similarity as a secondary signal
        4. Apply diversity factor to avoid recommending too-similar movies
        5. Return top N results
        
        WHY COSINE > JACCARD HERE:
        - Jaccard: "Does movie share genres with your profile? (yes/no)"
        - Cosine: "How closely does movie's genre direction match your preferences?"
        - Cosine respects the *strength* of your genre preferences
        """
        watched_set = set(watched_ids)
        watched_movies = [self.db.get_movie(mid) for mid in watched_ids if self.db.get_movie(mid)]

        # Step 1: Build user profile vector
        if preferred_genres:
            # Manual genre preference → build vector directly
            profile_vec = np.zeros(len(self.db.get_all_genres()), dtype=np.float32)
            for g in preferred_genres:
                if g in self.db.genre_to_idx:
                    profile_vec[self.db.genre_to_idx[g]] = 1.0
            genre_weights = {g: 1.0 for g in preferred_genres}
        else:
            profile_vec, genre_weights = self.get_genre_profile(watched_ids)

        if profile_vec.sum() == 0:
            return []

        # Step 2: Batch cosine similarity (fast numpy operation)
        all_similarities = self.batch_cosine_similarity(profile_vec)

        # Step 3: Calculate average year of watch history
        watched_years = [m.year for m in watched_movies if m.year]
        avg_year = float(np.mean(watched_years)) if watched_years else None

        # Step 4: Score candidates
        candidates = []
        for movie_id, movie in self.db.movies.items():
            if movie_id in watched_set:
                continue

            row_idx = self.db.movie_id_to_row[movie_id]
            cosine_score = float(all_similarities[row_idx])

            if cosine_score <= 0:
                continue

            # Year similarity
            year_score = 1.0
            if avg_year is not None:
                year_score = self.calculate_year_similarity(movie.year, int(avg_year))

            # Diversity bonus: penalize movies too similar to reduce echo chamber
            # diversity_factor=0 → pure relevance; diversity_factor=1 → max diversity
            diversity_bonus = (1 - diversity_factor) + diversity_factor * (1 - cosine_score)

            # Final score: weighted combination
            final_score = (
                cosine_score * (1 - year_weight) +
                year_score * year_weight
            ) * diversity_bonus

            candidates.append((asdict(movie), final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def get_trending(self, genre: str = None, limit: int = 10) -> List[Tuple[dict, float]]:
        """Get trending movies based on recency and genre count"""
        if genre:
            movie_ids = self.db.genre_index.get(genre, [])
            movies = [self.db.get_movie(mid) for mid in movie_ids]
        else:
            movies = list(self.db.movies.values())

        scored = []
        for movie in movies:
            score = 0
            if movie.year:
                recency = max(0, movie.year - 1980) / 50
                score += recency
            score += len(movie.genres) * 0.1
            scored.append((asdict(movie), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# ─────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure to a base64 string.
    
    WHY: Flask can't serve matplotlib figures directly.
    We convert to PNG bytes, then base64-encode so it can be
    embedded directly in JSON and displayed via <img src="data:...">
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_genre_distribution() -> str:
    """
    Bar chart of movie count per genre.
    
    INSIGHT: Shows which genres dominate the dataset.
    Important for understanding recommendation bias —
    if Drama has 5x more movies, it'll naturally appear more in recommendations.
    """
    genre_counts = Counter()
    for movie in db.movies.values():
        for genre in movie.genres:
            genre_counts[genre] += 1

    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    # Sort by count
    sorted_pairs = sorted(zip(counts, genres), reverse=True)
    counts, genres = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(genres), palette='viridis', ax=ax)
    ax.set_title('Movie Count by Genre', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Movies')
    ax.set_ylabel('Genre')
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_movies_per_year() -> str:
    """
    Line chart of movies released per year.
    
    INSIGHT: Shows dataset temporal distribution.
    Helps understand if year-based recommendations will be biased
    toward certain eras (datasets often have more modern movies).
    """
    year_counts = Counter()
    for movie in db.movies.values():
        if movie.year:
            year_counts[movie.year] += 1

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(years, counts, alpha=0.4, color='steelblue')
    ax.plot(years, counts, color='steelblue', linewidth=2)
    ax.set_title('Movies Released per Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Movies')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_genre_cooccurrence() -> str:
    """
    Heatmap of genre co-occurrence.
    
    THEORY: Co-occurrence shows which genres appear together.
    A bright cell at (Action, Thriller) means many movies are both.
    This reveals natural genre clusters and helps us understand
    whether our recommendations respect these clusters.
    
    Uses seaborn's heatmap — perfect for 2D matrix visualization.
    """
    genres = db.get_all_genres()
    matrix = db.cooccurrence.astype(float)

    # Normalize: divide by diagonal (total count of each genre)
    # This gives conditional probability: P(genre_j | genre_i)
    diag = np.diag(matrix).copy()
    diag[diag == 0] = 1
    normalized = matrix / diag[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        normalized,
        xticklabels=genres,
        yticklabels=genres,
        cmap='YlOrRd',
        ax=ax,
        square=True,
        linewidths=0.5,
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Co-occurrence Probability'}
    )
    ax.set_title('Genre Co-occurrence Heatmap\n(P(col genre | row genre))',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_genre_profile(watched_ids: List[int]) -> str:
    """
    Radar/bar chart of user's genre profile.
    
    INSIGHT: Shows what the algorithm *thinks* you like.
    Useful for debugging — if recommendations seem off,
    check if the profile matches your actual preferences.
    """
    _, genre_weights = engine.get_genre_profile(watched_ids)

    if not genre_weights:
        return ""

    # Sort by weight
    sorted_gw = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)
    genres = [g for g, _ in sorted_gw]
    weights = [w for _, w in sorted_gw]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("coolwarm", len(genres))
    bars = ax.barh(genres[::-1], weights[::-1], color=colors[::-1])
    ax.set_title('Your Genre Profile (Based on Watch History)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Preference Weight (normalized)')
    ax.set_xlim(0, 1.1)

    # Add value labels
    for bar, w in zip(bars, weights[::-1]):
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("🎬 INITIALIZING MOVIE RECOMMENDER SYSTEM")
print("=" * 70)
print("Loading movie database...")
db = MovieDatabase('movies.csv')
engine = RecommendationEngine(db)
min_year, max_year = db.get_year_range()
print(f"✓ Loaded {len(db.movies):,} movies")
print(f"✓ Available genres: {len(db.get_all_genres())}")
print(f"✓ Year range: {min_year} - {max_year}")
print(f"✓ Genre matrix shape: {db.genre_matrix.shape}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/movies/search')
def search_movies():
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))
    if len(query) < 2:
        return jsonify([])
    movies = db.search_movies(query, limit)
    return jsonify([asdict(m) for m in movies])


@app.route('/api/movies/<int:movie_id>')
def get_movie(movie_id):
    movie = db.get_movie(movie_id)
    if movie:
        return jsonify(asdict(movie))
    return jsonify({'error': 'Movie not found'}), 404


@app.route('/api/genres')
def get_genres():
    return jsonify(db.get_all_genres())


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    watched_ids = data.get('watched_ids', [])
    preferred_genres = data.get('preferred_genres', [])
    diversity = float(data.get('diversity', 0.3))
    n = int(data.get('n', 15))
    year_weight = float(data.get('year_weight', 0.2))

    if not watched_ids:
        return jsonify({'error': 'No watched movies provided'}), 400

    recommendations = engine.recommend(
        watched_ids,
        preferred_genres if preferred_genres else None,
        n, diversity, year_weight
    )
    return jsonify([
        {'movie': movie, 'score': float(score)}
        for movie, score in recommendations
    ])


@app.route('/api/trending')
def get_trending():
    genre = request.args.get('genre')
    limit = int(request.args.get('limit', 10))
    trending = engine.get_trending(genre, limit)
    return jsonify([
        {'movie': movie, 'score': float(score)}
        for movie, score in trending
    ])


@app.route('/api/stats')
def get_stats():
    min_year, max_year = db.get_year_range()
    genre_counts = Counter()
    for movie in db.movies.values():
        for genre in movie.genres:
            genre_counts[genre] += 1
    return jsonify({
        'total_movies': len(db.movies),
        'total_genres': len(db.get_all_genres()),
        'year_range': {'min': min_year, 'max': max_year},
        'top_genres': dict(genre_counts.most_common(10))
    })


@app.route('/api/similar/<int:movie_id>')
def get_similar(movie_id):
    movie = db.get_movie(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    recommendations = engine.recommend([movie_id], n=10, diversity_factor=0.1)
    return jsonify([
        {'movie': m, 'score': float(score)}
        for m, score in recommendations
    ])


# ── Visualization Routes ─────────────────────────────────────

@app.route('/api/viz/genre-distribution')
def viz_genre_distribution():
    """Returns base64 PNG of genre distribution bar chart"""
    img = plot_genre_distribution()
    return jsonify({'image': img, 'type': 'genre_distribution'})


@app.route('/api/viz/movies-per-year')
def viz_movies_per_year():
    """Returns base64 PNG of movies per year line chart"""
    img = plot_movies_per_year()
    return jsonify({'image': img, 'type': 'movies_per_year'})


@app.route('/api/viz/genre-cooccurrence')
def viz_genre_cooccurrence():
    """Returns base64 PNG of genre co-occurrence heatmap"""
    img = plot_genre_cooccurrence()
    return jsonify({'image': img, 'type': 'genre_cooccurrence'})


@app.route('/api/viz/genre-profile', methods=['POST'])
def viz_genre_profile():
    """Returns base64 PNG of user's genre profile"""
    data = request.json
    watched_ids = data.get('watched_ids', [])
    if not watched_ids:
        return jsonify({'error': 'No watched movies'}), 400
    img = plot_genre_profile(watched_ids)
    return jsonify({'image': img, 'type': 'genre_profile'})


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌐 SERVER RUNNING")
    print("=" * 70)
    print("📡 URL: http://localhost:5000")
    print("📊 Visualizations available at /api/viz/*")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)