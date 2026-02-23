from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import Counter
import re

app = Flask(__name__)

@dataclass
class Movie:
    """Movie data class"""
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None

class MovieDatabase:
    """Enhanced movie database with advanced features"""
    
    def __init__(self, filepath: str):
        self.df = self._load_and_clean("/home/suks/code/projects/movie-recommender/movies.csv")
        self.movies = self._create_movie_objects()
        self._build_genre_index()
        
    def _load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Load and clean the CSV data"""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        df['title'] = df['title'].str.strip()
        df['genres'] = df['genres'].str.strip()
        df['genre_list'] = df['genres'].str.split('|')
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')[0].astype('Int64')
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
        """Build index for fast genre lookup"""
        self.genre_index = {}
        for movie in self.movies.values():
            for genre in movie.genres:
                if genre not in self.genre_index:
                    self.genre_index[genre] = []
                self.genre_index[genre].append(movie.movie_id)
    
    @lru_cache(maxsize=1)
    def get_all_genres(self) -> List[str]:
        """Get all unique genres, cached"""
        genres = set()
        for movie in self.movies.values():
            genres.update(movie.genres)
        return sorted(list(genres))
    
    def search_movies(self, query: str, limit: int = 20) -> List[Movie]:
        """Enhanced search with fuzzy matching"""
        query_lower = query.lower()
        query_parts = query_lower.split()
        
        # Score each movie based on query match
        matches = []
        for movie in self.movies.values():
            title_lower = movie.title.lower()
            score = 0
            
            # Exact substring match
            if query_lower in title_lower:
                score += 100
            
            # Word matching
            for part in query_parts:
                if part in title_lower:
                    score += 10
            
            # Year matching
            if query.isdigit() and str(movie.year) == query:
                score += 50
            
            if score > 0:
                matches.append((movie, score))
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:limit]]
    
    def get_movie(self, movie_id: int) -> Optional[Movie]:
        """Get movie by ID"""
        return self.movies.get(movie_id)
    
    def get_movies_by_year(self, year: int) -> List[Movie]:
        """Get movies from specific year"""
        return [m for m in self.movies.values() if m.year == year]
    
    def get_year_range(self) -> Tuple[int, int]:
        """Get min and max years in database"""
        years = [m.year for m in self.movies.values() if m.year]
        return (min(years), max(years)) if years else (0, 0)

class RecommendationEngine:
    """Advanced recommendation engine with multiple algorithms"""
    
    def __init__(self, database: MovieDatabase):
        self.db = database
    
    def calculate_genre_similarity(self, genres1: Set[str], genres2: Set[str]) -> float:
        """Calculate Jaccard similarity between genre sets"""
        if not genres1 or not genres2:
            return 0.0
        intersection = len(genres1.intersection(genres2))
        union = len(genres1.union(genres2))
        return intersection / union if union > 0 else 0.0
    
    def get_genre_profile(self, watched_ids: List[int]) -> Tuple[Set[str], Dict[str, float]]:
        """Extract weighted genre profile with TF-IDF-like scoring"""
        genre_counts = Counter()
        total_movies = len(watched_ids)
        
        for movie_id in watched_ids:
            movie = self.db.get_movie(movie_id)
            if movie:
                for genre in movie.genres:
                    genre_counts[genre] += 1
        
        # Calculate weights with inverse frequency penalty
        total_genre_frequency = sum(genre_counts.values())
        genre_weights = {}
        
        for genre, count in genre_counts.items():
            # TF (term frequency)
            tf = count / total_movies
            # IDF-like penalty (popular genres get lower weight)
            total_with_genre = len(self.db.genre_index.get(genre, []))
            idf = np.log(len(self.db.movies) / (1 + total_with_genre))
            # Combined weight
            genre_weights[genre] = tf * (1 + idf * 0.1)
        
        # Normalize weights
        max_weight = max(genre_weights.values()) if genre_weights else 1
        genre_weights = {g: w/max_weight for g, w in genre_weights.items()}
        
        return set(genre_counts.keys()), genre_weights
    
    def calculate_year_similarity(self, year1: Optional[int], year2: Optional[int], max_diff: int = 10) -> float:
        """Calculate similarity based on release year"""
        if year1 is None or year2 is None:
            return 0.5  # Neutral score for unknown years
        
        diff = abs(year1 - year2)
        if diff == 0:
            return 1.0
        return max(0, 1 - (diff / max_diff))
    
    def recommend(
        self, 
        watched_ids: List[int], 
        preferred_genres: List[str] = None,
        n: int = 15,
        diversity_factor: float = 0.3,
        year_weight: float = 0.2
    ) -> List[Tuple[dict, float]]:
        """
        Advanced recommendation with multiple scoring factors
        
        Args:
            watched_ids: List of watched movie IDs
            preferred_genres: Optional list of preferred genres
            n: Number of recommendations
            diversity_factor: 0-1, higher = more diverse
            year_weight: 0-1, weight of year similarity in scoring
        """
        watched_set = set(watched_ids)
        watched_movies = [self.db.get_movie(mid) for mid in watched_ids if self.db.get_movie(mid)]
        
        # Get target genres and weights
        if preferred_genres:
            target_genres = set(preferred_genres)
            genre_weights = {g: 1.0 for g in target_genres}
        else:
            target_genres, genre_weights = self.get_genre_profile(watched_ids)
        
        if not target_genres:
            return []
        
        # Calculate average year of watched movies
        watched_years = [m.year for m in watched_movies if m.year]
        avg_year = np.mean(watched_years) if watched_years else None
        
        # Score all unwatched movies
        candidates = []
        for movie in self.db.movies.values():
            if movie.movie_id in watched_set:
                continue
            
            # Genre matching score
            genre_score = 0.0
            movie_genre_set = set(movie.genres)
            
            for genre in movie.genres:
                if genre in genre_weights:
                    genre_score += genre_weights[genre]
            
            # Normalize by movie's genre count
            genre_score = genre_score / len(movie.genres) if movie.genres else 0
            
            # Calculate genre similarity (Jaccard)
            similarity = self.calculate_genre_similarity(movie_genre_set, target_genres)
            
            # Diversity bonus (penalize exact matches slightly)
            diversity_bonus = (1 - diversity_factor) + diversity_factor * (1 - similarity)
            
            # Year similarity score
            year_score = 1.0
            if avg_year is not None:
                year_score = self.calculate_year_similarity(movie.year, avg_year)
            
            # Combined score with weights
            final_score = (
                genre_score * (1 - year_weight) +
                year_score * year_weight
            ) * diversity_bonus
            
            if final_score > 0:
                candidates.append((asdict(movie), final_score))
        
        # Sort and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]
    
    def get_trending(self, genre: str = None, limit: int = 10) -> List[Tuple[dict, float]]:
        """Get trending movies based on genre popularity"""
        if genre:
            movie_ids = self.db.genre_index.get(genre, [])
            movies = [self.db.get_movie(mid) for mid in movie_ids]
        else:
            movies = list(self.db.movies.values())
        
        # Score by recency and genre count
        scored = []
        for movie in movies:
            score = 0
            if movie.year:
                # Recent movies get higher scores
                recency = max(0, movie.year - 1980) / 50
                score += recency
            # More genres = more popular (rough heuristic)
            score += len(movie.genres) * 0.1
            
            scored.append((asdict(movie), score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

# Initialize database and engine
print("="*70)
print("🎬 INITIALIZING MOVIE RECOMMENDER SYSTEM")
print("="*70)
print("Loading movie database...")
db = MovieDatabase('movies.csv')
engine = RecommendationEngine(db)
min_year, max_year = db.get_year_range()
print(f"✓ Loaded {len(db.movies):,} movies")
print(f"✓ Available genres: {len(db.get_all_genres())}")
print(f"✓ Year range: {min_year} - {max_year}")
print("="*70)

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/movies/search')
def search_movies():
    """Enhanced search with scoring"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))
    
    if len(query) < 2:
        return jsonify([])
    
    movies = db.search_movies(query, limit)
    return jsonify([asdict(m) for m in movies])

@app.route('/api/movies/<int:movie_id>')
def get_movie(movie_id):
    """Get a specific movie"""
    movie = db.get_movie(movie_id)
    if movie:
        return jsonify(asdict(movie))
    return jsonify({'error': 'Movie not found'}), 404

@app.route('/api/genres')
def get_genres():
    """Get all genres"""
    return jsonify(db.get_all_genres())

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get advanced movie recommendations"""
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
        n,
        diversity,
        year_weight
    )
    
    return jsonify([
        {'movie': movie, 'score': float(score)}
        for movie, score in recommendations
    ])

@app.route('/api/trending')
def get_trending():
    """Get trending movies"""
    genre = request.args.get('genre')
    limit = int(request.args.get('limit', 10))
    
    trending = engine.get_trending(genre, limit)
    return jsonify([
        {'movie': movie, 'score': float(score)}
        for movie, score in trending
    ])

@app.route('/api/stats')
def get_stats():
    """Get enhanced database statistics"""
    min_year, max_year = db.get_year_range()
    
    # Genre distribution
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
    """Find movies similar to a specific movie"""
    movie = db.get_movie(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    
    # Use recommendation engine with just this movie
    recommendations = engine.recommend([movie_id], n=10, diversity_factor=0.1)
    
    return jsonify([
        {'movie': m, 'score': float(score)}
        for m, score in recommendations
    ])

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 SERVER RUNNING")
    print("="*70)
    print("📡 URL: http://localhost:5000")
    print("📱 Open this URL in your browser to access the interface")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
