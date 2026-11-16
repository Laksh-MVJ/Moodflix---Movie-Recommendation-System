from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib

MOOD_GENRES = {
 "Happy": ["Comedy","Animation","Musical","Children"],
 "Sad": ["Drama","Romance"],
 "Stressed": ["Comedy","Animation","Adventure"],
 "Bored": ["Mystery","Thriller","Adventure","Sci-Fi"],
 "Romantic": ["Romance","Drama","Comedy"],
 "Adventurous": ["Adventure","Action","Fantasy","Sci-Fi"],
 "Motivated": ["Drama","Sport","Documentary","War"],
 "Scared": ["Horror","Thriller","Mystery"],
}
class MovieRecommender:
    def __init__(self, movie_path, ratings_path, model_dir):
        self.movie_path, self.ratings_path = Path(movie_path), Path(ratings_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.moods = list(MOOD_GENRES)
        self.ready = False

    def _load_data(self):
        if not self.movie_path.exists():
            raise FileNotFoundError("Dataset missing. Run: setup.bat")
        df = pd.read_csv(self.movie_path)
        for col in ["imdb_rating","imdb_votes","moods","genres","title"]:
            if col not in df: df[col] = ""
        df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce").fillna(0)
        df["imdb_votes"] = pd.to_numeric(df["imdb_votes"], errors="coerce").fillna(0)
        df["moods"] = df["moods"].fillna("").apply(lambda s: [x for x in str(s).split("|") if x])
        df["genres"] = df["genres"].fillna("").astype(str)
        df["title"] = df["title"].fillna("Unknown")
        return df

    def load_or_train(self):
        self.movies = self._load_data()
        artifact = self.model_dir / "content_model.joblib"
        if artifact.exists():
            try:
                bundle = joblib.load(artifact)
                if bundle.get("movie_count") == len(self.movies):
                    self.vectorizer, self.matrix = bundle["vectorizer"], bundle["matrix"]
                    self.item_knn, self.item_matrix = bundle["item_knn"], bundle["item_matrix"]
                    self.cluster_labels = bundle["cluster_labels"]
                    self.movie_index = {int(mid): i for i, mid in enumerate(self.movies.movieId)}
                    self.ready = True
                    return
            except Exception:
                pass
        self.train()

    def train(self):
        text = (self.movies["genres"].astype(str) + " " + self.movies["title"].astype(str))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(text)
        self.movie_index = {int(mid): i for i, mid in enumerate(self.movies.movieId)}

        # Item-item collaborative KNN trained from real MovieLens user ratings.
        if self.ratings_path.exists():
            ratings = pd.read_csv(self.ratings_path)
            pivot = ratings.pivot_table(index="movieId", columns="userId", values="rating", fill_value=0)
            pivot = pivot.reindex(self.movies.movieId, fill_value=0)
            self.item_matrix = csr_matrix(pivot.values.astype(np.float32))
        else:
            self.item_matrix = csr_matrix((len(self.movies), 0), dtype=np.float32)
        self.item_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(16, max(1,len(self.movies))))
        if self.item_matrix.shape[1] > 0:
            self.item_knn.fit(self.item_matrix)
        else:
            self.item_knn = None

        # Unsupervised genre/content clusters; useful for discovery and explainability.
        cluster_count = min(18, max(2, len(self.movies)//100))
        self.cluster_labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(self.matrix)
        joblib.dump({
            "vectorizer":self.vectorizer, "matrix":self.matrix,
            "item_knn":self.item_knn, "item_matrix":self.item_matrix,
            "cluster_labels":self.cluster_labels, "movie_count":len(self.movies)
        }, self.model_dir / "content_model.joblib")
        self.ready = True

    def _popularity(self, df):
        rating = df.imdb_rating.clip(0,10) / 10
        confidence = np.log1p(df.imdb_votes) / max(1, np.log1p(max(1, df.imdb_votes.max())))
        return 0.82 * rating + 0.18 * confidence

    def recommend(self, mood="Happy", liked_ids=None, n=12):
        liked_ids = liked_ids or []
        df = self.movies.copy()
        if mood not in MOOD_GENRES: mood = "Happy"
        wanted = set(MOOD_GENRES[mood])
        df["mood_fit"] = df.genres.apply(lambda g: len(wanted.intersection(set(str(g).split("|")))) / max(1,len(wanted)))
        df["score"] = 0.42 * df.mood_fit + 0.28 * self._popularity(df) + 0.15 * (df.imdb_rating / 10)
        liked_rows = [self.movie_index[x] for x in liked_ids if x in self.movie_index]
        if liked_rows:
            # Content similarity profile from movies the user liked.
            profile = self.matrix[liked_rows].mean(axis=0)
            content_sim = cosine_similarity(self.matrix, np.asarray(profile)).ravel()
            df["score"] = 0.70 * df.score + 0.30 * content_sim
            # Collaborative item-item KNN: neighbours share rating patterns with liked movies.
            if self.item_knn is not None and self.item_matrix.shape[1] > 0:
                collab = np.zeros(len(self.movies), dtype=float)
                for idx in liked_rows:
                    distances, indices = self.item_knn.kneighbors(self.item_matrix[idx], n_neighbors=min(16,len(self.movies)))
                    for dist, neighbor in zip(distances[0], indices[0]):
                        collab[neighbor] += max(0.0, 1.0-float(dist))
                if collab.max() > 0:
                    collab /= collab.max()
                    df["score"] = 0.82 * df.score + 0.18 * collab
        df["cluster"] = self.cluster_labels
        df = df.sort_values(["score","imdb_rating"], ascending=False)
        if liked_ids: df = df[~df.movieId.isin(liked_ids)]
        out = df.head(n).copy()
        out["moods"] = out["moods"].apply(lambda x: x if isinstance(x,list) else [])
        return out[["movieId","title","genres","imdb_rating","imdb_votes","moods","score","cluster"]].to_dict("records")
