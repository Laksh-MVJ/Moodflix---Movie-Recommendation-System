from pathlib import Path
from recommender import MovieRecommender
ROOT=Path(__file__).resolve().parent
engine=MovieRecommender(ROOT/"data/movies_enriched.csv",ROOT/"data/ratings.csv",ROOT/"models")
engine.movies=engine._load_data()
engine.train()
print("MoodFlix recommendation model trained.")