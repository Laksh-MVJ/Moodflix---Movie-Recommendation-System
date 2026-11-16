from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd, random, subprocess, sys, threading
from recommender import MovieRecommender

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / "data", ROOT / "models"
app = Flask(__name__)
engine = None
lock = threading.Lock()

def boot():
    global engine
    movie_file = DATA / "movies_enriched.csv"
    if not movie_file.exists():
        subprocess.run([sys.executable, str(ROOT / "setup_data.py")], cwd=ROOT, check=True)
    engine = MovieRecommender(movie_file, DATA / "ratings.csv", MODELS)
    engine.load_or_train()

boot()

@app.get("/")
def home(): return render_template("index.html")

@app.get("/wheel")
def wheel(): return render_template("wheel.html")

@app.get("/api/health")
def health(): return jsonify({"ready": bool(engine and engine.ready), "count": len(engine.movies) if engine else 0})

@app.get("/api/moods")
def moods(): return jsonify(engine.moods)

@app.get("/api/movies")
def movies():
    q=request.args.get("q","").strip().lower()
    mood=request.args.get("mood","")
    limit=min(max(int(request.args.get("limit",60)),1),100)
    df=engine.movies.copy()
    if q:
        df=df[df.title.str.lower().str.contains(q,na=False) | df.genres.str.lower().str.contains(q,na=False)]
    if mood in engine.moods: df=df[df.moods.apply(lambda x:mood in x)]
    return jsonify(df.head(limit).to_dict("records"))

@app.get("/api/recommend")
def recommend():
    mood=request.args.get("mood","Happy")
    liked=[int(x) for x in request.args.get("liked","").split(",") if x.strip().isdigit()]
    return jsonify(engine.recommend(mood=mood,liked_ids=liked,n=min(int(request.args.get("n",18)),30)))

@app.get("/api/spin")
def spin():
    mood=request.args.get("mood","Any")
    pool=engine.movies
    if mood in engine.moods: pool=pool[pool.moods.apply(lambda x:mood in x)]
    pool=pool.sample(min(12,len(pool)))
    return jsonify({"winner":pool.sample(1).iloc[0].to_dict(),"wheel":pool.to_dict("records")})

@app.post("/api/rate")
def rate():
    p=request.get_json(force=True); mid=int(p["movieId"]); rating=float(p["rating"])
    if not .5 <= rating <= 5: return jsonify({"error":"Rating must be between 0.5 and 5"}),400
    path=DATA/"user_feedback.csv"
    row=pd.DataFrame([{"userId":"local_user","movieId":mid,"rating":rating}])
    if path.exists():
        old=pd.read_csv(path); old=old[old.movieId!=mid]; row=pd.concat([old,row],ignore_index=True)
    row.to_csv(path,index=False)
    return jsonify({"saved":True})

if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)