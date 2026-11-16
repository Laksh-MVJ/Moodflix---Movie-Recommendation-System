"""Download MovieLens + IMDb ratings, merge by IMDb ID, and create mood labels."""
from pathlib import Path
import urllib.request, zipfile, gzip, shutil, ssl, certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
IMDB_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

def download(url, target):
    if not target.exists():
        print("Downloading", target.name)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=90) as response, open(target, "wb") as output:
            shutil.copyfileobj(response, output)
    else:
        print("Using cached", target.name)

def moods_for(genres):
    g = set(str(genres).split("|"))
    rules = {
      "Happy":{"Comedy","Animation","Musical","Children"},
      "Sad":{"Drama","Romance"},
      "Stressed":{"Comedy","Animation","Adventure"},
      "Bored":{"Mystery","Thriller","Adventure","Sci-Fi"},
      "Romantic":{"Romance","Drama","Comedy"},
      "Adventurous":{"Adventure","Action","Fantasy","Sci-Fi"},
      "Motivated":{"Drama","Sport","Documentary","War"},
      "Scared":{"Horror","Thriller","Mystery"},
    }
    found = [m for m, gs in rules.items() if g & gs]
    return "|".join(found or ["Bored"])

def main():
    zip_path = DATA / "ml-latest-small.zip"
    download(ML_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z: z.extractall(DATA)
    folder = DATA / "ml-latest-small"
    movies = pd.read_csv(folder / "movies.csv")
    links = pd.read_csv(folder / "links.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    # IMDb's public title.ratings file is TSV.GZ. It is refreshed daily.
    imdb_gz = DATA / "title.ratings.tsv.gz"
    download(IMDB_URL, imdb_gz)
    imdb = pd.read_csv(imdb_gz, sep="\t", compression="gzip", na_values="\\N")
    imdb = imdb.rename(columns={"tconst":"imdb_id","averageRating":"imdb_rating","numVotes":"imdb_votes"})
    links["imdb_id"] = links["imdbId"].apply(lambda x: ("tt%07d" % int(x)) if pd.notna(x) and int(x)>0 else "")
    merged = movies.merge(links[["movieId","imdb_id"]], on="movieId", how="left")
    merged = merged.merge(imdb[["imdb_id","imdb_rating","imdb_votes"]], on="imdb_id", how="left")
    merged["imdb_rating"] = merged["imdb_rating"].fillna(0)
    merged["imdb_votes"] = merged["imdb_votes"].fillna(0)
    merged["moods"] = merged["genres"].apply(moods_for)
    merged.to_csv(DATA / "movies_enriched.csv", index=False)
    ratings.to_csv(DATA / "ratings.csv", index=False)
    print(f"Ready: {len(merged):,} movies, {len(ratings):,} MovieLens ratings.")
    print("IMDb ratings matched:", int((merged.imdb_rating > 0).sum()))
    print("Next: python train_model.py")

if __name__ == "__main__": main()
