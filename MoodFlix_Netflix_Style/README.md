# MoodFlix — Movie discovery and recommendation app

A complete Flask + machine-learning movie discovery website with a cinematic streaming-service-inspired interface, mood recommendations, movie search, taste-profile likes, IMDb rating data, poster/backdrop artwork, and a separate spinning-wheel page with both mood-based and pure-random modes.

## Start on Windows
1. Extract the ZIP.
2. Double-click `setup.bat` once. It creates a virtual environment, installs packages, downloads MovieLens and IMDb rating data, and trains the recommendation model.
3. Double-click `run.bat` to launch the website.
4. Open http://127.0.0.1:5000

The first setup needs an internet connection and may take a few minutes. The app uses CSV files and a local ML model; it does not require MySQL, SQLite, or a database server.

## Real movie posters and banners
The app uses TMDB's movie image catalogue for real poster and backdrop images. On the Discover page, click **Poster settings** and paste your free TMDB API key. It is stored only in this browser's local storage and is used to look up artwork by movie title. Without a key, the app still works and shows designed title-card fallbacks.

Get an API key from https://www.themoviedb.org/settings/api. Artwork is provided by TMDB; this product is not endorsed or certified by TMDB. Add the required TMDB attribution in any public deployment.

## Recommendation engine
- TF-IDF text features from movie titles and genres
- Item-item K-nearest-neighbours collaborative filtering trained on MovieLens user ratings
- K-Means movie clustering
- Mood-to-genre matching, IMDb rating and vote-confidence ranking
- Optional liked-movie profile to personalize recommendations

## Included pages
- **Discover:** mood selection, recommendations, search, like-to-build-taste-profile, movie posters and hero backdrop.
- **Movie Wheel:** a dedicated page with a mood-based wheel and a separate pure-random mode.

## Data notes
MovieLens latest-small and IMDb public title ratings are downloaded into CSV files by `setup_data.py`; no relational database is used. Mood labels are genre-based discovery tags, not psychological or clinical classifications. IMDb ratings may be missing for some MovieLens titles.

## Sources and attribution
MovieLens: GroupLens Research, University of Minnesota — https://grouplens.org/datasets/movielens/
IMDb datasets: https://developer.imdb.com/non-commercial-datasets/
TMDB images/API: https://www.themoviedb.org/ and https://developer.themoviedb.org/
Respect each provider's terms and attribution requirements. IMDb datasets are provided for personal/non-commercial use under IMDb's stated terms.