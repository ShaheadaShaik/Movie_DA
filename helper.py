import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests

ps = PorterStemmer()

# Convert stringified list of dicts to list of names
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

# Extract director's name from crew data
def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

# Remove spaces from names to make tags consistent
def collapse(L):
    return [i.replace(" ", "") for i in L]

# Apply stemming to tags text
def stem(text):
    return ' '.join([ps.stem(i) for i in text.split()])

# Preprocess the dataset and return necessary objects
def preprocess_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: x[0:3])
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_movies_df = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
    new_movies_df['tags'] = new_movies_df['tags'].apply(lambda x: " ".join(x).lower())
    new_movies_df['tags'] = new_movies_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_movies_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

# Fetch movie poster from TMDB API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# Recommend movies
def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters
