import streamlit as st
import helper

# Load and preprocess data
movies, similarity = helper.preprocess_data()

st.header('Movie Recommender System')

# Movie selection dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

# Show recommendations
if st.button('Show Recommendation'):
    names, posters = helper.recommend(selected_movie, movies, similarity)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
