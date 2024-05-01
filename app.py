import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load the data
df = pd.read_csv('data.csv')
titles = df['title'].values
tags = df['tags'].values
links = df['link'].values  # Assuming 'link' is a column in your CSV

# Count Vectorizer for processing tags into feature vectors
cv_tags = CountVectorizer(max_features=5000, stop_words='english')
tag_vectors = cv_tags.fit_transform(tags).toarray()
similarity_tags = cosine_similarity(tag_vectors)

# Tfidf Vectorizer for processing titles into feature vectors
tfidf_titles = TfidfVectorizer(stop_words='english')
title_vectors = tfidf_titles.fit_transform(titles).toarray()
similarity_titles = cosine_similarity(title_vectors)

# API key for TMDB
API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"

# Function to fetch movie posters from TMDB
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}')
    data = response.json()
    poster_path = data.get('poster_path', '')
    return f'https://image.tmdb.org/t/p/w500/{poster_path}' if poster_path else ''

# Function to recommend movies based on a given movie title
def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances_tags = similarity_tags[movie_index]
    distances_titles = similarity_titles[movie_index]
    combined_distances = (distances_tags + distances_titles) / 2  # Combine both similarities
    
    movies_list = sorted(list(enumerate(combined_distances)), reverse=True, key=lambda x: x[1])[1:11]
    recommendations = [(df.iloc[i[0]]['title'], fetch_poster(df.iloc[i[0]]['movie_id']), df.iloc[i[0]]['tags'], df.iloc[i[0]]['link']) for i in movies_list]
    return recommendations

# Streamlit UI Configuration
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('JELLY\'s MOVIE RECOMMENDER')

selected_movie = st.selectbox('Type a Movie', options=titles)

# Display recommended movies and posters when the button is clicked
if st.button('Recommend'):
    recommendations = recommender(selected_movie)
    cols_per_row = 5
    num_movies = len(recommendations)
    
    for i in range(0, num_movies, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            index = i + j
            if index < num_movies:
                col = cols[j]
                title, poster, tag, link = recommendations[index]
                if poster:
                    col.image(poster, use_column_width=True)
                col.text(title)
                with col.expander("More Info"):
                    st.markdown(f"### {title}")
                    st.markdown(f"[More Details]({link})")
                    st.write(f"Overview: {tag}")
