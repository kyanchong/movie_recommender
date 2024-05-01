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

# API key for TMDB
API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"

# Function to fetch movie posters from TMDB
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}')
    data = response.json()
    poster_path = data.get('poster_path', '')
    return f'https://image.tmdb.org/t/p/w500/{poster_path}' if poster_path else ''

def recommender(movie):
    movie_index = df[df['title'].str.lower() == movie.lower()].index[0]  # Case insensitive search
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])
    
    # Include the searched movie in the results if desired
    if movie_index not in [m[0] for m in movies_list[:1]]:  # Check if it's already at the top
        movies_list.insert(0, (movie_index, 1.0))  # Explicitly add with max similarity
    
    recommended_titles = []
    recommended_posters = []
    recommended_tags = []
    recommended_links = []
    
    for i in movies_list[1:11]:  # Skip the first one since it will be the searched movie itself
        idx = i[0]
        recommended_titles.append(df.iloc[idx]['title'])
        recommended_posters.append(fetch_poster(df.iloc[idx]['movie_id']))
        recommended_tags.append(df.iloc[idx]['tags'])
        recommended_links.append(df.iloc[idx]['link'])  # Retrieve link for each movie

    return recommended_titles, recommended_posters, recommended_tags, recommended_links

# Streamlit button to show recommendations
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_tags, recommended_links = recommender(selected_movie)
    recommended_movie_names.insert(0, selected_movie)  # Insert searched movie title at the top
    recommended_movie_posters.insert(0, fetch_poster(df[df['title'] == selected_movie]['movie_id'].values[0]))  # Insert searched movie poster
    recommended_movie_tags.insert(0, df[df['title'] == selected_movie]['tags'].values[0])  # Insert searched movie tags
    recommended_links.insert(0, df[df['title'] == selected_movie]['link'].values[0])  # Insert searched movie link

    num_movies = len(recommended_movie_names)
    cols_per_row = 5  # 5 columns per row

    for i in range(0, num_movies, cols_per_row):
        with st.container():
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                index = i + j
                if index < num_movies:
                    col = cols[j]
                    if recommended_movie_posters[index]:
                        col.image(recommended_movie_posters[index], use_column_width=True)
                        col.text(recommended_movie_names[index])
                        # Create an expander to show additional information when clicked
                        with col.expander(f"More Info"):
                            st.markdown(f"### {recommended_movie_names[index]}")
                            st.markdown(f"[More Details]({recommended_links[index]})")
                            st.write(f"Overview: {recommended_movie_tags[index]}")
