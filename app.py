import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load the data
df = pd.read_csv('data.csv')
titles = df['title'].values
tags = df['tags'].values
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(tags).toarray()
similarity = cosine_similarity(vectors)

# API key for TMDB
API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"

# Function to fetch movie posters from TMDB
def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}'
    )
    data = response.json()
    poster_path = data.get('poster_path', '')
    if not poster_path:
        return ''
    return f'https://image.tmdb.org/t/p/w500/{poster_path}'

# Function to recommend movies based on a given movie title
def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]
    recommended_titles = []
    recommended_posters = []
    recommended_tags = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        recommended_titles.append(df.iloc[i[0]]['title'])
        recommended_posters.append(fetch_poster(movie_id))
        recommended_tags.append(df.iloc[i[0]]['tags'])
    return recommended_titles, recommended_posters, recommended_tags

# Streamlit UI Configuration
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main page setup
st.title('⋅˚₊‧ ଳ⋆.ೃ࿔*:･+˚JELLY\'s MOVIE RECOMMENDER⋅˚₊‧ ଳ⋆.ೃ࿔*:･')
selected_movie = st.selectbox('Type a Movie', options=titles)

# Display recommended movies and posters when the button is clicked
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_tags = recommender(selected_movie)

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
                        # Image and text for each movie
                        col.image(recommended_movie_posters[index], use_column_width=True)
                        col.text(recommended_movie_names[index])
                        # Create an expander to show additional information when clicked
                        with col.expander(f"More Info"):
                            st.markdown.sidebar(f"### {recommended_movie_names[index]}")
                            st.write.sidebar(f"{recommended_movie_tags[index]}")
