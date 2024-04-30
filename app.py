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


# Initialize session state variables
if 'display_sidebar' not in st.session_state:
    st.session_state['display_sidebar'] = False
    st.session_state['details_index'] = -1

# Main page setup
st.title('⋅˚₊‧ ଳ⋆.ೃ࿔*:･+˚JELLY\'s MOVIE RECOMMENDER⋅˚₊‧ ଳ⋆.ೃ࿔*:･')
selected_movie = st.selectbox('Type a Movie', options=titles)

if st.button('Recommend'):
    st.session_state['display_sidebar'] = False  # Reset sidebar when showing new recommendations
    recommended_movie_names, recommended_movie_posters, recommended_movie_tags = recommender(selected_movie)
    num_movies = len(recommended_movie_names)    
    cols = st.columns(2 * num_movies)  # Create two columns for each movie
    for i in range(num_movies):
        with cols[2*i]:  # First column for the poster
            st.image(movie_posters[i], width=100)
            st.write(movie_titles[i])
        with cols[2*i+1]:  # Second column for the button
            if st.button("More Info", key=f"info_{i}"):
                if st.session_state['display_sidebar'] and st.session_state['details_index'] == i:
                    # If sidebar is already displaying this movie's info, hide it
                    st.session_state['display_sidebar'] = False
                    st.session_state['details_index'] = -1
                else:
                    # Otherwise, show the correct movie's info in the sidebar
                    st.session_state['display_sidebar'] = True
                    st.session_state['details_index'] = i

# Manage sidebar content based on state
if st.session_state['display_sidebar'] and st.session_state['details_index'] != -1:
    index = st.session_state['details_index']
    with st.sidebar:
        st.image(movie_posters[index], width=200)
        st.write(f"Title: {movie_titles[index]}")
        st.write(f"Tags: {movie_tags[index]}")
