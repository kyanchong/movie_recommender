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

st.title('⋅˚₊‧ ଳ⋆.ೃ࿔*:･+˚JELLY\'s MOVIE RECOMMENDER⋅˚₊‧ ଳ⋆.ೃ࿔*:･')

selected_movie = st.selectbox('Type a Movie', options=titles)

# Initialize session state
if 'last_clicked' not in st.session_state:
    st.session_state.last_clicked = None

# Display recommended movies and posters with additional information
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_tags = recommender(selected_movie)
    num_movies = len(recommended_movie_names)
    cols_per_row = 3  # Adjust number of columns per row based on your layout preference

    for i in range(0, num_movies, cols_per_row):
        with st.container():
            cols = st.columns([1, 1] * cols_per_row)  # Create two columns for each movie, repeated for each movie in the row
            for j in range(0, cols_per_row * 2, 2):  # Increment by 2 to process pairs of columns
                index = i + (j // 2)
                if index < num_movies:
                    with cols[j]:  # First column for the poster
                        st.markdown(f"#### {recommended_movie_names[index]}")
                        st.image(recommended_movie_posters[index], use_column_width=True)
                    with cols[j+1]:  # Second column for the expander
                        expander = st.expander("More Info")
                        if expander:
                            # Toggle details in the sidebar
                            if st.session_state.last_clicked != index:
                                st.session_state.last_clicked = index
                                with st.sidebar:
                                    st.markdown(f"**Title:** {recommended_movie_names[index]}")
                                    st.markdown(f"**Tags:** {recommended_movie_tags[index]}")
                                    st.image(recommended_movie_posters[index], use_column_width=True)
                            else:
                                # Clear the sidebar if the same button is clicked again
                                st.session_state.last_clicked = None
                                for key in st.sidebar.session_state.keys():
                                    del st.sidebar.session_state[key]
