import streamlit as st
import pickle
import pandas as pd
import requests

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])

    recommended_movies=[]
    # recommended_movies_poster=[]
    for i in distances[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
        # recommended_movies_poster.append(fetch_poster(i[0]))
    return recommended_movies

movies_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)

similarity=pickle.load(open('similarity.pkl','rb'))

st.title('movie recommendation system')

selected_movie_name= st.selectbox(
    'Select Movie',
    movies['title'].values
)

if st.button('Recommend'):
    recomendation=recommend(selected_movie_name)
    for i in recomendation:
        st.write(i)

