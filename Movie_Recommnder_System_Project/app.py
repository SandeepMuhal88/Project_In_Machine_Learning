import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np


movies_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
similarity=pickle.load(open('similarity.pkl','rb'))
st.title("Movie Recommander System")

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
#     responce=requests.get(url)
#     data=responce.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    # recommended_movies_poster=[]
    for i in movies_list:
        # movie_id=movies.iloc[i[0]].movie_id
        # fatch poster from API
        recommended_movies.append(movies.iloc[i[0]].title)
        # recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies #recommended_movies_poster

selected_movie_name = st.selectbox(
    "Enter movie name your interrest?",
    movies['title']
)


if st.button('Recommend'):
    names=recommend(selected_movie_name)
    for i in names:
        st.write(i)
    # col1, col2, col3, col4, col5 = st.beta_columns(5)
    # with col1:
    #     st.text(names[0])
    #     st.image(posters[0])
    # with col2:
    #     st.text(names[1])
    #     st.image(posters[1])

    # with col3:
    #     st.text(names[2])
    #     st.image(posters[2])
    # with col4:
    #     st.text(names[3])
    #     st.image(posters[3])
    # with col5:
    #     st.text(names[4])
    #     st.image(posters[4])
