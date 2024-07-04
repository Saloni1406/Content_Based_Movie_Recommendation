import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import difflib #To get the most closest match of movie given by the user
from sklearn.feature_extraction.text import TfidfVectorizer #Convert text data into numeric data
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title = "Movie_Recommendation", page_icon = ":Movie:", layout = "centered", )
st.title("🎬 Movie Recommendation System")
st.header("Movie marathon alert! 10 picks based on your last watch")

#Loading Dataset
#Caching the dataset, so that whenever the page reruns, the data won't be loaded again and again
@st.cache_data 
def load_data():
    df = pd.read_csv("movies.csv")
    return df
df = load_data()

#Taking Input(Movie) from user
movies_list = df['title'].unique().tolist()
movie_name  = st.selectbox("Enter you last watched movie...", movies_list)

#replacing the null values with null string
selected_features = ['genres','keywords','tagline','cast','director']
for feature in selected_features:
  df[feature] = df[feature].fillna('')

#combining all the 5 selected features
combined_features =df['genres']+' '+ df['keywords']+ ' '+ df['tagline']+' '+ df['cast']+' '+ df['director']

#converting the text data to numeric data
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vector)

#Finding the index of the movie based on the title 'x'
index = df[df.title == movie_name]['index'].values[0]

#Printing similar movies based on similarity score
sim = list(enumerate(similarity[index]))

#Sorting the movies based on there similarity socre
sort = sorted(sim, key = lambda x:x[1], reverse = True)

#Printing the movies similar to the movies entered by the user
st.write("SUGGESTED MOVIES ARE:")
i = 1
for movie in sort:
  index = movie[0]
  title  = df[df.index == index]['title'].values[0]
  if(i<11):
    st.write(i,". ",title)
    i = i+1
