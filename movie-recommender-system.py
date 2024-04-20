import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# print(movies.head())
# print(credits.head())

movies=movies.merge(credits,on='title')
print(movies.shape)

#we will not include the following attributes as it does not contribute to the recomendation system
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)
#others

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(movies.head())

#we will combine 'overview','genres','keywords','cast','crew' and made them as one column named keyword

# print(movies.isnull().sum())
print(movies.dropna(inplace=True))
# print(movies.isnull().sum())

print(movies.iloc[0].genres)
#[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}] we should convert this text in below format
#['Action','scify','adventure']

#see ast module in python also literal eval function

import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

movies['genres'] = movies['genres'].apply(convert)
print(movies['genres'])

movies['keywords'] = movies['keywords'].apply(convert)
print(movies['keywords'])

def convert3(text):#for cast we chose only top three cast to reduce redundancy
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

movies['cast'] = movies['cast'].apply(convert3)
print(movies['cast'])

def fetch_director(text):# for crew we only fetch director name
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

movies['crew'] = movies['crew'].apply(fetch_director)
print(movies['crew'])

#overview is in string format so we it convert into list
# print(movies['overview'])
movies['overview'] = movies['overview'].apply(lambda x: x.split())


def collapse(L):# we should remove the spaces between names
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
print(movies.head())
new_df=movies[['movie_id','title','tags']]

print(movies['tags'].dtype)

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))#to convert list into string
# print(new_df.head())

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
print(new_df.head())

import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)
# print(new_df['tags'])

#to find similarity between two songs we use vectorization method by converting each songs into vectors
#we use different methods such as bag-of-words,tf-idf etc (we have used bag-of-words in this project)
#combine all the tags and find the most frequent words in the comboned tags


#M |w1|w2|w3|..........w3000(use can chose any number of words according to our choices)
#m1|4|3|1.......
#m2|4|0|9..........
#avoid stopwords(and,or etc etc)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()
# print(vector)

# print(cv.get_feature_names_out())
# print(ps.stem('loved'))

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

print(similarity.shape)
print(similarity[1])#first movie has 1 cosine similarity with itself

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
print(recommend('Avatar'))

import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))