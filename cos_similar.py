import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# helper functions


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


df = pd.read_csv("movie_dataset.csv")
# print(df.columns)

# selecting features\
features = ['keywords', 'cast', 'genres', 'director']

# combining all the columns of features using df
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']


df["combine_features"] = df.apply(combine_features, axis=1)

print("combined_features:", df["combine_features"].head())

# creating a counting matrix for this column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combine_features"])

# creating a cosine similartiy

cosine_sim = cosine_similarity(count_matrix)
movie_user_like = "Avatar"

# getting the movie index from its title
movie_index = get_index_from_title(movie_user_like)

similar_movies = list(enumerate(cosine_sim[movie_index]))

# getting the similar movies in descending order
sorted_similar_movies = sorted(
    similar_movies, key=lambda x: x[1], reverse=True)

# print all these movies
i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i > 50:
        break
