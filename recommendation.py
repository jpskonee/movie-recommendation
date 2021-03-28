#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#importing the dataset 
df_credit=pd.read_csv("C:/Users/NifemiDev/Downloads/tmdb_5000_credits (1).csv")


# In[3]:


df=pd.read_csv("C:/Users/NifemiDev/Downloads/tmdb_5000_movies.csv")


# In[4]:


df.head(1)


# In[5]:


df.columns


# In[6]:


df_credit.head(5)


# In[7]:


#convert ids to int. required for merging 
df_credit['movie_id']= df_credit['movie_id'].astype('int')


# In[8]:


#convert ids to int. required for merging 
df['id']= df['id'].astype('int')


# In[9]:


#merge keywords and credits into the df 
df=df.merge(df_credit,on='title')


# In[10]:


#overview
df.head(2)


# In[11]:


# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)


# In[12]:


# Import Numpy
import numpy as np


# In[13]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[14]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[15]:


# Define new director, cast, genres and keywords features that are in a suitable form.
df['director'] = df['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(get_list)


# In[16]:


# Print the new features of the first 3 films
df[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[17]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[18]:


# Apply clean_data function to your features.
features = ['cast','director','keywords', 'genres']

for feature in features:
    df[feature] = df[feature].apply(clean_data)


# In[19]:


# Apply clean_data function to your features.
features = ['cast','director','keywords', 'genres']

for feature in features:
    df[feature] = df[feature].apply(clean_data)


# In[20]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


# In[21]:


# Create a new soup feature
df['soup'] = df.apply(create_soup, axis=1)


# In[22]:


df[['soup']].head(2)


# In[23]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])


# In[24]:


count_matrix.shape


# In[25]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[26]:


# Reset index of your main DataFrame and construct reverse mapping as before
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])


# In[27]:


def get_recommendations(title, cosine_sim=cosine_sim2):
    #getting the indice that matches the movie
    idx=indices[title]
    
    #getting the pairwise similiarity scoresof all movies
    sim_scores=list(enumerate(cosine_sim[idx]))
    #sort the movies based on similarities scores
    sim_scores=sorted(sim_scores, key=lambda x:x[1],reverse=True)
    
    #get the scores of the 10 most similar movies 
    sim_scores=sim_scores[1:11]
    
    #get the movie indices
    movie_indices=[i[0] for i in sim_scores]
    
    movie_director =df['director'].iloc[movie_indices]
    movie_title = df['title'].iloc[movie_indices]
    movie_genres = df['genres'].iloc[movie_indices]

    recommendation_data = pd.DataFrame(columns=['Director','Name','Genres'])

    recommendation_data['Director'] = movie_director
    recommendation_data['Name'] = movie_title
    recommendation_data['Genres'] = movie_genres

    return recommendation_data
    


# In[28]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[29]:


def results(movie_names):
    movie_name =movie_name.lower()
    find_movie=df
    transform_result=transform_data(find_movie)
    if movie_name not in find_movie['title'].unique():
        return 'movie not in Database'
    else:
        recommendations=get_recommendations(title, find_movie, transform_result)
        return recommendations.to_dict('records')
        


# In[31]:





# In[ ]:




