#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

column_names=('user_id','item_id','rating','timestamp')
df = pd.read_csv('file.tsv', sep='\t', names=column_names)
# Check the head of the data
ratings_data.head()


# In[20]:


# Check out all the movie titles
movie_titles=pd.read_csv('Movie_Id_Titles.xls')
movie_titles.head()


# In[21]:


data = pd.merge(df, movie_titles, on='item_id') 
data.head() 


# In[25]:


# Calculate mean rating of all movies 
data.groupby('title')['rating'].mean().sort_values(ascending=False).head() 


# In[26]:


# Calculate count rating of all movies 
data.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[27]:


# creating dataframe with 'rating' count values 
ratings = pd.DataFrame(data.groupby('title')['rating'].mean())  
  
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count()) 
  
ratings.head() 


# In[33]:


moviemat = data.pivot_table(index ='user_id', 
              columns ='title', values ='rating') 
  
moviemat.head()


# In[34]:


ratings.sort_values('num of ratings', ascending = False).head(10) 


# In[36]:


# analysing correlation with similar movies 
starwars_user_ratings = moviemat['Star Wars (1977)'] 
liarliar_user_ratings = moviemat['Liar Liar (1997)'] 
  
starwars_user_ratings.head() 


# In[37]:


# analysing correlation with similar movies 
similar_to_starwars = moviemat.corrwith(starwars_user_ratings) 
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings) 
  
corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation']) 
corr_starwars.dropna(inplace = True) 
  
corr_starwars.head() 


# In[38]:


# Similar movies like starwars 
corr_starwars.sort_values('Correlation', ascending = False).head(10) 
corr_starwars = corr_starwars.join(ratings['num of ratings']) 
  
corr_starwars.head() 
  
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head() 


# In[39]:


# Similar movies as of liarliar 
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation']) 
corr_liarliar.dropna(inplace = True) 
  
corr_liarliar = corr_liarliar.join(ratings['num of ratings']) 
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending = False).head() 


# In[52]:


def predict_movies (movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns= ['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    prediction=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return prediction    


# In[53]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[54]:


predict_my_movie.head()


# In[57]:


predict_my_movie=predict_movies("River Wild, The (1994)")


# In[59]:


predict_my_movie.head()


# In[ ]:




