import re
import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
from sentence_transformers import SentenceTransformer

conn = sqlite3.connect('movies.db')
c = conn.cursor()

# Use st.cache_data to cache data like CSV loading
@st.cache_data
def load_data():
    return pd.read_csv("imdb_top_1000.csv")

movies = load_data()

st.title("Movie Embedding and Search App")
st.write("This app creates embeddings for movie titles and overviews, then stores them in a SQLite database for cosine similarity search.")

st.write(movies[["Series_Title", "Overview"]].head(10))

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text

columns_to_use = ["Series_Title", "Overview"]
movies["Series_Title_Clean"] = movies["Series_Title"].apply(clean_text)
movies["Overview_Clean"] = movies["Overview"].apply(clean_text)
movies["Text_For_Embedding"] = movies["Series_Title_Clean"] + " " + movies["Overview_Clean"]

st.write("Cleaned Text for Embedding:")
st.write(movies[["Series_Title", "Overview", "Text_For_Embedding"]].head(10))

# Use st.cache_resource to cache heavy resources like models
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Cache the embeddings generation function using st.cache_data
@st.cache_data
def embed_text(text_list):
    return [model.encode(text) for text in text_list]

movies["Embeddings"] = embed_text(movies["Text_For_Embedding"])

def create_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Series_Title TEXT,
            Overview TEXT,
            Embeddings BLOB
        )
    ''')
    conn.commit()

def insert_into_db(movies):
    for i, row in movies.iterrows():
        embedding_blob = row["Embeddings"].tobytes()
        c.execute('''
            INSERT INTO movies (Series_Title, Overview, Embeddings) VALUES (?, ?, ?)
        ''', (row["Series_Title"], row["Overview"], embedding_blob))
    conn.commit()

create_table()

if st.button("Insert Movies and Embeddings into SQLite"):
    insert_into_db(movies)
    st.success("Movies and embeddings inserted into SQLite database!")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def search_by_embedding(query):
    query_embedding = model.encode(clean_text(query))
    c.execute('SELECT Series_Title, Overview, Embeddings FROM movies')
    all_movies = c.fetchall()
    
    results = []
    for movie in all_movies:
        title, overview, embedding_blob = movie
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(query_embedding, stored_embedding)
        results.append((title, overview, similarity))
    
    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5]

st.header("Search Movies by Overview")
query_text = st.text_input("Enter a movie overview or description:")

if st.button("Search"):
    results = search_by_embedding(query_text)
    if results:
        st.write("Top 5 Similar Movies:")
        for result in results:
            st.write(f"**Title**: {result[0]}")
            st.write(f"**Overview**: {result[1]}")
            st.write("---")
    else:
        st.write("No similar movies found.")
