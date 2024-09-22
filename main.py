import re
import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
from sentence_transformers import SentenceTransformer

# Initialize SQLite database
conn = sqlite3.connect('movies.db')
c = conn.cursor()

# Load and display the dataset
@st.cache
def load_data():
    return pd.read_csv("imdb_top_1000.csv")

movies = load_data()

st.title("Movie Embedding and Search App")
st.write("This app creates embeddings for movie titles and overviews, then stores them in a SQLite database for cosine similarity search.")

# Show the first 10 rows of the dataset
st.write(movies[["Series_Title", "Overview"]].head(10))

# Function to clean text
def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text

# Apply the cleaning function to both Series_Title and Overview
columns_to_use = ["Series_Title", "Overview"]
movies["Series_Title_Clean"] = movies["Series_Title"].apply(clean_text)
movies["Overview_Clean"] = movies["Overview"].apply(clean_text)
movies["Text_For_Embedding"] = movies["Series_Title_Clean"] + " " + movies["Overview_Clean"]

# Display cleaned text
st.write("Cleaned Text for Embedding:")
st.write(movies[["Series_Title", "Overview", "Text_For_Embedding"]].head(10))

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Embed the text
@st.cache
def embed_text(text_list):
    return [model.encode(text) for text in text_list]

movies["Embeddings"] = embed_text(movies["Text_For_Embedding"])

# Function to create SQLite table if it doesn't exist
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

# Function to insert movies into the SQLite database
def insert_into_db(movies):
    for i, row in movies.iterrows():
        embedding_blob = row["Embeddings"].tobytes()  # Convert numpy array to bytes
        c.execute('''
            INSERT INTO movies (Series_Title, Overview, Embeddings) VALUES (?, ?, ?)
        ''', (row["Series_Title"], row["Overview"], embedding_blob))
    conn.commit()

# Create the table before inserting data or querying
create_table()

# Button to trigger the insertion of embeddings into SQLite
if st.button("Insert Movies and Embeddings into SQLite"):
    insert_into_db(movies)
    st.success("Movies and embeddings inserted into SQLite database!")

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to search for similar movies based on the query embedding
def search_by_embedding(query):
    query_embedding = model.encode(clean_text(query))
    c.execute('SELECT Series_Title, Overview, Embeddings FROM movies')
    all_movies = c.fetchall()
    
    # Compare query embedding with each stored embedding
    results = []
    for movie in all_movies:
        title, overview, embedding_blob = movie
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)  # Convert bytes back to numpy array
        similarity = cosine_similarity(query_embedding, stored_embedding)
        results.append((title, overview, similarity))
    
    # Sort results by similarity in descending order
    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5]  # Return top 5 similar movies

# Search functionality
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
