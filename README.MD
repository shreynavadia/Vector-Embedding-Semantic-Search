# Movie Embedding and Search App

This is a Streamlit-based application that performs semantic search on a dataset of movie titles and overviews. It uses the `SentenceTransformer` model to create text embeddings, which are stored in a SQLite database. Users can search for movies by inputting a description, and the app will find the most similar movies based on cosine similarity.

## Features
- Load and display a dataset of the top 1000 IMDb movies.
- Clean and preprocess movie titles and overviews.
- Generate sentence embeddings using the `paraphrase-MiniLM-L6-v2` model from `SentenceTransformer`.
- Store the embeddings and movie information in a SQLite database.
- Search for similar movies using cosine similarity on the embeddings.

## Prerequisites

Make sure you have the following installed on your system:

1. **Python 3.8+**
2. **Streamlit**
3. **Pandas**
4. **Numpy**
5. **Sentence-Transformers**
6. **SQLite (bundled with Python via `sqlite3`)**

### Install Required Libraries

You can install the required libraries using `pip`:

```bash
pip install streamlit pandas numpy sentence-transformers



