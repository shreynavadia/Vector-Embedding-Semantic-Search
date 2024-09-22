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
Dataset
The app uses an IMDb dataset containing the top 1000 movies. Make sure you have the imdb_top_1000.csv file in the same directory as the app, or update the path accordingly in the code.

How to Run
Clone this repository or download the project files.
Ensure you have the imdb_top_1000.csv file in the same directory as the script.
Open a terminal and navigate to the directory containing your files.
Run the following command to launch the Streamlit app:
bash
Copy code
streamlit run <script_name>.py
Replace <script_name>.py with the actual name of your Python file (e.g., app.py).

How It Works
The app will load and display the first 10 rows of movie titles and overviews.
It cleans the text, combines movie titles and overviews, and creates embeddings using the SentenceTransformer model.
The embeddings are then stored in a SQLite database.
You can insert the movies and embeddings into the database by clicking the "Insert Movies and Embeddings into SQLite" button.
You can search for a movie by providing a description. The app will return the top 5 most similar movies based on cosine similarity between the query and the movie embeddings.
Example
When you search for a movie, the app will return results like this:

yaml
Copy code
**Title**: The Shawshank Redemption
**Overview**: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.
---
**Title**: The Godfather
**Overview**: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.
---
File Structure
bash
Copy code
├── imdb_top_1000.csv        # CSV file containing the movie dataset
├── <script_name>.py         # Python script to run the Streamlit app
└── README.md                # Readme file
Future Improvements
Add support for more complex search queries.
Use a more sophisticated database like PostgreSQL for larger datasets.
Improve the search ranking algorithm.
License
This project is licensed under the MIT License.

markdown
Copy code

### Instructions:

1. Replace `<script_name>.py` with the actual name of your Python file.
2. Place this `README.md` in your repository.

This `README.md` is fully formatted for GitHub and uses appropriate Markdown syntax for sections, code blocks, and file structure.
