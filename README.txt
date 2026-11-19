CN6005 Artificial Intelligence - Group Coursework (Source Code)
================================================================

Contents
--------
Task1_ImageClassification.ipynb  ANN vs CNN (plus MobileNetV2 extension) on the Garbage Classification dataset
Task2_TextMining.ipynb           Pre-processing, TF-IDF, K-Means and Sentence-BERT clustering of BBC News articles
Task3_AIAgent.ipynb              CineGuide movie recommendation agent: design, live demo and evaluation
movie_agent/                     Python package used by Task 3 (retriever, tools, LLM client, agent loop)
app.py                           Web chat interface for the agent (Streamlit)
tests/                           Unit tests for the agent (run offline)
datasets/links.txt               Links to all datasets
screenshots/                     Figures from every notebook (created automatically when run)
requirements.txt                 All libraries

Requirements
------------
Python 3.10 - 3.12.  Install the libraries:
    pip install -r requirements.txt

How to run
----------
1. Datasets: the notebooks download them automatically with kagglehub.
   A free Kaggle account is needed - put your kaggle.json API token in ~/.kaggle/
   (or set KAGGLE_USERNAME and KAGGLE_KEY). Alternatively download the datasets
   manually from datasets/links.txt and set GARBAGE_DATA_DIR / BBC_DATA_DIR / TMDB_DATA_DIR.
2. Open the notebooks with Jupyter (`jupyter notebook`) or Google Colab and run all cells
   in order. A GPU runtime is recommended for Task 1 (~25 min on a T4) and Task 2.
3. Task 3 needs an LLM API key. Copy .env.example to .env and add a free GROQ_API_KEY
   (https://console.groq.com/keys) or GEMINI_API_KEY. On Colab, add it as a secret.
   Without a key, the agent still runs in retrieval-only mode.
4. Chat with the agent:   streamlit run app.py      or   python -m movie_agent
5. Run the unit tests:    pytest -q

Notes
-----
- Random seeds are fixed to 42 for reproducibility (GPU training can still vary slightly).
- All figures are saved to screenshots/task1, screenshots/task2 and screenshots/task3.
