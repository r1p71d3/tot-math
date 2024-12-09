## Setup 
1. Install the requirements

`pip install -r requirements.txt ` 

2. Download the MATH dataset [here](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar)
3. Put the contents of the `train` directory into the `data` directory in the project folder
4. Create the problem embeddings for RAG

`python compute_embeddings.py`

5. Run the app 

`streamlit run app.py`
