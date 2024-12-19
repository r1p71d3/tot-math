## About
This is an implementation of the Tree of Thought algorithm that utilizes RAG and math tools. For the basic GPT and the Llama quantization & LoRA verions, please refer to the `gpt` and `main` branches of the following repository: https://github.com/emilyworks/tot.

## Setup 
1. Install the requirements

`pip install -r requirements.txt ` 

2. Download the MATH dataset [here](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar)
3. Put the contents of the `train` directory into the `data` directory in the project folder
4. Create the problem embeddings for RAG

`python compute_embeddings.py`

5. Export your OpenAI API key as an environment variable or place it in an `.env` file in the project directory

`export OPENAI_API_KEYU="your api key"`

(**NOTE**: by the nature of the ToT framework, it results in a considerable token usage, with costs as high as $0.5 - $1.2 per query.)

## Run
You interact with the GUI by running the Streamlit app:

`streamlit run app.py`
