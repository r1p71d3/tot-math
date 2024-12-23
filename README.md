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
To test the model on arbitrary problems you can use the Streamlit GUI:

`streamlit run app.py`

To evaluate on the 50 problem benchmark from the MATH evaluation set:

`python main.py`

**IMPROTANT:** running the benchmark will very likely result in double digit API costs.

## Results
Our runs suggested the benchmark accuracy of around 84%. You can view the results of other GPT agents [here](https://github.com/emilyworks/tot/blob/gpt/README.md) and the Llama models [here](https://github.com/emilyworks/tot/blob/main/README.md) in their respective **Results** sections.
