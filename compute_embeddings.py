import json
import os
from typing import Dict
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import logging
from logger import setup_logger


logger = setup_logger('embeddings')

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_and_save_embeddings(data_dir: str = "data", output_dir: str = "embeddings"):
    """Compute embeddings for all problems and save them to disk."""
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        logger.info(f"Using device: {device}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for problem_type in os.listdir(data_dir):
            type_dir = os.path.join(data_dir, problem_type)
            if os.path.isdir(type_dir):
                type_output_dir = os.path.join(output_dir, problem_type)
                os.makedirs(type_output_dir, exist_ok=True)
                
                files = [f for f in os.listdir(type_dir) if f.endswith('.json')]
                logger.info(f"Processing {len(files)} files in {problem_type}")
                
                for filename in tqdm(files, desc=f"Processing {problem_type}"):
                    file_path = os.path.join(type_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        encoded_input = tokenizer(data['problem'], 
                                               padding=True, 
                                               truncation=True, 
                                               return_tensors='pt')
                        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                        
                        with torch.no_grad():
                            model_output = model(**encoded_input)
                            embedding = mean_pooling(model_output, encoded_input['attention_mask'])
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                            embedding = embedding[0].cpu().numpy()
                        
                        output_data = {
                            'problem': data['problem'],
                            'solution': data['solution'],
                            'level': data.get('level', 'Unknown'),
                            'type': problem_type,
                            'embedding': embedding.tolist()
                        }
                        
                        output_path = os.path.join(type_output_dir, filename)
                        with open(output_path, 'w') as f:
                            json.dump(output_data, f, indent=2)
                        
                        logger.debug(f"Processed {filename} with shape {embedding.shape}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Completed embedding computation. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    compute_and_save_embeddings() 