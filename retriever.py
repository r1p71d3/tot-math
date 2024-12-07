from __future__ import annotations
from typing import List
from dataclasses import dataclass
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pydantic_ai import RunContext
from models import MathDependencies
from logger import setup_logger

logger = setup_logger('retriever')

@dataclass
class MathExample:
    problem: str
    solution: str
    level: str
    type: str
    embedding: np.ndarray
    similarity: float = 0.0

    def __lt__(self, other):
        return self.similarity < other.similarity

class MathRetriever:
    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = embeddings_dir
        self.examples: List[MathExample] = []
        self.load_embeddings()
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def load_embeddings(self):
        """Load pre-computed embeddings."""
        try:
            for problem_type in os.listdir(self.embeddings_dir):
                type_dir = os.path.join(self.embeddings_dir, problem_type)
                if os.path.isdir(type_dir):
                    for filename in os.listdir(type_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(type_dir, filename)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                example = MathExample(
                                    problem=data['problem'],
                                    solution=data['solution'],
                                    level=data['level'],
                                    type=data['type'],
                                    embedding=np.array(data['embedding'])
                                )
                                self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} examples with embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query using the same model used for examples."""
        try:
            encoded_input = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # mean pooling
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                # normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                return embedding[0].cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            raise

    def get_similar_examples(self, query: str, k: int = 3) -> List[MathExample]:
        """Retrieve k most similar examples to the query."""
        try:
            query_embedding = self.encode_query(query)
            
            # calculate similarities and store them
            examples_with_scores = []
            for example in self.examples:
                similarity = np.dot(query_embedding, example.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(example.embedding)
                )
                example_copy = MathExample(
                    problem=example.problem,
                    solution=example.solution,
                    level=example.level,
                    type=example.type,
                    embedding=example.embedding,
                    similarity=similarity
                )
                examples_with_scores.append(example_copy)
            
            # sort by sim and get top k
            examples_with_scores.sort(reverse=True)
            top_k = examples_with_scores[:k]
            
            logger.info(f"Retrieved {k} similar examples for query: \n" + "\n".join([
                f"- {example.problem}" 
                for example in top_k
            ]))
            return top_k
            
        except Exception as e:
            logger.error(f"Error retrieving similar examples: {str(e)}")
            return []


def retrieve_similar(ctx: RunContext[MathDependencies], query: str, k: int = 3) -> str:
    """Tool for retrieving similar examples."""
    try:
        examples = ctx.deps.retriever.get_similar_examples(query, k)
        
        results = []
        for example in examples:
            results.append(
                f"Problem: {example.problem}\n"
                f"Solution: {example.solution}\n"
                f"Type: {example.type}\n"
            )
        
        return "\n---\n".join(results)
        
    except Exception as e:
        logger.error(f"Error retrieving similar examples: {str(e)}")
        return "Error retrieving similar examples" 