from src.config import settings
from google import genai

import dspy
import numpy as np

client = genai.Client(api_key=settings.GEMINI_API_KEY)

def configure_llm():
    lm = dspy.LM(
        model=f"gemini/{settings.LLM_MODEL}",
        api_key=settings.GEMINI_API_KEY,
        temperature=0.6,
        max_tokens=4000,
    )
    dspy.configure(lm=lm)

def generate_embedding(content: str): 
    result = [
        np.array(e.values) for e in client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=content,
            config={
                'output_dimensionality': 1536,
                'task_type': "SEMANTIC_SIMILARITY"
            }).embeddings
    ]
    embeddings_matrix = np.array(result)
    
    return embeddings_matrix

