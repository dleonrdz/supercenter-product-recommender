from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_faiss_index(df, transformer, name):
    #model = SentenceTransformer(transformer)
    model=HuggingFaceEmbeddings(model_name=transformer)
    path = os.path.join(PROJECT_ROOT, f'vectorstores/{name}')
    vector_store = FAISS.from_texts(df['text_feature'].tolist(), model)
    vector_store.save_local(path)

def load_faiss_index(path, transformer):
    model = HuggingFaceEmbeddings(model_name=transformer)
    return FAISS.load_local(path, model)