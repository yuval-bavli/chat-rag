from typing import NamedTuple


VECTOR_MODEL_NAME1 = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
VECTOR_MODEL_NAME2 = "intfloat/multilingual-e5-small"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
GPT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JSON_FILE = R"C:\Users\yuval\src\facebook_summarizer\input\comments.json"
CHROMA_DIR = "chroma_db"     # folder to store Chroma DB
VECTOR_MODEL_NAME = VECTOR_MODEL_NAME2
COLLECTION_NAME = "facebook_comments"

N_RESULTS = 2


class Configuration(NamedTuple):
    input: str
    chroma_dir: str
    embedder_model_name: str
    rerank_model_name: str
    gpt_model_name: str
    collection_name: str
    n_results: int
    clear_collection: bool


    @staticmethod
    def default_config():
        return Configuration(JSON_FILE, CHROMA_DIR, VECTOR_MODEL_NAME, RERANK_MODEL_NAME, GPT_MODEL_NAME, COLLECTION_NAME, N_RESULTS, True)
