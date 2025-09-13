from src.chroma import Chroma
from src.config import Configuration
from src.embedder import Embedder
from src.reranker import Reranker
from src.data_reader import DataReader
from src.gpt import Gpt

class Flow:

    def __init__(self, config: Configuration) -> None:
        self._reader = DataReader(config.input)
        self._chroma = Chroma(config.chroma_dir, config.collection_name, config.clear_collection)
        self._embedder = Embedder(config.embedder_model_name)
        self._reranker = Reranker(config.rerank_model_name)
        self._context_results_count = config.context_results_count
        self._refined_context_results_count = config.refined_context_results_count
        self._gpt = Gpt(config.gpt_model_name)
        self._user_logs = []


    def read_and_embed_logs(self) -> None:
        self._user_logs = self._reader.read_logs()
        embeded_results = self._embedder.embed_messages(self._user_logs)
        self._chroma.add_documents(embeded_results)


    def ask_question(self, question: str):

        # Query the DB
        embedded_question = self._embedder.embed_question(question)
        results = self._chroma.find_similar(
            embeddings=[embedded_question],
            where=None, # for now
            n_results=self._context_results_count  # number of most relevant comments to return
        )

        if len(results) == 0:
            print("No results")
            return
    
        candidates = [r.document for r in results]
        # metadatas = results["metadatas"][0]

        # --- Re-ranking stage ---
        pairs = [(question, doc) for doc in candidates]
        top_indexes = self._reranker._get_closest_indexes(pairs, top_k=self._refined_context_results_count)

        contexts = [candidates[i] for i in top_indexes]
        # metas = [metadatas[i] for i in top_indexes]
        answer = self._gpt.generate_answer(question, contexts)
        print("\n\n=== Answer ===\n")
        print(answer)

