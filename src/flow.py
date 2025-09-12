from chroma import Chroma
from config import Configuration
from embedder import Embedder
from reranker import Reranker
from data_reader import DataReader
from gpt import Gpt

class Flow:

    def __init__(self, config: Configuration) -> None:
        self._reader = DataReader(config.input)
        self._chroma = Chroma(config.chroma_dir, config.collection_name, config.clear_collection)
        self._embedder = Embedder(
            config.embedder_model_name,
            self._chroma.collection
        )
        self._reranker = Reranker(config.rerank_model_name)
        self._context_results_count = config.context_results_count
        self._refined_context_results_count = config.refined_context_results_count
        self._gpt = Gpt(config.gpt_model_name)
        self._user_logs = []


    def read_and_embed_logs(self) -> None:
        self._user_logs = self._reader.read_comments()
        self._embedder.embed_messages(self._user_logs)


    def ask_question(self, question: str):

        # Query the DB
        results = self._chroma.collection.query(
            query_texts=[question],
            n_results=self._context_results_count  # number of most relevant comments to return
        )

        if not (results and results["documents"] and results["metadatas"]):
            print("No results")
            return
    
        candidates = results["documents"][0]
        # metadatas = results["metadatas"][0]

        # --- Re-ranking stage ---
        pairs = [(question, doc) for doc in candidates]
        top_indexes = self._reranker._get_closest_indexes(pairs, top_k=self._refined_context_results_count)

        contexts = [candidates[i] for i in top_indexes]
        # metas = [metadatas[i] for i in top_indexes]
        answer = self._gpt.generate_answer(question, contexts)
        print("\n\n=== Answer ===\n")
        print(answer)


if __name__ == "__main__":
    config = Configuration.default_config()
    flow = Flow(config)
    flow.read_and_embed_logs()
    flow.ask_question("What's the weather like for Alice?")
