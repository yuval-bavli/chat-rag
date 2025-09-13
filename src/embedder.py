from typing import Mapping, Optional, Union, List, Tuple, cast, Any

# Install these packages if you haven't already:
# pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer

from src.embed_result import EmbedResult, Metadata
from src.model_dirs import get_model_dir
from src.user_log import UserLog


class Embedder:

    def __init__(self, embedder_model_name: str) -> None:
        self._embedder = self._get_embedder_model(embedder_model_name)


    def embed_question(self, question: str) -> list[float]:
        embedding = self._embedder.encode([question], show_progress_bar=True, convert_to_numpy=True).tolist()
        return embedding[0]


    def embed_messages(self, user_logs: List[UserLog]) -> list[EmbedResult]:
        print("Creating embeddings...")
        texts = [f"{c.name}: {c.message}" for c in user_logs]
        # texts = [f"At {c.timestamp}, {c.name} wrote: {c.message}" for c in user_logs]
        # texts = [c.message for c in user_logs]
        # request numpy output so we can easily convert to lists for Chroma
        embeddings: list[list[float]] = self._embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

        ids = [ul.id for ul in user_logs]

        # <-- annotate metadatas as List[Metadata] so the typechecker sees the right type
        metadatas: List[Metadata] = [
            {
                "name": ul.name,
                "timestamp": ul.timestamp.isoformat(),
                "type": "question" if "?" in ul.message else ""
            }
            for ul in user_logs
        ]

        results: list[EmbedResult] = []
        for i in range(len(ids)):
            results.append(EmbedResult(
                id=ids[i],
                document=texts[i],
                metadata=metadatas[i],
                embedding=embeddings[i]
            ))

        print("Embeddings created")

        return results


    def _get_embedder_model(self, model_name: str) -> SentenceTransformer:
        model_path = get_model_dir(model_name)
        print(f"Loading embedding model {model_name} from {model_path}...")
        embedder = SentenceTransformer(model_path)
        print(f"Embedding model {model_name} loaded")
        return embedder

