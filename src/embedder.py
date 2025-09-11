from typing import Mapping, Optional, Union

# Install these packages if you haven't already:
# pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from user_log import UserLog
from models.model_dirs import get_model_dir


Metadata = Mapping[str, Optional[Union[str, int, float, bool]]]


class Embedder:

    def __init__(self, embedder_model_name: str, str, collection: Collection) -> None:
        self._embedder = self._get_embedder_model(embedder_model_name)
        self._collection = collection


    def embed_messages(self, user_logs: list[UserLog]) -> None:
        print("Creating embeddings...")
        texts = [f"At {c.timestamp}, {c.name} wrote: {c.message}" for c in user_logs]
        embeddings = self._embedder.encode(texts, show_progress_bar=True)

        ids = [ul.id for ul in user_logs]

        # <-- annotate metadatas as List[Metadata] so the typechecker sees the right type
        metadatas: list[Metadata] = [
            {
                "name": ul.name,
                "timestamp": ul.timestamp.isoformat(),
                "type": "question" if "?" in ul.message else ""
            }
            for ul in user_logs
        ]

        print(f"Adding to ChromaDB {len(ids)} entries...")
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        print("Done! ChromaDB updated.")


    def _get_embedder_model(self, model_name: str) -> SentenceTransformer:
        model_path = get_model_dir(model_name)
        print(f"Loading embedding model {model_name} from {model_path}...")
        embedder = SentenceTransformer(model_path)
        print(f"Embedding model {model_name} loaded")
        return embedder

