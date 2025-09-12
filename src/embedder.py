from typing import Mapping, Optional, Union, List, Tuple, cast, Any

# Install these packages if you haven't already:
# pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from model_dirs import get_model_dir
from user_log import UserLog


Metadata = Mapping[str, Optional[Union[str, int, float, bool]]]


class Embedder:

    def __init__(self, embedder_model_name: str, collection: Collection) -> None:
        self._embedder = self._get_embedder_model(embedder_model_name)
        self._collection = collection


    def embed_messages(self, user_logs: List[UserLog]) -> None:
        print("Creating embeddings...")
        texts = [f"At {c.timestamp}, {c.name} wrote: {c.message}" for c in user_logs]
        # request numpy output so we can easily convert to lists for Chroma
        embeddings = self._embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

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

        print(f"Adding to ChromaDB {len(ids)} entries...")
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        print("Done! ChromaDB updated.")


    def _get_embedder_model(self, model_name: str) -> SentenceTransformer:
        model_path = get_model_dir(model_name)
        print(f"Loading embedding model {model_name} from {model_path}...")
        embedder = SentenceTransformer(model_path)
        print(f"Embedding model {model_name} loaded")
        return embedder


    def find_similar(self, query: str, person_name: Optional[str] = None, n_results: int = 5) -> List[Tuple[str, str]]:
        """
        Find top-N semantically similar messages to `query`.

        If `person_name` is provided, results will be filtered to that author's messages
        using the `name` metadata field in ChromaDB. Returns a list of (query, document)
        tuples in rank order.
        """
        # embed the query
        q_emb = self._embedder.encode([query], convert_to_numpy=True)[0].tolist()

        where = {"name": person_name} if person_name else None
        # cast to avoid static typing mismatch with chroma's Where type
        where_cast = cast(Any, where)

        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where_cast,
        )

        # Chroma returns lists-of-lists for batch queries; get first batch safely
        docs = (res.get("documents") or [[]])[0]

        return [(query, d) for d in docs]

