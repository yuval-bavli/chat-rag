
from typing import Any, Mapping, Optional, Union, cast
import chromadb
from chromadb.api.types import Embedding, PyEmbeddings, PyEmbedding
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from torch import Tensor

from src.embed_result import EmbedResult, get_documents, get_embeddings, get_ids, get_metadatas
from src.metadata import Metadata


class Chroma:
        
    def __init__(
            self,
            chroma_dir: str,
            collection_name: str,
            delete_if_exists: bool = False
        ) -> None:
        self._client = self.create_client(chroma_dir)
        self.collection = self.get_collection(self._client, collection_name, delete_if_exists)
        
    def create_client(self, chrome_dir) -> ClientAPI:
        client = chromadb.PersistentClient(
            path=chrome_dir
        )
        print(f"Created ChromaDB client in {chrome_dir}")
        return client


    def delete_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.delete_collection(name=collection_name)
        print(f"Deleted collection {collection_name}")


    def add_documents(self, embedded_results: list[EmbedResult]) -> None:
        ids = get_ids(embedded_results)
        documents = get_documents(embedded_results)
        metadatas_list = get_metadatas(embedded_results)
        embeddings = [cast(PyEmbedding, e) for e in get_embeddings(embedded_results)]

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas_list, # type: ignore
            embeddings=embeddings
        )


    def find_similar(self, embeddings: list[list[float]], where_clause: Optional[dict] = None, n_results: int = 5) -> list[EmbedResult]:
        """
        Find top-N semantically similar messages to `query`.

        If `person_name` is provided, results will be filtered to that author's messages
        using the `name` metadata field in ChromaDB. Returns a list of (query, document)
        tuples in rank order.
        """
        # embed the query
        # q_emb = self._embedder.encode([query], convert_to_numpy=True)[0].tolist()

        embeddings_list = [cast(Embedding, e) for e in embeddings]

        query_result = self.collection.query(
            query_embeddings=embeddings_list,
            n_results=n_results,
            where=where_clause,
        )

        # Chroma returns lists-of-lists for batch queries; get first batch safely
        documents = (query_result.get("documents") or [[]])[0]
        ids = (query_result.get("ids") or [[]])[0]
        metadatas_list = (query_result.get("metadatas") or [[]])[0]

        results: list[EmbedResult] = []

        for i, doc in enumerate(documents):
            id: str = ids[i] if i < len(ids) else ""
            raw_meta = metadatas_list[i] if i < len(metadatas_list) else {}
            if isinstance(raw_meta, Metadata):
                metadata: Metadata = raw_meta
            else:
                # raw_meta is expected to be a dict from Chroma; convert to Metadata
                metadata = Metadata.from_dict(raw_meta if isinstance(raw_meta, dict) else {})
            embedding = []  # placeholder, not returned by Chroma
            result = EmbedResult(id, doc, metadata, embedding)
            results.append(result)

        return results


    def get_collection(self, client: ClientAPI, collection_name: str, delete_if_exists: bool) -> Collection:

        create_new = False
        if collection_name in [c.name for c in client.list_collections()]:
            if delete_if_exists:
                self.delete_collection(client, collection_name)
                create_new = True
            else:
                collection = client.get_collection(collection_name)
                print(f"Retrieved existing collection {collection_name}")
        else:
            create_new = True

        if create_new:
            collection = client.create_collection(name=collection_name)
            print(f"Created new collection {collection_name}")

        return collection