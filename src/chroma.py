import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


class Chroma:
        
    def __init__(self, chroma_dir: str, collection_name: str, delete_if_exists: bool = False) -> None:
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