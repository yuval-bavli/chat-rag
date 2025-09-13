import os

from src.chroma import Chroma
from src.config import Configuration
from src.embedder import Embedder
from src.data_reader import DataReader


def test_embedder_find_similar_flow() -> None:
    _embedder_find_similar_flow(with_name=False)


def _embedder_find_similar_flow(with_name: bool) -> None:
    # Arrange:

    question = "What's the weather like for Alice?"
    n_results = 3
    name = "Alice" if with_name else None

    # Read test data
    base = os.path.dirname(__file__)
    json_path = os.path.join(base, "test_files", "comments.json")
    reader = DataReader(json_path)
    user_logs = reader.read_logs()

    # Set up chroma
    test_chroma_dir = "test_chroma_db"
    test_collection_name = "test_collection1"
    default_config = Configuration.default_config()
    chroma = Chroma(test_chroma_dir, test_collection_name, delete_if_exists=True)
    
    # Set up embedder (class under test)
    embedder = Embedder(default_config.embedder_model_name)

    # Act (embed and query):
    message_embeddings = embedder.embed_messages(user_logs)
    chroma.add_documents(message_embeddings)

    embedded_question = embedder.embed_question(question)

    results = chroma.find_similar([embedded_question], name, n_results)

    # Assert:
    assert len(results) == 3

    docs = [r.document for r in results]
    target_message = "Hi everyone! It's raining cats and dogs here."
    # validate one of returned documents contains the target message text
    assert any(target_message in doc for doc in docs), "Target message was not found in results"
