from typing import Mapping, NamedTuple, Optional, Union

from torch import Tensor

Metadata = Mapping[str, Optional[Union[str, int, float, bool]]]


class EmbedResult(NamedTuple):
    id: str
    document: str
    metadata: Metadata
    embedding: list[float]


def get_ids(embed_results: list[EmbedResult]) -> list[str]:
    return [er.id for er in embed_results]

def get_documents(embed_results: list[EmbedResult]) -> list[str]:
    return [er.document for er in embed_results]

def get_metadatas(embed_results: list[EmbedResult]) -> list[Metadata]:
    return [er.metadata for er in embed_results]    

def get_embeddings(embed_results: list[EmbedResult]) -> list[list[float]]:
    return [er.embedding for er in embed_results]   
