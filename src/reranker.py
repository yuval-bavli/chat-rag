import numpy as np
from sentence_transformers import CrossEncoder

from model_dirs import get_model_dir


class Reranker:
 
    def __init__(self, rerank_model_name: str) -> None:
         model_file = get_model_dir(rerank_model_name)
         self._reranker_model = self._get_reranker_model(model_file)


    # def _get_reranker_model2(self, model_name: str):
    #         from transformers import AutoModelForSequenceClassification, AutoTokenizer
    #         import torch
    
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         print(f"Loading reranker model {model_name} on {device}")
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #         model.to(device)
    
    #         def predict(pairs: list[tuple[str, str]]) -> list[float]:
    #             texts = [f"[CLS] {q} [SEP] {d} [SEP]" for q, d in pairs]
    #             inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    #             inputs = {k: v.to(device) for k, v in inputs.items()}
    #             with torch.no_grad():
    #                 outputs = model(**inputs)
    #                 scores = outputs.logits[:, 1].cpu().numpy()  # Assuming binary classification and taking the score for the positive class
    #             return scores
    
    #         return predict

    def _get_closest_indexes(self, pairs: list[tuple[str, str]], top_k: int = 1) -> list[int]:

        print("Question to rerank:", pairs[0][0])
        contexts_str = "\n".join([p[1] for p in pairs])
        print(f"Reranking pairs {contexts_str} to get top {top_k} results...")

        scores = self._reranker_model.predict(pairs)

        # convert to array for easy indexing
        scores = np.array(scores)

        # get indices of top 3
        top3_idx_darray = scores.argsort()[::-1][:top_k]  # descending sort
        top3_idx = [i for i in top3_idx_darray]

        # fetch top 3 pairs and scores

        top3_pairs = [pairs[i] for i in top3_idx]
        top3_scores = scores[top3_idx]

        for i, (pair, score) in enumerate(zip(top3_pairs, top3_scores), 1):
            print(f"Rank {i}: Score={score:.4f}, Pair={pair}")
        
        return top3_idx
    

    def _get_reranker_model(self, model_name) -> CrossEncoder:
        model_path = get_model_dir(model_name)
        print(f"Loading reranker model {model_name} from {model_path}...")
        cross_encoder = CrossEncoder(model_path)
        print(f"Reranker model {model_name} loaded")
        return cross_encoder


