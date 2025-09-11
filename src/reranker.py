from sentence_transformers import CrossEncoder

from models.model_dirs import get_model_dir


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

    def _get_closest_indexes(self, pairs: list[tuple[str, str]]):
        return self._reranker_model.predict(pairs)

    def _get_reranker_model(self, model_name) -> CrossEncoder:
        model_path = get_model_dir(model_name)
        print(f"Loading reranker model {model_name} from {model_path}...")
        cross_encoder = CrossEncoder(model_path)
        print(f"Reranker model {model_name} loaded")
        return cross_encoder


