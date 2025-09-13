from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from src.stopwatch import Stopwatch
from src.model_dirs import get_model_dir

# Load a local instruct-tuned LLM (quantized GGUF version with llama.cpp is even easier)

class Gpt:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        model_file = get_model_dir(model_name)
        print(f"Loading GPT model {model_name} from {model_file}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_file)

        # Quanization to 4-bit to reduce VRAM usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_file,
            device_map="auto",
            quantization_config=bnb_config,
            # torch_dtype=torch.bfloat16  # torch.float16
        )

    def get_system_prompt(self) -> str:
        # We tell it to end answer with ";;;" so we can parse the answer out of the full text it generates.
        # It tends to hallucinate answers after the first answer, so we want to fish out only the relevant answer.
        return (
            "You are a helpful assistant that helps people find information. "
            "You are given a context and a question, and you answer the question based on the context. "
            "If you don't know the answer, just say you don't know, don't try to make up an answer. "
            "Answer in a concise and clear manner, and finish the answer with three semicolons (;;;)."
        )
    
    def _print_bottleneck(self):
        torch.set_printoptions(profile="short")
        for name, param in self._model.named_parameters():
            if param.device.type != "cuda":
                print("Stuck on CPU:", name, param.device)

    def generate_answer(self, question: str, retrieved_docs: list[str]) -> str:
        system_prompt = self.get_system_prompt()
        context = "\n\n".join(retrieved_docs)
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        self._print_bottleneck()

        print(f"Prompt to GPT:\n{prompt}\n")
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"Generating answer with model {self._model_name}...")
        
        stopwatch = Stopwatch.create_and_start()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            eos_token_id=self._tokenizer.eos_token_id,  # stop at EOS
        )
        stopwatch.stop()
        print(f"Finished generating answer")
        full_answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full answer from GPT:\n{full_answer}\n\nExtracting relavnt answer\n")

        answer = self.find_answer(prompt, full_answer, ";;;")
        return answer

    def find_answer(self, prompt: str, full_answer: str, substring: str) -> str:
        # we want to find the second occurrence of substring after prompt.
        # first one is the example we give in the prompt (e.g. answer ends with ";;;")
        # second one is the end of the actual answer.
        # so we find the second occurrence and return the length of the prompt until the second occurrence (minus the occurance).
        # e.g. prompt = "blah blah end answer with ;;;"; full_answer = "blah blah end answer with ;;; this is the actual answer ;;;"
        #      answer = "blah blah end answer with ;;;| this is the actual answer |;;;"

        first_occurrence = full_answer.find(substring)
        if first_occurrence == -1:
            return full_answer

        second_occurrence = full_answer.find(substring, first_occurrence + len(substring))
        if second_occurrence == -1:
            return full_answer

        answer = full_answer[len(prompt):second_occurrence].strip()
        return answer