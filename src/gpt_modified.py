import os

from src.config import Configuration
from src.gpt import Gpt


class GPTModidied(Gpt):
    def get_system_prompt(self) -> str:
        return (
            "You are a helpful assistant that helps people find information they seek. "
        )
    


if __name__ == "__main__":
    config = Configuration.default_config()
    gpt = GPTModidied(config.gpt_model_name)
    # read all text from file:
    with open(R"C:\Users\yuval\temp\keywords.txt", 'r') as file:
        keywords = file.read()
        answer = gpt.generate_answer(
            "Given the keywords, continue the list with 100 keywords with the same context, new keywords should not contain previous keywords or their substring, and finish the answer with three semicolons (;;;). "
            "Keywords: " + keywords, ["keywords"]
            )

    print(f"Answer: {answer}")