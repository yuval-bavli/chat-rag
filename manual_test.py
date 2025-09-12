from src.config import Configuration
from src.flow import Flow


if __name__ == "__main__":
    config = Configuration.default_config()
    flow = Flow(config)
    flow.read_and_embed_logs()
    flow.ask_question("What's the weather like where Alice is?")
