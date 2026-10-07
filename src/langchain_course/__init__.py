import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    print(os.getenv("OLLAMA_API_KEY"))
    print(os.getenv("OLLAMA_BASE_URL"))
    print(os.getenv("OLLAMA_MODEL"))


if __name__ == "__main__":
    main()
