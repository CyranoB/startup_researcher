# Startup Researcher

Startup Researcher is a Python-based tool designed to gather and analyze information about startups using AI-powered web crawling and natural language processing techniques.

## Features

- Web crawling to gather information about startups
- AI-powered analysis of startup data
- Retrieval-Augmented Generation (RAG) for answering questions about startups
- Support for multiple AI models and embedding providers
- Semantic text splitting for improved context understanding
- Vector store integration for efficient information retrieval

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/startup-researcher.git
   cd startup-researcher
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Copy the `.env.sample` file to `.env` and fill in the necessary API keys and credentials.

## Usage

To research a startup, use the following command:
python startup_researcher.py --startup_name "YourStartupName" --llm_provider "groq" --embedding_provider "cohere" --output_file "samples/yourstartupname.md" --copy_to_clipboard

Replace "YourStartupName" with the name of the startup you want to research, and adjust the other parameters as needed.

## Configuration

The project uses various AI models and embedding providers. You can configure these in the `models.py` file. Supported providers include:

- OpenAI
- Anthropic
- Bedrock
- OpenRouter
- Ollama
- Together
- etc.

## Project Structure

- `startup_researcher.py`: Main script for researching startups
- `rag.py`: Retrieval-Augmented Generation module
- `web_crawler.py`: Web crawling functionality
- `models.py`: AI model and embedding provider configurations
- `nlp_rag.py`: Natural Language Processing and RAG utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.