# Standard library imports
import os
import time
import random
from typing import List, Dict

# Third-party library imports
import click  # Command line interface creation
import dotenv  # Environment variable management
import pyperclip  # Clipboard operations
from rich.console import Console  # Enhanced console output
from rich.progress import Progress  # Progress bar functionality
from selenium import webdriver  # Web browser automation
from selenium.webdriver.chrome.options import Options  # Chrome-specific options
from selenium.common.exceptions import WebDriverException  # Selenium exception handling
from webdriver_manager.chrome import ChromeDriverManager  # Chrome driver management
from selenium.webdriver.chrome.service import Service  # Chrome service management
from pinecone import Pinecone, ServerlessSpec  # Vector database operations
from langchain.callbacks import LangChainTracer  # LangChain tracing
from langchain_pinecone import PineconeVectorStore  # Pinecone vector store integration
from langsmith import Client  # LangSmith client for LangChain

# Local module imports
import rag as wr  # Custom RAG (Retrieval-Augmented Generation) module
import web_crawler as wc  # Custom web crawling module
import models as md  # Custom model management module
import nlp_rag as nr  # Custom NLP RAG module

# Additional vector store option (currently unused)
from langchain_community.vectorstores import FAISS

console = Console()
dotenv.load_dotenv()

# Define verbose as a global variable
verbose = False

def get_selenium_driver():
    """
    Set up and return a Selenium WebDriver with Chrome options.
    Includes anti-detection measures and random user agent selection.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import WebDriverException
    import random
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Random user agent to mimic different browsers
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    # Additional stealth options
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        # Overwrite the navigator.webdriver property to avoid detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            """
        })
        return driver
    except WebDriverException as e:
        console.log(f"Error creating Selenium WebDriver: {e}")
        return None

# Set up LangChain callbacks if API key is available
callbacks = []
if os.getenv("LANGCHAIN_API_KEY"):
    callbacks.append(
        LangChainTracer(client=Client())
    )

def add_to_vector_store(contents, vector_store, embedding_model):
    """
    Split documents and add them to the vector store in batches.
    """
    with console.status(f"[bold green]Splitting documents"):
        split_documents = wr.split_docs_semantic(contents, embedding_model)

    with Progress() as progress:
        task = progress.add_task("[bold green]Adding content to vector store", total=len(split_documents))
        batch_size = 250  # Slightly less than 256 to be safe
        for i in range(0, len(split_documents), batch_size):
            batch = split_documents[i:i+batch_size]
            vector_store.add_documents(batch)
            progress.update(task, advance=len(batch))

    return vector_store

def get_info(query: str, max_pages: int = 10, domain: str = None):
    """
    Search for information based on the given query and extract content from web pages.
    """
    with console.status(f"[bold green]Searching info for {query}"):
        sources = wc.get_sources(query, max_pages=max_pages, domain=domain)
        contents = wc.get_links_contents(sources, get_selenium_driver, use_browser=True)
        contents = [content for content in contents if content.get('page_content')]
        if verbose:
            console.log(f"Managed to extract content from {len(contents)} sources for {query}")

    return contents

def extract_info(startup_name: str, vector_store, embedding_model):
    """
    Extract information about a startup using predefined search queries.
    """
    search_queries = [
        "startup",
        "products and services",
        "founders",
        "executives team",
        "investors",
        "competitors",
        "market size",
        "revenue model",
        "growth",
        "funding history"
    ]
    
    contents = []
    for query in search_queries:
        contents += get_info(f"{startup_name} {query}")

    # Add all contents to the vector store
    add_to_vector_store(contents, vector_store, embedding_model)

def write_results_to_markdown(file_path: str, startup_name: str, results: list):
    """
    Write research results to a markdown file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Research Results for {startup_name}\n\n")
        f.write("---\n\n")  # Horizontal line at the start
        
        for result in results:
            f.write(f"## {result['question']}\n\n")
            f.write(f"{result['response']}\n\n")
            f.write("---\n\n")  # Horizontal line after each answer

# Define verbose_global as a global variable
verbose_global = False

@click.command()
@click.argument('startup_name', required=True)
@click.option('-m', '--model_name', default='groq', help='The name of the model to use.')
@click.option('-o', '--output_file', help='The name of the file to write the results to.')
@click.option('-e', '--embedding_model_name', default='openai', help='The name of the embedding model to use.')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Enable verbose output.')
@click.option('-c', '--copy_to_clipboard', is_flag=True, default=False, help='Copy the results to clipboard.')
@click.option('-f', '--force_refresh', is_flag=True, default=False, help='Force refresh of information even if index exists.')
def main(startup_name, model_name, output_file, embedding_model_name, verbose, copy_to_clipboard, force_refresh):
    global verbose_global
    verbose_global = verbose

    """
    Main function to research a startup and generate a report.
    """
    # Set up index name and output file
    index_name = startup_name.lower().replace(' ', '-')
    output_file = f"{index_name}.md" if output_file is None else output_file

    # Initialize language model and embedding model
    llm = md.get_model(model_name)
    embedding_model = md.get_embedding_model(embedding_model_name)   

    # Set up Pinecone vector database
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Modify the logic for creating/using the index
    should_look_info = True
    if index_name not in existing_indexes or force_refresh:
        if index_name in existing_indexes and force_refresh:
            if verbose_global:
                print(f"Force refresh requested. Deleting existing index '{index_name}'.")
            pc.delete_index(index_name)

        with console.status(f"[bold green]Creating index {index_name} in Pinecone"):
            sample_text = "This is a sample text to check embedding dimensions."
            vector = embedding_model.embed_query(sample_text)
            dimensions = len(vector)

            pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
    else:
        should_look_info = False
        if verbose_global:
            print(f"Using existing index '{index_name}'. Use --force_refresh to update information.")

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    # Extract information if needed
    if should_look_info:
        extract_info(startup_name, vector_store, embedding_model)

    # Define queries for startup research
    queries = [
        (f"Tell me about {startup_name}", startup_name),
        (f"Who is {startup_name} founding team", f"{startup_name} founders"),
        (f"What are the main products and/or services of {startup_name}?", f"{startup_name} products"),
        (f"Who is {startup_name} executive team", f"{startup_name} executives"),
        (f"What is {startup_name} funding history", f"{startup_name} funding"),
        (f"Who are {startup_name} investors", f"{startup_name} investors"),
        (f"Who are {startup_name} competitors", f"{startup_name} competitors")
    ]

    # Process queries and generate results
    results = []
    for question in queries:
        search_query = f"{startup_name} {question[1]}"
        response = wr.query_rag(llm, question[0], search_query, vector_store, top_k=20)
        
        print(f"\nQuestion: {question[0]}")
        print(f"Answer: {response}")

        results.append({
            "question": question[0],
            "response": response
        })

    # Write results to file if specified
    if output_file:
        write_results_to_markdown(output_file, startup_name, results)

    # Copy results to clipboard if specified
    if copy_to_clipboard:
        with open(output_file, 'r', encoding='utf-8') as f:
            pyperclip.copy(f.read())

if __name__ == "__main__":
    main()