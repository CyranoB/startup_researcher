import spacy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langsmith import Client, traceable
import concurrent.futures

# Function to load or download the spaCy model
def get_nlp_model(model="en_core_web_md"):
    if not spacy.util.is_package(model):
        print(f"Downloading {model} model...")
        spacy.cli.download(model)
        print("Model downloaded successfully!")
    nlp = spacy.load(model)
    return nlp

# Initialize the spaCy model globally to share across threads
nlp = get_nlp_model("en_core_web_md")

@lru_cache(maxsize=10000)
def get_sentence_vector(sentence_text: str) -> tuple:
    """
    Retrieve the vector for a given sentence, using caching to speed up repeated accesses.

    :param sentence_text: The sentence text.
    :return: The vector representation as a tuple.
    """
    doc = nlp(sentence_text)
    return tuple(doc.vector)

def semantic_splitting_batch(
        documents: List[str],
        max_chunk_size: int = 100,
        similarity_threshold: float = 0.5
    ) -> List[List[str]]:
    """
    Splits multiple documents into semantically coherent chunks using batch processing.

    :param documents: List of documents to split.
    :param max_chunk_size: Maximum number of sentences per chunk.
    :param similarity_threshold: Threshold below which a new chunk starts.
    :return: A list where each element is a list of chunks for a document.
    """
    # Process all documents in a single spaCy pipeline run
    docs = list(nlp.pipe(documents))
    all_doc_chunks = []

    for doc in docs:
        sentences = list(doc.sents)

        if not sentences:
            all_doc_chunks.append([])
            continue

        # Generate embeddings for all sentences using cached vectors
        sentence_embeddings = np.array([get_sentence_vector(sent.text) for sent in sentences])

        chunks = []
        current_chunk = [sentences[0].text]
        current_chunk_size = 1

        for i in range(1, len(sentences)):
            # Compute cosine similarity between consecutive sentence embeddings
            similarity = cosine_similarity(
                sentence_embeddings[i - 1].reshape(1, -1),
                sentence_embeddings[i].reshape(1, -1)
            )[0][0]  # Properly access the similarity score

            if similarity < similarity_threshold or current_chunk_size >= max_chunk_size:
                # Push the current chunk to chunks
                chunks.append(' '.join(current_chunk))
                # Start a new chunk
                current_chunk = [sentences[i].text]
                current_chunk_size = 1
            else:
                current_chunk.append(sentences[i].text)
                current_chunk_size += 1

        # Push the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        all_doc_chunks.append(chunks)

    return all_doc_chunks

def process_batch(batch: List[Dict]) -> List[Dict]:
    """
    Process a single batch of documents to split them into chunks.

    :param batch (List[Dict]): A list of documents.
    :return (List[Dict]): A list of processed chunks.
    """
    try:
        documents = [content.get('page_content', '') for content in batch]
        titles = [content.get('title', '') for content in batch]
        sources = [content.get('link', '') for content in batch]
        
        # Split documents into chunks
        split_results = semantic_splitting_batch(documents)
        
        batch_chunks = []
        for doc_chunks, title, source in zip(split_results, titles, sources):
            for chunk in doc_chunks:
                batch_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'title': title,
                        'source': source
                    }
                })
        return batch_chunks
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def semantic_split_documents(contents: List[Dict], batch_size: int = 10) -> List[Dict]:
    """
    Semantically split an array of documents into coherent chunks using batch processing.

    :param contents: List of dictionaries containing document information
    :param batch_size: Number of documents to process in each batch
    :return: List of semantically split chunks
    """
    all_chunks = []
    
    with ThreadPoolExecutor() as executor:
        # Split contents into batches
        batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size)]
        
        # Submit all batches to the executor
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        for future in concurrent.futures.as_completed(futures):
            all_chunks.extend(future.result())
    
    return all_chunks

def semantic_search(query, chunks, nlp, top_n=10, similarity_threshold=0.5):
    """
    Perform semantic search to find the most relevant text chunks related to the query.

    Args:
        query (str): The search query provided by the user.
        chunks (list of dict): A list of text chunks where each chunk is a dictionary
                              containing at least a 'text' key.
        nlp: The spaCy language model.
        top_n (int, optional): The maximum number of top relevant chunks to return. Defaults to 5.
        similarity_threshold (float, optional): The minimum similarity score a chunk must
                                                have to be considered relevant. Defaults to 0.5.

    Returns:
        list of tuple: A list of tuples where each tuple contains a relevant chunk and its
                       corresponding similarity score, sorted in descending order of similarity.
    """
    import numpy as np

    # ----------------------------
    # Step 1: Precompute Query Vector
    # ----------------------------
    # Disable all pipeline components except 'tok2vec' to speed up processing and
    # focus only on generating the vector representation.
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'tok2vec']):
        # Process the query to obtain its vector representation.
        query_doc = nlp(query)
        query_vector = query_doc.vector

    # Compute the norm (magnitude) of the query vector and add a small epsilon to
    # prevent division by zero in similarity calculations.
    query_norm = np.linalg.norm(query_vector) + 1e-8

    # ----------------------------
    # Step 2: Extract Texts from Chunks
    # ----------------------------
    # Create a list of texts from the provided chunks for processing.
    texts = [chunk['text'] for chunk in chunks]

    # ----------------------------
    # Step 3: Define Vector Computation Function
    # ----------------------------
    def compute_vector(text: str) -> np.ndarray:
        """
        Compute the vector representation of a given text using spaCy.

        Args:
            text (str): The text to process.

        Returns:
            numpy.ndarray: The vector representation of the text.
        """
        # Disable all pipeline components except 'tok2vec' to optimize performance.
        with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'tok2vec']):
            # Process the text to obtain its vector.
            doc = nlp(text)
            return doc.vector

    # ----------------------------
    # Step 4: Compute Vectors for All Chunks in Parallel
    # ----------------------------
    with ThreadPoolExecutor() as executor:
        chunk_vectors = list(executor.map(compute_vector, texts))

    # Check if chunk_vectors is empty to prevent AxisError
    if not chunk_vectors:
        print("No chunks available for semantic search.")
        return []

    # Convert the list of vectors to a NumPy array for efficient numerical operations.
    chunk_vectors = np.array(chunk_vectors)

    # Verify the dimensionality of chunk_vectors
    if chunk_vectors.ndim != 2:
        raise ValueError(f"Expected chunk_vectors to be 2D, but got {chunk_vectors.ndim}D array.")

    # Compute the norm for each chunk vector and add a small epsilon to avoid division by zero.
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1) + 1e-8

    # ----------------------------
    # Step 5: Calculate Cosine Similarities
    # ----------------------------
    # Compute the cosine similarity between the query vector and each chunk vector.
    # This is done by taking the dot product of the chunk vectors with the query vector
    # and then dividing by the product of their norms.
    similarities = np.dot(chunk_vectors, query_vector) / (chunk_norms * query_norm)

    # ----------------------------
    # Step 6: Filter and Sort Relevant Chunks
    # ----------------------------
    # Pair each chunk with its similarity score and filter out those below the threshold.
    results = [
        (chunk, sim) for chunk, sim in zip(chunks, similarities) if sim > similarity_threshold
    ]

    # Sort the relevant chunks in descending order based on their similarity scores.
    results.sort(key=lambda x: x[1], reverse=True)

    # ----------------------------
    # Step 7: Return Top N Relevant Chunks
    # ----------------------------
    # Return only the top_n chunks that are most relevant to the query.
    return results[:top_n]

# Function to perform RAG (Retrieval-Augmented Generation) query
def query_rag(chat_llm, query, relevant_results):
    import rag as wr

    # Format the relevant chunks into XML-like structure
    formatted_chunks = ""
    for chunk, similarity in relevant_results:
        formatted_chunk = f"""
        <source>
            <url>{chunk['metadata']['source']}</url>
            <title>{chunk['metadata']['title']}</title>
            <text>{chunk['text']}</text>
        </source>
        """
        formatted_chunks += formatted_chunk

    # Get the RAG prompt template and format it with the query and context
    messages = wr.get_rag_prompt(query, formatted_chunks)  

    # Generate a response using the chat LLM
    draft = chat_llm.invoke(messages).content
    return draft

# Function to perform semantic splitting of documents
def recursive_split_documents(contents, max_chunk_size=1000, overlap=100):
    from langchain_core.documents.base import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    documents = []
    # Convert input contents to Document objects
    for content in contents:
        try:
            page_content = content['page_content']
            if page_content:
                metadata = {'title': content['title'], 'source': content['link']}
                doc = Document(page_content=content['page_content'], metadata=metadata)
                documents.append(doc)
        except Exception as e:
            print(f"Error processing content for {content['link']}: {e}")

    # Initialize recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=overlap)

    # Split documents
    split_documents = text_splitter.split_documents(documents)

    # Convert split documents to the desired format
    chunks = []
    for doc in split_documents:
        chunk = {
            'text': doc.page_content,
            'metadata': {
                'title': doc.metadata.get('title', ''),
                'source': doc.metadata.get('source', '')
            }
        }
        chunks.append(chunk)

    return chunks

def main():
    import cProfile, pstats, io
    import multiprocessing

    # Profile the main function
    pr = cProfile.Profile()
    pr.enable()

    # Generate a list of content for testing
    test_contents = [
        {
            'page_content': "Artificial Intelligence (AI) is revolutionizing various industries. " 
                            "Machine Learning, a subset of AI, enables systems to learn from data. "
                            "Deep Learning, inspired by the human brain, has shown remarkable results in image and speech recognition. "
                            "Natural Language Processing allows machines to understand and generate human language. "
                            "These technologies are driving innovation in fields like healthcare, finance, and autonomous vehicles.",
            'title': 'Introduction to AI and Its Subfields',
            'link': 'https://example.com/ai-introduction'
        },
        {
            'page_content': "Climate change is a pressing global issue that affects every corner of our planet. "
                            "Rising temperatures are causing sea levels to rise and weather patterns to become more extreme. "
                            "Greenhouse gas emissions from human activities, particularly the burning of fossil fuels, are the primary driver of climate change. "
                            "The impacts of climate change are far-reaching and include more frequent and intense heatwaves, droughts, and storms. "
                            "Melting glaciers and ice sheets contribute to sea level rise, threatening coastal communities and ecosystems. "
                            "Changes in precipitation patterns affect agriculture and water resources, potentially leading to food and water insecurity in many regions. "
                            "The ocean is also affected, with rising temperatures and increased acidity impacting marine life and coral reefs. "
                            "Efforts to mitigate climate change include transitioning to renewable energy sources, improving energy efficiency, and reducing deforestation. "
                            "Carbon pricing and emissions trading schemes are economic tools being implemented to reduce greenhouse gas emissions. "
                            "Adaptation strategies are also necessary to deal with the impacts that are already occurring and those that are inevitable. "
                            "These may include building sea walls, developing drought-resistant crops, and improving early warning systems for extreme weather events. "
                            "International cooperation, such as the Paris Agreement, aims to limit global temperature rise and support vulnerable nations. "
                            "Individual actions, like reducing energy consumption and choosing sustainable transportation, also play a crucial role in combating climate change. "
                            "The urgency of addressing climate change cannot be overstated, as the window for effective action is rapidly closing.",
            'title': 'Understanding Climate Change',
            'link': 'https://example.com/climate-change-overview'
        },
        {
            'page_content': "The human genome project was a landmark scientific achievement. "
                            "It involved mapping all human genes, which are instructions for building and maintaining cells. "
                            "This project has revolutionized our understanding of genetics and human biology. "
                            "It has paved the way for personalized medicine and new treatments for genetic disorders. "
                            "Ethical considerations surrounding genetic information continue to be an important topic of discussion.",
            'title': 'The Human Genome Project and Its Impact',
            'link': 'https://example.com/human-genome-project'
        }
    ]

    print("Test contents generated for semantic splitting:")
    for content in test_contents:
        print(f"\nTitle: {content['title']}")
        print(f"Source: {content['link']}")
        print(f"Content preview: {content['page_content'][:100]}...")

    # Semantic split the documents using ThreadPoolExecutor
    chunks = semantic_split_documents(test_contents, batch_size=20)

    print("\nDocument Chunks:")
    for idx, chunk in enumerate(chunks, 1):
        print(f"\nChunk {idx}:\n{chunk}")

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)  # Print top 10 functions
    print(s.getvalue())

if __name__ == "__main__":
    main()