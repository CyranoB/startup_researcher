"""
Module for performing retrieval-augmented generation (RAG) using LangChain.

This module provides functions to optimize search queries, retrieve relevant documents,
and generate answers to questions using the retrieved context. It leverages the LangChain
library for building the RAG pipeline.

Functions:
- get_optimized_search_messages(query: str) -> list:
    Generate optimized search messages for a given query.
- optimize_search_query(chat_llm: BaseChatModel, query: str, callbacks: list = []) -> str:
    Optimize the search query using the chat language model.
- get_rag_prompt(question: str, context: str) -> list:
    Get the prompt messages for retrieval-augmented generation (RAG).
- format_docs(docs: list) -> str:
    Format the retrieved documents into an XML string.
- build_rag_prompt(question: str, search_query: str, vectorstore, top_k: int = 10, callbacks: list = []) -> list:
    Build the RAG prompt by retrieving relevant documents and formatting them.
- query_rag(chat_llm: BaseChatModel, question: str, search_query: str, vectorstore, top_k: int = 10, callbacks: list = []) -> str:
    Perform RAG using a single query to retrieve relevant documents and generate an answer.

Note: The multi_query_rag function mentioned in the original docstring is not present in the provided code.
"""

from langchain.schema import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langsmith import traceable
from langchain.prompts import load_prompt
from langchain.chat_models.base import BaseChatModel
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document



def split_docs(contents):
    documents = []
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Split documents
    split_documents = text_splitter.split_documents(documents)

    return split_documents


def split_docs_semantic(contents, embedding_model):
    documents = []
    for content in contents:
        try:
            page_content = content['page_content']
            if page_content:
                metadata = {'title': content['title'], 'source': content['link']}
                doc = Document(page_content=content['page_content'], metadata=metadata)
                documents.append(doc)
        except Exception as e:
            print(f"Error processing content for {content['link']}: {e}")

    # Initialize semantic chunker
    text_splitter = SemanticChunker(embedding_model)

    # Split documents
    split_documents = text_splitter.split_documents(documents)

    return split_documents



@traceable(run_type="embedding")
def vectorize(split_documents, embedding_model):
    
    # Create vector store
    vector_store = None
    batch_size = 250  # Slightly less than 256 to be safe

    for i in range(0, len(split_documents), batch_size):
        batch = split_documents[i:i+batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embedding_model)
        else:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            embeddings = embedding_model.embed_documents(texts)
            vector_store.add_embeddings(
                list(zip(texts, embeddings)),
                metadatas
            )

    return vector_store


def get_rag_prompt(question: str, context: str) -> list:
    system_prompt = SystemMessage(load_prompt("prompts/rag_sys.yaml").format())
    human_prompt = HumanMessage(load_prompt("prompts/rag.yaml").format(question=question, context=context))

    return [system_prompt, human_prompt]

def format_docs(docs: list) -> str:
    formatted_docs = []
    for d in docs:
        content = d.page_content
        title = d.metadata['title']
        source = d.metadata['source']
        doc = f"""
        <document>
            <content>{content}</content>
            <title>{title}</title>
            <link>{source}</link>
        </document>
        """
        formatted_docs.append(doc)
    docs_as_xml = f"<documents>\n{''.join(formatted_docs)}</documents>"
    return docs_as_xml
        

def get_similar_docs(search_query: str, vectorstore, top_k: int = 10, callbacks: list = []) -> list:
    return vectorstore.similarity_search(search_query, k=top_k)

@traceable(run_type="retriever")    
def build_rag_prompt(question: str, search_query: str, vectorstore, top_k: int = 10, callbacks: list = []) -> list:
    unique_docs = get_similar_docs(search_query, vectorstore, top_k=top_k)
    context = format_docs(unique_docs)
    messages = get_rag_prompt(question, context)
    return messages

@traceable(run_type="llm", name="query_rag")
def query_rag(chat_llm: BaseChatModel, question: str, search_query: str, vectorstore, top_k: int = 10, callbacks: list = []) -> str:
    messages = build_rag_prompt(question, search_query, vectorstore, top_k=top_k, callbacks=callbacks)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    
    # Ensure we're returning a string
    if isinstance(response.content, list):
        # If it's a list, join the elements into a single string
        return ' '.join(str(item) for item in response.content)
    elif isinstance(response.content, str):
        # If it's already a string, return it as is
        return response.content
    else:
        # If it's neither a list nor a string, convert it to a string
        return str(response.content)
