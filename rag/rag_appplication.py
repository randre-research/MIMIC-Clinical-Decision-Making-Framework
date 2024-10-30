# --- Embedding Model Container ---
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from os.path import join

class EmbeddingModelContainer:
    def __init__(
        self, model_name_or_path='dunzhang/stella_en_400M_v5', device='cpu', trust_remote_code=True
    ):
        """
        Initialize the embedding model container with a SentenceTransformer model.
        :param model_name_or_path: The name or path of the embedding model.
        :param device: The device to run the model on ('cpu' or 'cuda').
        :param trust_remote_code: Whether to trust remote code for custom models.
        """
        self.model_name = model_name_or_path
        self.device = device
        self.trust_remote_code = trust_remote_code

    def load_model(self, base_models: str) -> None:
        self.embedding_model = SentenceTransformer(
            join(base_models, self.model_name),            
            trust_remote_code=self.trust_remote_code,
            device=self.device
        )

    def embed(self, texts):
        """
        Embed a list of texts.
        :param texts: A list of strings to embed.
        :return: A numpy array of embeddings.
        """
        embeddings = self.embedding_model.encode(texts)
        return embeddings

    def embed_query(self, query, prompt_name='s2p_query'):
        """
        Embed a single query string.
        :param query: A query string to embed.
        :param prompt_name: The prompt name to use for the query.
        :return: An embedding vector.
        """
        embedding = self.embedding_model.encode([query], prompt_name=prompt_name)
        return embedding[0]

# --- Vector Store ---

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

class VectorStore:
    def __init__(
        self, document_paths, embedding_model_container, chunk_size=250, chunk_overlap=0
    ):
        """
        Initialize the vector store with documents and embeddings.
        :param document_paths: List of document file paths.
        :param embedding_model_container: An instance of EmbeddingModelContainer.
        :param chunk_size: Size of text chunks for splitting documents.
        :param chunk_overlap: Overlap size between chunks.
        """
        self.embedding_model = embedding_model_container
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Load documents from paths
        self.docs = self.load_documents(document_paths)
        # Split documents into chunks
        self.doc_chunks = self.split_documents(self.docs)
        # Create vector store
        self.index, self.doc_chunks = self.create_vector_store(self.doc_chunks)

    def load_documents(self, document_paths):
        """
        Load documents from the provided file paths.
        :param document_paths: List of document file paths.
        :return: List of Document objects.
        """
        docs = []
        for path in document_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        return docs

    def split_documents(self, documents):
        """
        Split documents into chunks.
        :param documents: List of Document objects.
        :return: List of Document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_chunks = text_splitter.split_documents(documents)
        return doc_chunks

    def create_vector_store(self, doc_chunks):
        """
        Create the vector store from document chunks and embeddings.
        :param doc_chunks: List of Document chunks.
        :return: A FAISS index and corresponding documents.
        """
        texts = [doc.page_content for doc in doc_chunks]
        embeddings = self.embedding_model.embed(texts)
        embeddings = np.array(embeddings).astype('float32')

        # Add embeddings to documents
        for doc, emb in zip(doc_chunks, embeddings):
            doc.metadata['embedding'] = emb

        # Create a FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, doc_chunks

    def get_vector_store(self):
        """
        Get the underlying vector store.
        :return: The FAISS index and documents.
        """
        return self.index, self.doc_chunks

# --- Retriever ---

class Retriever:
    def __init__(
        self, vector_store, embedding_model_container, top_k=4, re_rank=False, prompt_name='s2p_query'
    ):
        """
        Initialize the retriever.
        :param vector_store: The vector store instance.
        :param embedding_model_container: An instance of EmbeddingModelContainer.
        :param top_k: Number of top documents to retrieve.
        :param re_rank: Whether to re-rank the retrieved documents.
        :param prompt_name: The prompt name to use for query embedding.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model_container
        self.top_k = top_k
        self.re_rank = re_rank
        self.prompt_name = prompt_name

    def retrieve(self, query):
        """
        Retrieve relevant documents for a query.
        :param query: The query string.
        :return: List of relevant Document objects.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query, prompt_name=self.prompt_name)
        query_embedding = np.array([query_embedding]).astype('float32')

        # Retrieve top_k similar documents
        D, I = self.vector_store.index.search(query_embedding, self.top_k)
        # Get the corresponding documents
        retrieved_docs = [self.vector_store.doc_chunks[i] for i in I[0]]
        # Optionally, re-rank the documents
        if self.re_rank:
            retrieved_docs = self.re_rank_documents(query_embedding[0], retrieved_docs)
        return retrieved_docs

    def re_rank_documents(self, query_embedding, documents):
        """
        Re-rank the documents based on similarity scores.
        :param query_embedding: The embedding vector of the query.
        :param documents: List of Document objects.
        :return: Re-ranked list of Document objects.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        doc_embeddings = [doc.metadata['embedding'] for doc in documents]
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        # Pair documents with their similarity scores
        doc_similarity_pairs = list(zip(documents, similarities))
        # Sort documents by similarity score in descending order
        doc_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        # Return the sorted documents
        sorted_docs = [pair[0] for pair in doc_similarity_pairs]
        return sorted_docs


### Heres how you can use it

# # Initialize the embedding model container
# embedding_model_container = EmbeddingModelContainer(
#     model_name_or_path="dunzhang/stella_en_400M_v5",
#     device="cuda"  # or "cpu"
# )

# # Paths to your documents (PDF files)
# document_paths = ["path/to/document1.pdf", "path/to/document2.pdf"]

# # Initialize the vector store
# vector_store = VectorStore(
#     document_paths=document_paths,
#     embedding_model_container=embedding_model_container,
#     chunk_size=250,
#     chunk_overlap=0
# )

# # Initialize the retriever
# retriever = Retriever(
#     vector_store=vector_store,
#     embedding_model_container=embedding_model_container,
#     top_k=4,
#     re_rank=True,
#     prompt_name="s2p_query"  # or "s2s_query" depending on your task
# )

# # Retrieve documents for a query
# query = "What are the health benefits of green tea?"
# retrieved_docs = retriever.retrieve(query)

# # Print retrieved documents
# for doc in retrieved_docs:
#     print(doc.page_content)
