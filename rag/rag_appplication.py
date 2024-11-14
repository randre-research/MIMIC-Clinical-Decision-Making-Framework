from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from os.path import join

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss
import tiktoken

# --- Embedding Model Container ---

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
        self.is_nv_embed = "nv-embed" in self.model_name.lower()
        self.is_stella = "stella" in self.model_name.lower()

    def load_model(self, base_models: str) -> None:
        self.embedding_model = SentenceTransformer(
            join(base_models, self.model_name),            
            trust_remote_code=self.trust_remote_code,
            device=self.device
        )
        if self.is_nv_embed:
            self.embedding_model.max_seq_length = 32768
            self.embedding_model.tokenizer.padding_side = "right"

    def add_eos(self, texts):
        """
        Add EOS token to the end of each text.
        :param texts: A list of strings.
        :return: A list of strings with EOS tokens appended.
        """
        return [text + self.embedding_model.tokenizer.eos_token for text in texts]

    def embed(self, texts):
        """
        Embed a list of texts.
        :param texts: A list of strings to embed.
        :return: A numpy array of embeddings.
        """
        if self.is_nv_embed:
            texts = self.add_eos(texts)
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
            )
        else:
            embeddings = self.embedding_model.encode(texts)
        return embeddings

    def embed_query(self, query, prompt_name='s2p_query'):
        """
        Embed a single query string.
        :param query: A query string to embed.
        :param prompt_name: The prompt name to use for the query.
        :return: An embedding vector.
        """
        if self.is_nv_embed:
            # task_instruction = "Given a question, retrieve passages that answer the question"
            task_instruction = "Given the current patient's information, retrieve relevant medical literature passages"
            query_prefix = f"Instruct: {task_instruction}\nQuery: "
            query_text = query_prefix + query + self.embedding_model.tokenizer.eos_token
            embedding = self.embedding_model.encode(
                [query_text],
                prompt=None,
                normalize_embeddings=True,
            )
        else:
            embedding = self.embedding_model.encode([query], prompt_name=prompt_name)
        return embedding[0]

# --- Vector Store ---

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
            if "chunkr" in path:
                docs.extend(self.load_chunkr_file(path))
            else:
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        return docs

    def load_chunkr_file(self, path):
        """
        Load a chunkr JSON file and create Document objects from it.
        :param path: Path to the chunkr JSON file.
        :return: List of Document objects.
        """
        with open(path, 'r') as file:
            chunk_data = json.load(file)

        # Get document name
        doc_name = path.split("/")[-1]
        
        docs = []
        for chunk in chunk_data:
            for segment in chunk["segments"]:
                content = segment["content"]
                metadata = {
                    "segment_id": segment["segment_id"],
                    "bbox": segment["bbox"],
                    "page": segment["page_number"],
                    "page_width": segment["page_width"],
                    "page_height": segment["page_height"],
                    "segment_type": segment["segment_type"],
                    "image": segment.get("image"),
                    "html": segment.get("html"),
                    "markdown": segment.get("markdown"),
                    "chunk_length": chunk["chunk_length"],
                    "source": doc_name,
                }
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
        return docs

    def split_documents(self, documents):
        """
        Split documents into chunks and add metadata.
        :param documents: List of Document objects.
        :return: List of Document chunks.
        """
        tokenizer = self.embedding_model.embedding_model.tokenizer
        max_length = self.chunk_size
        doc_chunks = []

        for doc in documents:
            text = doc.page_content
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length - self.chunk_overlap)]
            
            for i, token_chunk in enumerate(token_chunks):
                chunk_text = tokenizer.decode(token_chunk)
                chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                
                chunk_id = f"chunk_{len(doc_chunks)}"
                chunk.metadata['chunk_id'] = chunk.metadata.get('segment_id', chunk_id)
                chunk.metadata['token_size'] = len(token_chunk)
                chunk.metadata['document_reference'] = chunk.metadata.get('source', 'unknown')
                chunk.metadata['page_number'] = chunk.metadata.get('page', 'unknown')
                chunk.metadata['order_in_document'] = i
                
                doc_chunks.append(chunk)
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

        for doc, emb in zip(doc_chunks, embeddings):
            doc.metadata['embedding'] = emb

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
        :return: List of dictionaries with chunk content and metadata.
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
        # Extract the content and metadata
        retrieved_info = []
        for doc in retrieved_docs:
            chunk_info = {
                'chunk_id': doc.metadata.get('chunk_id'),
                'document_reference': doc.metadata.get('document_reference'),
                'page_number': doc.metadata.get('page_number'),
                'token_size': doc.metadata.get('token_size'),
                'order_in_document': doc.metadata.get('order_in_document'),
                'content': doc.page_content
            }
            retrieved_info.append(chunk_info)
        return retrieved_info

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
