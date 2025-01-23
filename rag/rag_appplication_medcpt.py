import os
import json
from os.path import join

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
import faiss
import tiktoken

from sklearn.metrics.pairwise import cosine_similarity

# --- Embedding Model Container ---
class EmbeddingModelContainer:
    def __init__(
        self, 
        model_name_or_path='dunzhang/stella_en_400M_v5', 
        device='cpu', 
        trust_remote_code=True
    ):
        """
        Initialize the embedding model container. If the model is "MedCPT-Cross-Encoder",
        we'll load it as a cross-encoder; otherwise, we'll load a SentenceTransformer.
        """
        self.model_name = model_name_or_path
        self.device = device
        self.trust_remote_code = trust_remote_code

        # For special-cased behavior:
        self.is_nv_embed = "nv-embed" in self.model_name.lower()
        self.is_stella = "stella" in self.model_name.lower()
        self.is_medcpt = "medcpt" in self.model_name.lower()

    def load_model(self, base_models: str) -> None:
        """
        Load either a SentenceTransformer or a Cross-Encoder model, depending on self.is_medcpt.
        """
        if self.is_medcpt:
            # --- Load Cross-Encoder ---
            self.tokenizer = AutoTokenizer.from_pretrained(
                join(base_models, self.model_name),
                trust_remote_code=self.trust_remote_code,
                device=self.device
            )
            # self.tokenizer.to(self.device)
            self.cross_model = AutoModelForSequenceClassification.from_pretrained(
                join(base_models, self.model_name),
                trust_remote_code=self.trust_remote_code
            ).to(self.device)
            # self.cross_model.to(self.device)
        else:
            # --- Load SentenceTransformer ---
            self.embedding_model = SentenceTransformer(
                join(base_models, self.model_name),
                trust_remote_code=self.trust_remote_code,
                device=self.device
            )
            if self.is_nv_embed:
                # Increase max sequence length if using "nv-embed"
                self.embedding_model.max_seq_length = 32768
                self.embedding_model.tokenizer.padding_side = "right"

    def add_eos(self, texts):
        """
        Add EOS token to the end of each text (used only for some "nv-embed" models).
        """
        return [text + self.embedding_model.tokenizer.eos_token for text in texts]

    def embed(self, texts):
        """
        Embed a list of texts. If we're using a Cross-Encoder (MedCPT), 
        we typically do *not* embed single texts. We'll raise an exception 
        or skip unless you implement a custom approach.
        """
        if self.is_medcpt:
            raise NotImplementedError(
                "MedCPT-Cross-Encoder is not used for embedding single texts. "
                "Use `rank_texts(query, texts)` instead."
            )

        # Standard embedding with SentenceTransformer
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
        Embed a single query string. This also doesn't apply to the Cross-Encoder approach. 
        We'll raise an exception if the user attempts it.
        """
        if self.is_medcpt:
            raise NotImplementedError(
                "MedCPT-Cross-Encoder doesn't produce a single embedding for a query. "
                "Use `rank_texts(query, texts)` instead."
            )

        # Normal embedding-model flow:
        if self.is_nv_embed:
            # For 'nv-embed' style
            task_instruction = (
                "Given the current patient's information, "
                "retrieve relevant medical literature passages"
            )
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

    def rank_texts(self, query, texts, batch_size=1024):
        """
        MedCPT Cross-Encoder approach for scoring (ranking) a list of texts 
        against a given query. Returns logits for each [query, text] pair.

        :param query: The query string.
        :param texts: A list of text strings to be scored.
        :param batch_size: Number of pairs to process at a time.
        :return: A numpy array of logits (scores) for each pair.
        """
        if not self.is_medcpt:
            raise ValueError(
                "rank_texts is only available when using MedCPT-Cross-Encoder."
            )

        # We'll accumulate logits from each batch here
        all_logits = []

        # Process in mini-batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Construct [query, text] pairs for this batch
            pairs = [[query, t] for t in batch_texts]

            with torch.no_grad():
                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=64,
                    # max_length=1024,
                ).to(self.device)

                batch_logits = self.cross_model(**encoded).logits.squeeze(dim=1)
                # Move logits to CPU and convert to numpy
                batch_logits = batch_logits.cpu().numpy()

            all_logits.extend(batch_logits)

        # Convert the collected logits to a NumPy array
        return np.array(all_logits)

# --- Vector Store ---

class VectorStore:
    def __init__(
        self,
        document_paths,
        embedding_model_container,
        chunk_size=250,
        chunk_overlap=0,
        smart_chunking=False
    ):
        """
        Initialize the vector store with documents and embeddings.
        :param document_paths: List of document file paths.
        :param embedding_model_container: An instance of EmbeddingModelContainer.
        :param chunk_size: Size of text chunks for splitting documents.
        :param chunk_overlap: Overlap size between chunks.
        :param smart_chunking: When chunking markdown, split by headers and prepend headings.
        """
        self.embedding_model = embedding_model_container
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.smart_chunking = smart_chunking

        # 1) Load documents from paths
        self.docs = self.load_documents(document_paths)

        # 2) Split documents into chunks
        self.doc_chunks = self.split_documents(self.docs)

        # 3) Create the vector store (FAISS) if not MedCPT-Cross-Encoder
        self.index, self.doc_chunks = self.create_vector_store(self.doc_chunks)

    def load_documents(self, document_paths):
        """
        Load documents from the provided file paths.
        :param document_paths: List of document file paths.
        :return: List of Document objects.
        """
        docs = []
        for path in document_paths:
            if "chunkr" in path.lower():
                # Handle JSON chunkr files
                docs.extend(self.load_chunkr_file(path))
            elif path.lower().endswith('.pdf'):
                # Handle PDF documents
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif path.lower().endswith('.md'):
                # Handle Markdown documents
                docs.extend(self.load_markdown_file(path))
            else:
                raise ValueError(f"Unsupported file format for: {path}")
        return docs

    def load_markdown_file(self, path):
        """
        Load a Markdown file and create Document objects from it.
        :param path: Path to the Markdown file.
        :return: List of Document objects.
        """
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        doc_name = os.path.basename(path)
        doc = Document(
            page_content=content,
            metadata={
                "source": doc_name,
                "format": "markdown"
            }
        )
        return [doc]

    def load_chunkr_file(self, path):
        """
        Load a chunkr JSON file and create Document objects from it.
        :param path: Path to the chunkr JSON file.
        :return: List of Document objects.
        """
        with open(path, 'r', encoding='utf-8') as file:
            chunk_data = json.load(file)

        doc_name = os.path.basename(path)

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
        # 1) Decide which tokenizer to use.
        #    For cross-encoder (MedCPT), we store the tokenizer in self.embedding_model.tokenizer
        #    For normal embedding models, the tokenizer is in self.embedding_model.embedding_model.tokenizer
        if self.embedding_model.is_medcpt:
            tokenizer = self.embedding_model.tokenizer
        else:
            tokenizer = self.embedding_model.embedding_model.tokenizer

        max_length = self.chunk_size
        doc_chunks = []

        for doc in documents:
            if self.smart_chunking and doc.metadata.get('format') == 'markdown':
                # --- Markdown header splitter ---
                headers_to_split_on = [
                    ("#", "header_1"),
                    ("##", "header_2"),
                    ("###", "header_3"),
                    ("####", "header_4"),
                    ("#####", "header_5"),
                    ("######", "header_6"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
                chunks = markdown_splitter.split_text(doc.page_content)
                # --- Markdown header splitter end ---

                # --- Constrain chunk size with recursive splitter ---
                chunk_size = self.chunk_size
                chunk_overlap = self.chunk_overlap
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_documents(chunks)
                # --- Constrain chunk size end ---

                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.page_content
                    chunk_metadata = chunk.metadata

                    # Reinsert headings at the start of each chunk
                    chunk_title = ""
                    for key in chunk_metadata:
                        if "header" in key:
                            # 'header_1' => #, 'header_2' => ##, etc.
                            level = int(key.split("_")[-1])
                            chunk_title += "#" * level + " " + chunk_metadata[key] + "\n"

                    chunk_text = chunk_title + chunk_text
                    tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

                    new_chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                    chunk_id = f"chunk_{len(doc_chunks)}"
                    new_chunk.metadata['chunk_id'] = new_chunk.metadata.get('segment_id', chunk_id)
                    new_chunk.metadata['token_size'] = len(tokens)
                    new_chunk.metadata['document_reference'] = new_chunk.metadata.get('source', 'unknown')
                    new_chunk.metadata['page_number'] = new_chunk.metadata.get('page', 'unknown')
                    new_chunk.metadata['order_in_document'] = i
                    # Copy header info
                    new_chunk.metadata['header_1'] = chunk_metadata.get('header_1', 'unknown')
                    new_chunk.metadata['header_2'] = chunk_metadata.get('header_2', 'unknown')
                    new_chunk.metadata['header_3'] = chunk_metadata.get('header_3', 'unknown')
                    new_chunk.metadata['header_4'] = chunk_metadata.get('header_4', 'unknown')
                    new_chunk.metadata['header_5'] = chunk_metadata.get('header_5', 'unknown')
                    new_chunk.metadata['header_6'] = chunk_metadata.get('header_6', 'unknown')

                    doc_chunks.append(new_chunk)
            else:
                # Simple chunking (no markdown headings)
                text = doc.page_content
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Create overlapping token chunks
                token_chunks = [
                    tokens[i:i + max_length]
                    for i in range(0, len(tokens), max_length - self.chunk_overlap)
                ]

                for i, token_chunk in enumerate(token_chunks):
                    chunk_text = tokenizer.decode(token_chunk)
                    new_chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())

                    chunk_id = f"chunk_{len(doc_chunks)}"
                    new_chunk.metadata['chunk_id'] = new_chunk.metadata.get('segment_id', chunk_id)
                    new_chunk.metadata['token_size'] = len(token_chunk)
                    new_chunk.metadata['document_reference'] = new_chunk.metadata.get('source', 'unknown')
                    new_chunk.metadata['page_number'] = new_chunk.metadata.get('page', 'unknown')
                    new_chunk.metadata['order_in_document'] = i

                    doc_chunks.append(new_chunk)

        return doc_chunks

    def create_vector_store(self, doc_chunks):
        """
        Create the vector store from document chunks and embeddings.
        :param doc_chunks: List of Document chunks.
        :return: A (FAISS index or None) and the (possibly updated) list of doc_chunks.
        """
        # If we're dealing with MedCPT (cross-encoder), we skip embedding and return None
        if self.embedding_model.is_medcpt:
            # We don’t store doc embeddings for cross-encoders
            return None, doc_chunks

        # Otherwise, embed the chunks
        texts = [doc.page_content for doc in doc_chunks]
        embeddings = self.embedding_model.embed(texts)  # shape = [num_docs, embedding_dim]
        embeddings = np.array(embeddings).astype('float32')

        # Store each chunk embedding in metadata
        for doc, emb in zip(doc_chunks, embeddings):
            doc.metadata['embedding'] = emb

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, doc_chunks

    def get_vector_store(self):
        """
        Get the underlying vector store (FAISS index) and doc_chunks.
        :return: (faiss.Index or None, [Document chunks]).
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
        :param re_rank: Whether to re-rank the retrieved documents (only applies for standard embeddings).
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
        # --------------------------------------------------
        # Case 1: If we're using a SentenceTransformer or other normal embed model
        # --------------------------------------------------
        if not self.embedding_model.is_medcpt:
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_query(query, prompt_name=self.prompt_name)
            query_embedding = np.array([query_embedding]).astype('float32')

            # Retrieve top_k similar documents via the vector store
            D, I = self.vector_store.index.search(query_embedding, self.top_k)
            retrieved_docs = [self.vector_store.doc_chunks[i] for i in I[0]]

            # Optionally, re-rank the documents using stored embeddings
            # (i.e., doc.metadata['embedding'])
            if self.re_rank:
                retrieved_docs = self.re_rank_documents(query_embedding[0], retrieved_docs)

        # --------------------------------------------------
        # Case 2: If the model is MedCPT-Cross-Encoder (pairwise ranking)
        # --------------------------------------------------
        else:
            # We can’t do an embedding-based search in FAISS (no doc embeddings).
            # Instead, we do a brute-force rank of *all* doc_chunks:
            all_docs = self.vector_store.doc_chunks
            # Score each doc with cross-encoder
            scores = self.embedding_model.rank_texts(
                query, [doc.page_content for doc in all_docs]
            )
            # Sort docs by descending score
            doc_score_pairs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
            # Take top_k docs
            retrieved_docs = [pair[0] for pair in doc_score_pairs[:self.top_k]]

        # --------------------------------------------------
        # Build the return structure
        # --------------------------------------------------
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
        Re-rank the documents based on cosine similarity scores between
        the query embedding and the stored doc embeddings.
        :param query_embedding: The embedding vector of the query.
        :param documents: List of Document objects.
        :return: Re-ranked list of Document objects.
        """
        doc_embeddings = [doc.metadata['embedding'] for doc in documents]
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        # Pair documents with their similarity scores
        doc_similarity_pairs = list(zip(documents, similarities))
        # Sort documents by similarity score in descending order
        doc_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        # Return the sorted documents
        sorted_docs = [pair[0] for pair in doc_similarity_pairs]
        return sorted_docs
