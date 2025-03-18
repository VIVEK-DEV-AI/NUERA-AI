import sqlite3
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack as sparse_vstack

class KnowledgeBase:
    def __init__(self, db_path="knowledge_base.db", index_path="faiss.index"):
        self.db_path = db_path
        self.index_path = index_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()
        self.embedder = SentenceTransformer('all-MiniLM-L12-v2')  # Improved model
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.embedding_dim = 384
        self.titles = []
        self.contents = []
        self.tags = []  # Tags for metadata filtering
        self.tfidf = TfidfVectorizer()
        self.tfidf_fitted = False
        self._load_data()
        self._build_hybrid_index()

    def create_table(self):
        """Create the knowledge base table if it doesn't exist"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT NOT NULL,
                tags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _load_data(self):
        """Load all entries from SQLite database"""
        self.cursor.execute("SELECT title, content, tags FROM knowledge_base")
        entries = self.cursor.fetchall()
        self.titles = [entry[0] for entry in entries]
        self.contents = [entry[1] for entry in entries]
        self.tags = [entry[2] for entry in entries]

    def _build_hybrid_index(self):
        """Build FAISS index and TF-IDF matrix with proper initialization"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Cosine similarity

            # Build vector index
            if self.contents:
                embeddings = self.embedder.encode(self.contents)
                embeddings = embeddings.astype(np.float32)
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                faiss.write_index(self.index, self.index_path)

            # Build TF-IDF matrix
            if self.contents and not self.tfidf_fitted:
                self.tfidf_matrix = self.tfidf.fit_transform(self.contents)
                self.tfidf_fitted = True
            else:
                self.tfidf_matrix = None

        except Exception as e:
            print(f"Index build error: {str(e)}")
            self.tfidf_matrix = None

    def add_text(self, text, title="Manual Entry", tags=None):
        """Add new text with proper index updates"""
        try:
            # Add to SQLite database
            self.cursor.execute(
                "INSERT INTO knowledge_base (title, content, tags) VALUES (?, ?, ?)",
                (title, text, tags)
            )
            self.conn.commit()

            # Update in-memory data
            self.titles.append(title)
            self.contents.append(text)
            self.tags.append(tags)

            # Generate embedding and update FAISS index
            new_embedding = self.embedder.encode([text]).astype(np.float32)
            faiss.normalize_L2(new_embedding)
            self.index.add(new_embedding)  # Add to FAISS
            faiss.write_index(self.index, self.index_path)  # Save to disk

            # Update TF-IDF matrix
            if self.tfidf_fitted:
                new_tfidf = self.tfidf.transform([text])
                if self.tfidf_matrix is not None:
                    self.tfidf_matrix = sparse_vstack([self.tfidf_matrix, new_tfidf])  # Append new row
                else:
                    self.tfidf_matrix = new_tfidf
            else:
                self.tfidf_matrix = self.tfidf.fit_transform(self.contents)
                self.tfidf_fitted = True  # Mark as fitted

        except Exception as e:
            print(f"Error adding text: {str(e)}")

    def search(self, query, filters={}, k=5):
        """Hybrid search with full validation"""
        if not self.contents:
            return []

        try:
            # Filter contents by tags if provided
            filtered_indices = [
                i for i, tag in enumerate(self.tags) 
                if not filters or any(f in (tag or "") for f in filters.get("tags", []))
            ]
            filtered_contents = [self.contents[i] for i in filtered_indices]

            # 1. Vector search (FAISS)
            query_embedding = self.embedder.encode([query]).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            _, faiss_indices = self.index.search(query_embedding, k*3)
            
            # Validate FAISS indices
            faiss_indices = [i for i in faiss_indices[0] if i < len(self.contents)]
            faiss_results = [self.contents[i] for i in faiss_indices]

            # 2. Keyword search (TF-IDF)
            tfidf_results = []
            if self.tfidf_fitted and self.tfidf_matrix is not None:
                tfidf_scores = self.tfidf.transform([query]).toarray().flatten()
                valid_indices = [i for i in range(len(tfidf_scores)) if i < len(self.contents)]
                tfidf_indices = np.argsort(-tfidf_scores[valid_indices])[:k*3]
                tfidf_results = [self.contents[i] for i in tfidf_indices]

            # 3. Combine and deduplicate results
            combined = list(dict.fromkeys(faiss_results + tfidf_results))[:k*3]

            # 4. Re-rank with cross-encoder
            pairs = [(query, content) for content in combined]
            scores = self.cross_encoder.predict(pairs)
            ranked = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)

            # Format results with titles
            results = []
            for content, _ in ranked[:k]:
                idx = self.contents.index(content)
                results.append((self.titles[idx], content))
            
            # Debug logs
            print(f"\n🔍 Query: {query}")
            print("Retrieved Contexts:")
            for title, content in results:
                print(f"  - Title: {title} | Content: {content[:100]}...")
            
            return results

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def get_all_knowledge(self):
        """Retrieve all stored entries with error handling"""
        try:
            self.cursor.execute("SELECT title, content FROM knowledge_base")
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Database error: {str(e)}")
            return []

    def __del__(self):
        try:
            self.conn.close()
        except Exception as e:
            print(f"Connection close error: {str(e)}")