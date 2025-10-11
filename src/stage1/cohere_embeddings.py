from typing import List, Dict, Any
from tqdm import tqdm
import time
import cohere

from src.models.feature import Feature
from src.utils.logger import get_logger
from src.utils.file_ops import FileOperations

logger = get_logger(__name__)


class CohereEmbeddings:
    """Generate embeddings using Cohere"""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        """
        Initialize Cohere embeddings

        Args:
            api_key: Cohere API key
            model: Embedding model
                - embed-english-v3.0 (1024 dim, best for English)
                - embed-multilingual-v3.0 (1024 dim, multilingual)
        """
        self.client = cohere.Client(api_key)
        self.model = model
        self.dimension = 1024  # Cohere v3 models
        self.total_tokens = 0
        self.total_cost = 0.0

        logger.info("Cohere embeddings initialized: %s", model)

    def generate_embedding(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed
            input_type: "search_document" or "search_query"

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
            )

            embedding = response.embeddings[0]

            # Track usage
            # Cohere pricing: $0.10 per 1M tokens for embed-english-v3.0
            tokens = len(text.split()) * 1.5  # Estimate
            self.total_tokens += tokens
            self.total_cost += (tokens / 1_000_000) * 0.10

            return embedding

        except Exception as e:
            logger.error("Cohere embedding error: %s", str(e))
            return []

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 96,  # Cohere limit
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Generate embeddings in batches

        Args:
            texts: List of texts
            batch_size: Batch size (max 96 for Cohere)
            input_type: "search_document" or "search_query"

        Returns:
            List of embeddings
        """
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i : i + batch_size]

            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                )

                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)

                # Track usage
                tokens = sum(len(t.split()) * 1.5 for t in batch)
                self.total_tokens += tokens
                self.total_cost += (tokens / 1_000_000) * 0.10

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error("Batch embedding error: %s", str(e))
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])

        return embeddings

    def embed_features(
        self, features: List[Feature], batch_size: int = 96
    ) -> List[Feature]:
        """
        Generate embeddings for features

        Args:
            features: List of features
            batch_size: Batch size

        Returns:
            Features with embeddings
        """
        logger.info("Generating Cohere embeddings for %d features", len(features))

        # Extract search texts
        texts = [f.get_search_text() for f in features]

        # Generate embeddings as documents
        embeddings = self.generate_embeddings_batch(
            texts,
            batch_size,
            input_type="search_document"
        )

        # Assign embeddings to features
        for feature, embedding in zip(features, embeddings):
            feature.embedding = embedding

        logger.info(
            "Generated embeddings: tokens=%d, cost=$%.4f",
            self.total_tokens,
            self.total_cost,
        )

        return features

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        model: str = "rerank-english-v3.0",
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere Rerank

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results
            model: Rerank model

        Returns:
            Reranked results with scores
        """
        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=model,
            )

            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    "document": documents[result.index],
                })

            logger.info("Reranked %d documents to top %d", len(documents), top_k)
            return results

        except Exception as e:
            logger.error("Rerank error: %s", str(e))
            return []

    def save_embeddings(self, features: List[Feature], filepath: str):
        """Save features with embeddings"""
        data = {
            "features": [f.to_dict() for f in features],
            "count": len(features),
            "model": self.model,
            "dimension": self.dimension,
            "provider": "cohere",
        }
        FileOperations.write_json(filepath, data)
        logger.info("Saved Cohere embeddings to %s", filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            "model": self.model,
            "provider": "cohere",
            "dimension": self.dimension,
            "total_tokens": int(self.total_tokens),
            "total_cost": round(self.total_cost, 4),
            "cost_per_1k_tokens": round(
                (self.total_cost / self.total_tokens * 1000)
                if self.total_tokens > 0
                else 0,
                4,
            ),
        }
