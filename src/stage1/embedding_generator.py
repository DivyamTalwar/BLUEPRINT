from typing import List, Dict, Any
from tqdm import tqdm
import time

from openai import OpenAI

from src.models.feature import Feature
from src.utils.logger import get_logger
from src.utils.file_ops import FileOperations

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for features using OpenAI"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """
        Initialize embedding generator

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # Default for text-embedding-3-large
        self.total_tokens = 0
        self.total_cost = 0.0

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.model)

            embedding = response.data[0].embedding
            tokens = response.usage.total_tokens

            # Track usage
            self.total_tokens += tokens
            # Pricing: $0.13 per 1M tokens for text-embedding-3-large
            self.total_cost += (tokens / 1_000_000) * 0.13

            return embedding

        except Exception as e:
            logger.error("Error generating embedding: %s", str(e))
            return []

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings in batches

        Args:
            texts: List of texts
            batch_size: Batch size

        Returns:
            List of embeddings
        """
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i : i + batch_size]

            try:
                response = self.client.embeddings.create(input=batch, model=self.model)

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

                tokens = response.usage.total_tokens
                self.total_tokens += tokens
                self.total_cost += (tokens / 1_000_000) * 0.13

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error("Error in batch embedding: %s", str(e))
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])

        return embeddings

    def embed_features(
        self, features: List[Feature], batch_size: int = 100
    ) -> List[Feature]:
        """
        Generate embeddings for features

        Args:
            features: List of features
            batch_size: Batch size

        Returns:
            Features with embeddings
        """
        logger.info("Generating embeddings for %d features", len(features))

        # Extract search texts
        texts = [f.get_search_text() for f in features]

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size)

        # Assign embeddings to features
        for feature, embedding in zip(features, embeddings):
            feature.embedding = embedding

        logger.info(
            "Generated embeddings: tokens=%d, cost=$%.4f",
            self.total_tokens,
            self.total_cost,
        )

        return features

    def save_embeddings(self, features: List[Feature], filepath: str):
        """Save features with embeddings"""
        data = {
            "features": [f.to_dict() for f in features],
            "count": len(features),
            "model": self.model,
            "dimension": self.dimension,
        }
        FileOperations.write_json(filepath, data)
        logger.info("Saved embeddings to %s", filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cost_per_1k_tokens": (self.total_cost / self.total_tokens * 1000)
            if self.total_tokens > 0
            else 0,
        }
