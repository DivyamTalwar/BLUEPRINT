from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import time

from src.models.feature import Feature
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str = "blueprint-features",
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        """
        Initialize vector store

        Args:
            api_key: Pinecone API key
            index_name: Index name
            dimension: Vector dimension
            metric: Distance metric
            cloud: Cloud provider
            region: Cloud region
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.index = None

    def create_index(self, delete_if_exists: bool = False):
        """Create Pinecone index"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]

            if self.index_name in existing_indexes:
                if delete_if_exists:
                    logger.info("Deleting existing index: %s", self.index_name)
                    self.pc.delete_index(self.index_name)
                    time.sleep(1)
                else:
                    logger.info("Index already exists: %s", self.index_name)
                    self.index = self.pc.Index(self.index_name)
                    return

            logger.info("Creating index: %s", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)
            logger.info("Index created successfully")

        except Exception as e:
            logger.error("Error creating index: %s", str(e))
            raise

    def upsert_features(self, features: List[Feature], batch_size: int = 100):
        """
        Upload features to vector database

        Args:
            features: List of features with embeddings
            batch_size: Batch size for upload
        """
        if not self.index:
            self.index = self.pc.Index(self.index_name)

        logger.info("Upserting %d features to Pinecone", len(features))

        vectors = []
        for feature in features:
            if not feature.embedding:
                logger.warning("Feature %s has no embedding, skipping", feature.id)
                continue

            metadata = {
                "domain": feature.domain,
                "subdomain": feature.subdomain,
                "name": feature.name,
                "description": feature.description,
                "complexity": feature.complexity.value,
                "keywords": ",".join(feature.keywords),
                "use_cases": ",".join(feature.use_cases),
            }

            vectors.append((feature.id, feature.embedding, metadata))

        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)

            if i % (batch_size * 10) == 0:
                logger.info("Upserted %d/%d vectors", i, len(vectors))

        logger.info("Upsert complete: %d vectors", len(vectors))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar features

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Metadata filters

        Returns:
            List of matching features with scores
        """
        if not self.index:
            self.index = self.pc.Index(self.index_name)

        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
            )

            matches = []
            for match in results.matches:
                matches.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                    }
                )

            return matches

        except Exception as e:
            logger.error("Error searching: %s", str(e))
            return []

    def search_by_domain(
        self,
        query_embedding: List[float],
        domain: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search within specific domain"""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict={"domain": {"$eq": domain}},
        )

    def search_by_complexity(
        self,
        query_embedding: List[float],
        complexity: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search by complexity level"""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict={"complexity": {"$eq": complexity}},
        )

    def get_diverse_features(
        self,
        query_embedding: List[float],
        top_k: int = 50,
        diversity_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Get diverse features (avoid too similar results)

        Args:
            query_embedding: Query vector
            top_k: Initial retrieval size
            diversity_threshold: Similarity threshold for diversity

        Returns:
            Diverse feature set
        """
        # Get initial results
        results = self.search(query_embedding, top_k=top_k)

        if not results:
            return []

        # Select diverse results
        diverse = [results[0]]  # Always include top result

        for result in results[1:]:
            # Check if sufficiently different from already selected
            is_diverse = True
            for selected in diverse:
                if result["score"] - selected["score"] < diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse.append(result)

        logger.debug("Selected %d diverse features from %d", len(diverse), len(results))
        return diverse

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            self.index = self.pc.Index(self.index_name)

        try:
            stats = self.index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "total_vectors": stats.total_vector_count,
                "namespaces": stats.namespaces,
            }
        except Exception as e:
            logger.error("Error getting stats: %s", str(e))
            return {}

    def delete_index(self):
        """Delete the index"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info("Deleted index: %s", self.index_name)
        except Exception as e:
            logger.error("Error deleting index: %s", str(e))
