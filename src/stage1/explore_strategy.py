from typing import List, Dict, Any, Set
import random

from src.stage1.vector_store import VectorStore
from src.stage1.embedding_generator import EmbeddingGenerator
from src.stage1.user_input_processor import RepositoryRequest
from src.models.feature import Feature, ComplexityLevel
from src.models.taxonomy import get_all_domains, UNIVERSAL_TAXONOMY
from src.core.llm_router import LLMRouter
from src.utils.logger import get_logger
import json

logger = get_logger(__name__)


class ExploreStrategy:
    """Explore strategy for diverse feature discovery"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        llm_router: LLMRouter,
    ):
        """
        Initialize explore strategy

        Args:
            vector_store: Vector database
            embedding_generator: Embedding generator
            llm_router: LLM router
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_router = llm_router

    def discover_diverse_features(
        self,
        request: RepositoryRequest,
        visited_domains: Set[str],
        top_k: int = 10,
        diversity_threshold: float = 0.85,
    ) -> List[Feature]:
        """
        Discover diverse features from unvisited domains

        Args:
            request: Repository request
            visited_domains: Domains already explored
            top_k: Number of features to discover
            diversity_threshold: Similarity threshold for diversity

        Returns:
            Diverse features
        """
        logger.info(
            "Explore: Discovering %d diverse features (visited: %d domains)",
            top_k,
            len(visited_domains),
        )

        # Get unvisited domains
        all_domains = set(get_all_domains())
        unvisited = all_domains - visited_domains

        if not unvisited:
            logger.warning("All domains visited, selecting from visited domains")
            unvisited = all_domains

        # Sample domains to explore
        num_domains = min(3, len(unvisited))
        explore_domains = random.sample(list(unvisited), num_domains)

        logger.info("Exploring domains: %s", explore_domains)

        # Build exploration query
        query_text = self._build_exploration_query(request, explore_domains)
        query_embedding = self.embedding_generator.generate_embedding(query_text)

        if not query_embedding:
            logger.error("Failed to generate exploration query embedding")
            return []

        # Retrieve diverse features
        all_matches = []

        for domain in explore_domains:
            matches = self.vector_store.search_by_domain(
                query_embedding=query_embedding,
                domain=domain,
                top_k=top_k,
            )
            all_matches.extend(matches)

        # Get diverse subset
        diverse_matches = self.vector_store.get_diverse_features(
            query_embedding=query_embedding,
            top_k=len(all_matches),
            diversity_threshold=diversity_threshold,
        )

        # Convert to features
        from src.stage1.exploit_strategy import ExploitStrategy
        exploit = ExploitStrategy(self.vector_store, self.embedding_generator)
        features = exploit._matches_to_features(diverse_matches[:top_k])

        logger.info("Explore: Discovered %d diverse features", len(features))

        return features

    def suggest_complementary_features(
        self,
        request: RepositoryRequest,
        selected_features: List[Feature],
        top_k: int = 10,
    ) -> List[Feature]:
        """
        Suggest features that complement already selected features

        Args:
            request: Repository request
            selected_features: Already selected features
            top_k: Number of suggestions

        Returns:
            Complementary features
        """
        logger.info("Explore: Suggesting %d complementary features", top_k)

        # Build prompt for LLM
        selected_names = [f.name for f in selected_features[:20]]

        prompt = f"""Given this repository goal and already selected features, suggest diverse complementary features.

Repository Type: {request.repo_type}
Primary Domain: {request.primary_domain}

Already Selected Features ({len(selected_features)}):
{', '.join(selected_names)}

Task: Suggest {top_k} diverse, complementary features that would enhance this repository.

Focus on:
- Cross-domain functionality (integrate different aspects)
- Supporting infrastructure (testing, logging, monitoring)
- Quality features (error handling, validation, security)
- Developer experience (CLI, documentation, examples)
- Non-obvious but valuable additions

Avoid:
- Features similar to already selected
- Redundant functionality
- Overly complex features for this project

Output as JSON:
{{
  "features": [
    {{
      "name": "Feature name",
      "description": "What it does",
      "domain": "domain_key",
      "reasoning": "Why this adds value"
    }}
  ]
}}"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                temperature=0.7,  # Higher temp for creativity
                max_tokens=2000,
                json_mode=True,
            )

            data = json.loads(response.content)
            suggestions = data.get("features", [])

            # Search for similar features in vector DB
            complementary = []

            for suggestion in suggestions[:top_k]:
                # Generate embedding for suggestion
                search_text = f"{suggestion['name']}: {suggestion['description']}"
                embedding = self.embedding_generator.generate_embedding(search_text)

                if not embedding:
                    continue

                # Search for matching features
                matches = self.vector_store.search(
                    query_embedding=embedding,
                    top_k=1,
                )

                if matches:
                    from src.stage1.exploit_strategy import ExploitStrategy
                    exploit = ExploitStrategy(self.vector_store, self.embedding_generator)
                    features = exploit._matches_to_features(matches)

                    if features:
                        feature = features[0]
                        feature.metadata["llm_suggestion"] = suggestion.get("reasoning", "")
                        complementary.append(feature)

            logger.info("Explore: Found %d complementary features", len(complementary))

            return complementary

        except Exception as e:
            logger.error("Complementary feature suggestion failed: %s", str(e))
            return []

    def discover_by_keywords(
        self,
        keywords: List[str],
        exclude_domains: Set[str],
        top_k: int = 10,
    ) -> List[Feature]:
        """
        Discover features by keywords, excluding certain domains

        Args:
            keywords: Search keywords
            exclude_domains: Domains to exclude
            top_k: Number of features

        Returns:
            Discovered features
        """
        logger.info("Explore: Discovering by keywords: %s", keywords[:5])

        # Build query from keywords
        query_text = " ".join(keywords)
        query_embedding = self.embedding_generator.generate_embedding(query_text)

        if not query_embedding:
            return []

        # Search across all domains
        matches = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Get more, then filter
        )

        # Filter out excluded domains
        filtered_matches = []
        for match in matches:
            domain = match.get("metadata", {}).get("domain", "")
            if domain not in exclude_domains:
                filtered_matches.append(match)

            if len(filtered_matches) >= top_k:
                break

        # Convert to features
        from src.stage1.exploit_strategy import ExploitStrategy
        exploit = ExploitStrategy(self.vector_store, self.embedding_generator)
        features = exploit._matches_to_features(filtered_matches[:top_k])

        logger.info("Explore: Discovered %d features by keywords", len(features))

        return features

    def discover_cross_domain_features(
        self,
        request: RepositoryRequest,
        primary_domain: str,
        top_k: int = 10,
    ) -> List[Feature]:
        """
        Discover features that bridge multiple domains

        Args:
            request: Repository request
            primary_domain: Primary domain
            top_k: Number of features

        Returns:
            Cross-domain features
        """
        logger.info("Explore: Discovering cross-domain features")

        # Identify related domains
        related_domains = self._get_related_domains(primary_domain)

        # Sample features from each related domain
        cross_domain = []

        for domain in related_domains[:3]:
            # Build query
            query_text = f"{request.repo_type} {domain} integration functionality"
            embedding = self.embedding_generator.generate_embedding(query_text)

            if not embedding:
                continue

            # Search in this domain
            matches = self.vector_store.search_by_domain(
                query_embedding=embedding,
                domain=domain,
                top_k=max(1, top_k // 3),  # Ensure positive
            )

            # Convert to features
            from src.stage1.exploit_strategy import ExploitStrategy
            exploit = ExploitStrategy(self.vector_store, self.embedding_generator)
            features = exploit._matches_to_features(matches)
            cross_domain.extend(features)

        logger.info("Explore: Discovered %d cross-domain features", len(cross_domain))

        return cross_domain[:top_k]

    def _build_exploration_query(
        self,
        request: RepositoryRequest,
        explore_domains: List[str],
    ) -> str:
        """Build exploration query"""
        parts = [
            f"Repository: {request.repo_type}",
            f"Exploring domains: {', '.join(explore_domains)}",
        ]

        if request.recommended_features:
            parts.append(f"Looking for: {', '.join(request.recommended_features[:3])}")

        query = " | ".join(parts)

        return query

    def _get_related_domains(self, domain: str) -> List[str]:
        """Get domains related to given domain"""
        # Define domain relationships
        relationships = {
            "api_web": ["authentication_security", "database_orm", "caching_optimization"],
            "data_operations": ["database_orm", "file_system", "caching_optimization"],
            "machine_learning": ["data_operations", "data_visualization", "file_system"],
            "authentication_security": ["api_web", "database_orm", "logging_monitoring"],
            "testing_quality": ["logging_monitoring", "error_handling", "cli_interface"],
        }

        related = relationships.get(domain, [])

        # Add some random domains for serendipity
        all_domains = list(get_all_domains())
        random_domains = random.sample(all_domains, min(2, len(all_domains)))

        return related + random_domains
