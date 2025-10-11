import json
import uuid
from typing import List, Dict, Any
from tqdm import tqdm

from src.core.llm_router import LLMRouter
from src.models.feature import Feature, ComplexityLevel
from src.models.taxonomy import UNIVERSAL_TAXONOMY, get_all_domains
from src.utils.logger import get_logger
from src.utils.file_ops import FileOperations

logger = get_logger(__name__)


class FeatureGenerator:
    """Generate synthetic features for the feature tree"""

    def __init__(self, llm_router: LLMRouter):
        """
        Initialize feature generator

        Args:
            llm_router: LLM router for API calls
        """
        self.llm_router = llm_router
        self.generated_features: List[Feature] = []

    def generate_features_for_domain(
        self,
        domain_name: str,
        subdomain: str,
        num_features: int = 20,
        complexity_distribution: Dict[ComplexityLevel, float] = None,
    ) -> List[Feature]:
        """
        Generate features for a specific domain/subdomain

        Args:
            domain_name: Domain name
            subdomain: Subdomain name
            num_features: Number of features to generate
            complexity_distribution: Distribution of complexity levels

        Returns:
            List of generated features
        """
        if complexity_distribution is None:
            complexity_distribution = {
                ComplexityLevel.BASIC: 0.4,
                ComplexityLevel.INTERMEDIATE: 0.35,
                ComplexityLevel.ADVANCED: 0.20,
                ComplexityLevel.EXPERT: 0.05,
            }

        domain = UNIVERSAL_TAXONOMY.get(domain_name)
        if not domain:
            logger.error("Domain not found: %s", domain_name)
            return []

        # Calculate features per complexity level
        features_per_complexity = {}
        remaining = num_features

        for complexity, ratio in complexity_distribution.items():
            count = int(num_features * ratio)
            features_per_complexity[complexity] = count
            remaining -= count

        # Distribute remaining features
        if remaining > 0:
            features_per_complexity[ComplexityLevel.BASIC] += remaining

        logger.info(
            "Generating %d features for %s/%s: %s",
            num_features,
            domain_name,
            subdomain,
            features_per_complexity,
        )

        generated = []

        for complexity, count in features_per_complexity.items():
            if count > 0:
                features = self._generate_batch(
                    domain_name, subdomain, complexity, count
                )
                generated.extend(features)

        self.generated_features.extend(generated)
        return generated

    def _generate_batch(
        self,
        domain_name: str,
        subdomain: str,
        complexity: ComplexityLevel,
        count: int,
    ) -> List[Feature]:
        """Generate batch of features"""
        domain = UNIVERSAL_TAXONOMY[domain_name]

        prompt = f"""Generate {count} diverse software features for this domain/subdomain:

Domain: {domain.name}
Description: {domain.description}
Subdomain: {subdomain}
Complexity: {complexity.value}

Requirements:
1. Each feature should be realistic and implementable
2. Features should be diverse and non-overlapping
3. Include clear, concise descriptions
4. Add relevant keywords (3-5 per feature)
5. List 2-3 use cases
6. Mention any dependencies (other features needed)

Output as JSON array:
{{
  "features": [
    {{
      "name": "Feature name (concise, 2-5 words)",
      "description": "Clear description of what it does (1-2 sentences)",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "use_cases": ["use case 1", "use case 2"],
      "dependencies": ["dependency1", "dependency2"]
    }}
  ]
}}

Generate {count} features:"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=2000,
                json_mode=True,
            )

            data = json.loads(response.content)
            features_data = data.get("features", [])

            features = []
            for feat_data in features_data[:count]:
                feature = Feature(
                    id=str(uuid.uuid4()),
                    domain=domain_name,
                    subdomain=subdomain,
                    name=feat_data.get("name", ""),
                    description=feat_data.get("description", ""),
                    complexity=complexity,
                    keywords=feat_data.get("keywords", []),
                    use_cases=feat_data.get("use_cases", []),
                    dependencies=feat_data.get("dependencies", []),
                )
                features.append(feature)

            logger.debug(
                "Generated %d features for %s/%s (%s)",
                len(features),
                domain_name,
                subdomain,
                complexity.value,
            )

            return features

        except Exception as e:
            logger.error("Error generating features: %s", str(e))
            return []

    def generate_all_features(
        self,
        features_per_subdomain: int = 20,
        save_path: str = "data/features.json",
    ) -> List[Feature]:
        """
        Generate features for all domains and subdomains

        Args:
            features_per_subdomain: Features to generate per subdomain
            save_path: Path to save features

        Returns:
            List of all generated features
        """
        all_features = []
        total_subdomains = sum(
            len(domain.subdomains) for domain in UNIVERSAL_TAXONOMY.values()
        )

        logger.info(
            "Generating features for %d domains, %d subdomains",
            len(UNIVERSAL_TAXONOMY),
            total_subdomains,
        )

        with tqdm(total=total_subdomains, desc="Generating features") as pbar:
            for domain_name, domain in UNIVERSAL_TAXONOMY.items():
                for subdomain in domain.subdomains:
                    features = self.generate_features_for_domain(
                        domain_name, subdomain, features_per_subdomain
                    )
                    all_features.extend(features)
                    pbar.update(1)

                    # Save periodically
                    if len(all_features) % 100 == 0:
                        self._save_features(all_features, save_path)

        # Final save
        self._save_features(all_features, save_path)

        logger.info("Generated %d total features", len(all_features))
        return all_features

    def _save_features(self, features: List[Feature], filepath: str):
        """Save features to file"""
        data = [f.to_dict() for f in features]
        FileOperations.write_json(filepath, {"features": data, "count": len(features)})

    def load_features(self, filepath: str) -> List[Feature]:
        """Load features from file"""
        data = FileOperations.read_json(filepath)
        if not data:
            return []

        features = [Feature.from_dict(f) for f in data.get("features", [])]
        self.generated_features = features
        logger.info("Loaded %d features from %s", len(features), filepath)
        return features

    def get_features_by_domain(self, domain_name: str) -> List[Feature]:
        """Get all features for a domain"""
        return [f for f in self.generated_features if f.domain == domain_name]

    def get_features_by_subdomain(
        self, domain_name: str, subdomain: str
    ) -> List[Feature]:
        """Get all features for a subdomain"""
        return [
            f
            for f in self.generated_features
            if f.domain == domain_name and f.subdomain == subdomain
        ]

    def get_features_by_complexity(self, complexity: ComplexityLevel) -> List[Feature]:
        """Get all features of specific complexity"""
        return [f for f in self.generated_features if f.complexity == complexity]

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = len(self.generated_features)

        by_domain = {}
        by_complexity = {}

        for feature in self.generated_features:
            # Count by domain
            if feature.domain not in by_domain:
                by_domain[feature.domain] = 0
            by_domain[feature.domain] += 1

            # Count by complexity
            if feature.complexity not in by_complexity:
                by_complexity[feature.complexity.value] = 0
            by_complexity[feature.complexity.value] += 1

        return {
            "total_features": total,
            "features_by_domain": by_domain,
            "features_by_complexity": by_complexity,
            "avg_keywords_per_feature": (
                sum(len(f.keywords) for f in self.generated_features) / total
                if total > 0
                else 0
            ),
        }
