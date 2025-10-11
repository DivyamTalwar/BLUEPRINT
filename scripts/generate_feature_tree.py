import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.taxonomy import UNIVERSAL_TAXONOMY
from src.models.feature import Feature, ComplexityLevel
from src.stage1.cohere_embeddings import CohereEmbeddings
from dotenv import load_dotenv

# Load environment
load_dotenv()

@dataclass
class FeatureGenerationStats:
    """Track feature generation statistics"""
    total_features: int = 0
    features_per_domain: Dict[str, int] = None
    features_per_complexity: Dict[str, int] = None
    generation_time: float = 0.0

    def __post_init__(self):
        if self.features_per_domain is None:
            self.features_per_domain = {}
        if self.features_per_complexity is None:
            self.features_per_complexity = {}


class FeatureTreeGenerator:
    """Generate comprehensive feature tree from taxonomy"""

    # Feature templates for different complexities
    BASIC_TEMPLATES = [
        "Basic {feature_name} implementation",
        "Simple {feature_name} functionality",
        "Core {feature_name} support",
        "Standard {feature_name} operations",
        "Essential {feature_name} features",
    ]

    INTERMEDIATE_TEMPLATES = [
        "Advanced {feature_name} with configuration",
        "{feature_name} with error handling",
        "Configurable {feature_name} system",
        "{feature_name} with validation",
        "Flexible {feature_name} implementation",
        "{feature_name} with logging and monitoring",
        "Robust {feature_name} with retries",
    ]

    ADVANCED_TEMPLATES = [
        "High-performance {feature_name} with caching",
        "Scalable {feature_name} architecture",
        "{feature_name} with distributed support",
        "Enterprise-grade {feature_name} system",
        "{feature_name} with advanced optimization",
        "Production-ready {feature_name} with monitoring",
        "{feature_name} with auto-scaling",
        "Cloud-native {feature_name} implementation",
    ]

    EXPERT_TEMPLATES = [
        "Highly optimized {feature_name} with ML",
        "{feature_name} with AI-powered optimization",
        "Next-generation {feature_name} architecture",
        "{feature_name} with real-time analytics",
        "Advanced {feature_name} with predictive capabilities",
        "Cutting-edge {feature_name} implementation",
        "{feature_name} with automated tuning",
        "State-of-the-art {feature_name} system",
    ]

    def __init__(self):
        self.features: List[Feature] = []
        self.stats = FeatureGenerationStats()

    def generate_all_features(self, target_count: int = 5000) -> List[Feature]:
        """
        Generate comprehensive feature set

        Args:
            target_count: Target number of features (default: 5000)

        Returns:
            List of generated features
        """
        print(f"\n*** GENERATING {target_count} FEATURES FROM UNIVERSAL TAXONOMY ***\n")
        start_time = time.time()

        # Calculate features per domain
        num_domains = len(UNIVERSAL_TAXONOMY)
        features_per_domain = target_count // num_domains

        print(f"Domains: {num_domains}")
        print(f"Target features per domain: ~{features_per_domain}")
        print()

        feature_id = 1

        for domain_key, domain in UNIVERSAL_TAXONOMY.items():
            print(f"Processing domain: {domain.name} ({domain_key})")

            # Generate features for this domain
            domain_features = self._generate_domain_features(
                domain_key=domain_key,
                domain_name=domain.name,
                domain_description=domain.description,
                subdomains=domain.subdomains,
                target_count=features_per_domain,
                start_id=feature_id
            )

            self.features.extend(domain_features)
            feature_id += len(domain_features)

            # Track stats
            self.stats.features_per_domain[domain_key] = len(domain_features)

            print(f"  -> Generated {len(domain_features)} features")

        # Update stats
        self.stats.total_features = len(self.features)
        self.stats.generation_time = time.time() - start_time

        # Calculate complexity distribution
        for feature in self.features:
            complexity = feature.complexity.value
            self.stats.features_per_complexity[complexity] = \
                self.stats.features_per_complexity.get(complexity, 0) + 1

        print(f"\n*** GENERATION COMPLETE ***")
        print(f"Total features: {self.stats.total_features}")
        print(f"Time taken: {self.stats.generation_time:.2f}s")
        print()

        return self.features

    def _generate_domain_features(
        self,
        domain_key: str,
        domain_name: str,
        domain_description: str,
        subdomains: List[str],
        target_count: int,
        start_id: int
    ) -> List[Feature]:
        """Generate features for a specific domain"""

        features = []
        features_per_subdomain = target_count // len(subdomains) if subdomains else target_count

        # Distribute across complexities
        complexity_distribution = {
            ComplexityLevel.BASIC: 0.30,      # 30%
            ComplexityLevel.INTERMEDIATE: 0.35,  # 35%
            ComplexityLevel.ADVANCED: 0.25,   # 25%
            ComplexityLevel.EXPERT: 0.10,     # 10%
        }

        feature_id = start_id

        for subdomain in subdomains:
            # Generate features for each complexity level
            for complexity, ratio in complexity_distribution.items():
                count = int(features_per_subdomain * ratio)

                for i in range(count):
                    feature = self._create_feature(
                        feature_id=f"feat_{feature_id:05d}",
                        subdomain=subdomain,
                        domain_key=domain_key,
                        domain_name=domain_name,
                        complexity=complexity,
                        index=i
                    )

                    features.append(feature)
                    feature_id += 1

        return features

    def _create_feature(
        self,
        feature_id: str,
        subdomain: str,
        domain_key: str,
        domain_name: str,
        complexity: ComplexityLevel,
        index: int
    ) -> Feature:
        """Create a single feature"""

        # Select template based on complexity
        templates = {
            ComplexityLevel.BASIC: self.BASIC_TEMPLATES,
            ComplexityLevel.INTERMEDIATE: self.INTERMEDIATE_TEMPLATES,
            ComplexityLevel.ADVANCED: self.ADVANCED_TEMPLATES,
            ComplexityLevel.EXPERT: self.EXPERT_TEMPLATES,
        }

        template_list = templates[complexity]
        template = template_list[index % len(template_list)]

        # Create readable feature name
        feature_name = subdomain.replace("_", " ").title()
        name = template.format(feature_name=feature_name)

        # Create description
        description = f"{name} for {domain_name}. " \
                     f"Provides {complexity.value}-level capabilities for {subdomain} operations."

        # Add use cases based on complexity
        use_cases = self._generate_use_cases(subdomain, complexity)

        # Add keywords
        keywords = [domain_key, subdomain, complexity.value, domain_name.lower().replace(" ", "_")]

        return Feature(
            id=feature_id,
            name=name,
            description=description,
            domain=domain_key,
            subdomain=subdomain,
            complexity=complexity,
            use_cases=use_cases,
            keywords=keywords
        )

    def _generate_use_cases(self, subdomain: str, complexity: ComplexityLevel) -> List[str]:
        """Generate use cases based on subdomain and complexity"""

        base_use_cases = {
            ComplexityLevel.BASIC: [
                f"Basic {subdomain} operations",
                f"Simple {subdomain} workflows",
            ],
            ComplexityLevel.INTERMEDIATE: [
                f"Production {subdomain} with monitoring",
                f"Scalable {subdomain} implementation",
                f"{subdomain} with error handling",
            ],
            ComplexityLevel.ADVANCED: [
                f"Enterprise {subdomain} architecture",
                f"High-performance {subdomain} system",
                f"{subdomain} with advanced features",
                f"Cloud-native {subdomain} deployment",
            ],
            ComplexityLevel.EXPERT: [
                f"AI-powered {subdomain} optimization",
                f"Next-gen {subdomain} with ML",
                f"Cutting-edge {subdomain} implementation",
                f"{subdomain} with predictive analytics",
                f"Research-grade {subdomain} system",
            ],
        }

        return base_use_cases[complexity]

    def save_features(self, output_path: str):
        """Save features to JSON file"""

        features_data = [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "domain": f.domain,
                "subdomain": f.subdomain,
                "complexity": f.complexity.value,
                "use_cases": f.use_cases,
                "keywords": f.keywords,
            }
            for f in self.features
        ]

        output = {
            "metadata": {
                "total_features": self.stats.total_features,
                "generation_time": self.stats.generation_time,
                "features_per_domain": self.stats.features_per_domain,
                "features_per_complexity": self.stats.features_per_complexity,
            },
            "features": features_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Features saved to: {output_path}")

    def print_stats(self):
        """Print generation statistics"""

        print("\n" + "=" * 80)
        print("FEATURE GENERATION STATISTICS")
        print("=" * 80)
        print()
        print(f"Total Features: {self.stats.total_features}")
        print(f"Generation Time: {self.stats.generation_time:.2f}s")
        print()

        print("Features by Complexity:")
        for complexity, count in sorted(self.stats.features_per_complexity.items()):
            percentage = (count / self.stats.total_features) * 100
            print(f"  {complexity:15s}: {count:5d} ({percentage:5.1f}%)")
        print()

        print("Features by Domain (Top 10):")
        sorted_domains = sorted(
            self.stats.features_per_domain.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for domain, count in sorted_domains:
            print(f"  {domain:30s}: {count:4d}")
        print()
        print("=" * 80)


class PineconeUploader:
    """Upload features to Pinecone with embeddings"""

    def __init__(self):
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            self.embeddings = CohereEmbeddings(api_key=cohere_api_key)
        else:
            self.embeddings = None
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""

        print("\n*** CHECKING PREREQUISITES ***\n")

        issues = []

        # Check Cohere API key
        if not os.getenv("COHERE_API_KEY"):
            issues.append("COHERE_API_KEY not set in .env")
        else:
            print("[OK] Cohere API key found")

        # Check Pinecone API key
        if not self.pinecone_api_key:
            issues.append("PINECONE_API_KEY not set in .env")
        else:
            print("[OK] Pinecone API key found")

        # Try importing pinecone
        try:
            import pinecone
            print("[OK] Pinecone library installed")
        except ImportError:
            issues.append("Pinecone library not installed (run: pip install pinecone-client)")

        print()

        if issues:
            print("[ERROR] Prerequisites not met:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("[OK] All prerequisites met!")
        return True

    def upload_features(self, features: List[Feature], batch_size: int = 100):
        """Upload features to Pinecone with embeddings"""

        if not self.check_prerequisites():
            print("\n[ERROR] Cannot upload - prerequisites not met")
            print("Please set API keys in .env file")
            return False

        print(f"\n*** UPLOADING {len(features)} FEATURES TO PINECONE ***\n")

        try:
            from pinecone import Pinecone, ServerlessSpec

            # Initialize Pinecone (v3 API)
            pc = Pinecone(api_key=self.pinecone_api_key)

            index_name = "blueprint-features"

            # Create index if doesn't exist
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                print(f"Creating Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=1024,  # Cohere embed-english-v3.0 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_env if self.pinecone_env != "gcp-starter" else "us-east-1"
                    )
                )
                time.sleep(5)  # Wait for index creation

            index = pc.Index(index_name)
            print(f"Connected to index: {index_name}")
            print()

            # Upload in batches
            total_batches = (len(features) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(features))
                batch = features[start_idx:end_idx]

                print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} features)...")

                # Generate embeddings for batch
                texts = [f"{f.name}. {f.description}" for f in batch]
                embeddings = self.embeddings.generate_embeddings_batch(texts, batch_size=len(batch))

                # Prepare vectors for upload
                vectors = []
                for i, feature in enumerate(batch):
                    vectors.append({
                        "id": feature.id,
                        "values": embeddings[i],
                        "metadata": {
                            "name": feature.name,
                            "description": feature.description[:500],  # Truncate for metadata
                            "domain": feature.domain,
                            "subdomain": feature.subdomain,
                            "complexity": feature.complexity.value,
                            "keywords": ",".join(feature.keywords[:5]),  # First 5 keywords
                        }
                    })

                # Upload to Pinecone
                index.upsert(vectors=vectors)

                print(f"  -> Uploaded {len(vectors)} vectors")
                time.sleep(0.5)  # Rate limiting

            # Get index stats
            stats = index.describe_index_stats()
            print(f"\n[SUCCESS] Upload complete!")
            print(f"Total vectors in index: {stats.total_vector_count}")

            return True

        except Exception as e:
            print(f"\n[ERROR] Upload failed: {e}")
            return False


def main():
    """Main execution"""

    print("=" * 80)
    print("BLUEPRINT FEATURE TREE GENERATOR - PRODUCTION READY")
    print("=" * 80)

    # Step 1: Generate features
    generator = FeatureTreeGenerator()
    features = generator.generate_all_features(target_count=5000)

    # Print stats
    generator.print_stats()

    # Step 2: Save to JSON
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "feature_tree.json"

    generator.save_features(str(output_path))

    # Step 3: Upload to Pinecone
    print("\n" + "=" * 80)
    print("PINECONE UPLOAD")
    print("=" * 80)

    uploader = PineconeUploader()

    # Ask user if they want to upload
    print("\nDo you want to upload features to Pinecone?")
    print("This requires COHERE_API_KEY and PINECONE_API_KEY in .env")
    response = input("Upload to Pinecone? (y/n): ").strip().lower()

    if response == 'y':
        success = uploader.upload_features(features, batch_size=100)
        if success:
            print("\n[SUCCESS] Features uploaded to Pinecone!")
        else:
            print("\n[WARNING] Upload skipped or failed")
            print("You can upload later by running this script again")
    else:
        print("\n[INFO] Skipping Pinecone upload")
        print("Features saved to JSON - you can upload later")

    print("\n" + "=" * 80)
    print("FEATURE TREE GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
