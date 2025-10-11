import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.core.llm_router import LLMRouter
from src.stage1.feature_generator import FeatureGenerator
from src.stage1.embedding_generator import EmbeddingGenerator
from src.stage1.vector_store import VectorStore
from src.models.taxonomy import get_taxonomy_stats
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_file="logs/feature_tree.log")
logger = get_logger(__name__)


def main():
    """Build complete feature tree"""
    print("=" * 70)
    print("BLUEPRINT FEATURE TREE GENERATION")
    print("=" * 70)

    # Load config
    config = get_config()

    # Check API keys
    env_check = config.validate_env_vars()
    missing = [k for k, v in env_check.items() if not v]

    if missing:
        print("\n❌ Missing API keys:")
        for key in missing:
            print(f"   - {key}")
        print("\nPlease add your API keys to .env file")
        return 1

    print("\n✓ API keys validated")

    # Show taxonomy stats
    tax_stats = get_taxonomy_stats()
    print(f"\n📊 Taxonomy:")
    print(f"   Domains: {tax_stats['total_domains']}")
    print(f"   Subdomains: {tax_stats['total_subdomains']}")
    print(f"   Avg subdomains per domain: {tax_stats['avg_subdomains_per_domain']:.1f}")

    # Calculate features to generate
    features_per_subdomain = 25  # ~25 features per subdomain
    total_features = tax_stats["total_subdomains"] * features_per_subdomain
    print(f"\n🎯 Target: ~{total_features} features")

    # Ask for confirmation
    print(f"\n⚠️  This will:")
    print(f"   1. Generate ~{total_features} features using LLM (~$5-10)")
    print(f"   2. Create embeddings (~$1-2)")
    print(f"   3. Upload to Pinecone (free tier: 100K vectors)")
    print(f"   Total cost: ~$6-12")

    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted.")
        return 0

    print("\n" + "=" * 70)
    print("STEP 1: GENERATE FEATURES")
    print("=" * 70)

    # Initialize LLM Router
    llm_router = LLMRouter(config.get_all())
    feature_generator = FeatureGenerator(llm_router)

    # Check if features already exist
    features_path = "data/features.json"
    if Path(features_path).exists():
        print(f"\n⚠️  Features file exists: {features_path}")
        response = input("Load existing features? (yes/no): ").strip().lower()

        if response == "yes":
            features = feature_generator.load_features(features_path)
            print(f"✓ Loaded {len(features)} existing features")
        else:
            print("\nGenerating new features...")
            features = feature_generator.generate_all_features(
                features_per_subdomain=features_per_subdomain,
                save_path=features_path,
            )
    else:
        print("\nGenerating features...")
        features = feature_generator.generate_all_features(
            features_per_subdomain=features_per_subdomain,
            save_path=features_path,
        )

    # Show stats
    stats = feature_generator.get_stats()
    print(f"\n✓ Generated features:")
    print(f"   Total: {stats['total_features']}")
    print(f"   By complexity: {stats['features_by_complexity']}")
    print(f"   Avg keywords: {stats['avg_keywords_per_feature']:.1f}")

    # Show LLM cost
    llm_stats = llm_router.get_stats()
    print(f"\n💰 LLM Cost:")
    print(f"   Total: ${llm_stats['total_cost']:.2f}")
    print(f"   API calls: {llm_stats['api_calls']}")

    print("\n" + "=" * 70)
    print("STEP 2: GENERATE EMBEDDINGS")
    print("=" * 70)

    # Check if embeddings already exist
    embeddings_path = "data/features_with_embeddings.json"
    if Path(embeddings_path).exists():
        print(f"\n⚠️  Embeddings file exists: {embeddings_path}")
        response = input("Skip embedding generation? (yes/no): ").strip().lower()

        if response == "yes":
            # Load features with embeddings
            from src.models.feature import Feature
            from src.utils.file_ops import FileOperations

            data = FileOperations.read_json(embeddings_path)
            features = [Feature.from_dict(f) for f in data["features"]]
            print(f"✓ Loaded {len(features)} features with embeddings")
        else:
            # Generate embeddings
            embedding_gen = EmbeddingGenerator(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-large",
            )

            features = embedding_gen.embed_features(features, batch_size=100)
            embedding_gen.save_embeddings(features, embeddings_path)

            emb_stats = embedding_gen.get_stats()
            print(f"\n✓ Generated embeddings:")
            print(f"   Model: {emb_stats['model']}")
            print(f"   Dimension: {emb_stats['dimension']}")
            print(f"   Cost: ${emb_stats['total_cost']:.2f}")
    else:
        # Generate embeddings
        embedding_gen = EmbeddingGenerator(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large",
        )

        features = embedding_gen.embed_features(features, batch_size=100)
        embedding_gen.save_embeddings(features, embeddings_path)

        emb_stats = embedding_gen.get_stats()
        print(f"\n✓ Generated embeddings:")
        print(f"   Model: {emb_stats['model']}")
        print(f"   Dimension: {emb_stats['dimension']}")
        print(f"   Cost: ${emb_stats['total_cost']:.2f}")

    print("\n" + "=" * 70)
    print("STEP 3: UPLOAD TO PINECONE")
    print("=" * 70)

    # Initialize Pinecone
    vector_store = VectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="blueprint-features",
        dimension=1536,
    )

    # Create index
    print("\nCreating/connecting to Pinecone index...")
    vector_store.create_index(delete_if_exists=False)

    # Upload features
    print("\nUploading features to Pinecone...")
    vector_store.upsert_features(features, batch_size=100)

    # Show final stats
    vs_stats = vector_store.get_stats()
    print(f"\n✓ Upload complete:")
    print(f"   Index: {vs_stats['index_name']}")
    print(f"   Total vectors: {vs_stats['total_vectors']}")

    print("\n" + "=" * 70)
    print("✅ FEATURE TREE BUILD COMPLETE!")
    print("=" * 70)

    print(f"\n📁 Files created:")
    print(f"   {features_path}")
    print(f"   {embeddings_path}")
    print(f"\n🎯 Ready for Stage 1 implementation!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
