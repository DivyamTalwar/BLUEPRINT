import sys
import os
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.core.llm_router import LLMRouter
from src.stage1.feature_generator import FeatureGenerator
from src.models.taxonomy import UNIVERSAL_TAXONOMY, get_taxonomy_stats
from src.models.feature import ComplexityLevel


def test_taxonomy():
    """Test taxonomy"""
    print("\n" + "=" * 60)
    print("TEST: TAXONOMY")
    print("=" * 60)

    stats = get_taxonomy_stats()
    print(f"\n✓ Taxonomy loaded:")
    print(f"   Domains: {stats['total_domains']}")
    print(f"   Subdomains: {stats['total_subdomains']}")

    # Show first few domains
    print(f"\nFirst 5 domains:")
    for i, (name, domain) in enumerate(list(UNIVERSAL_TAXONOMY.items())[:5]):
        print(f"   {i+1}. {domain.name}")
        print(f"      Subdomains: {len(domain.subdomains)}")
        print(f"      Keywords: {', '.join(domain.keywords[:5])}")

    return True


def test_feature_generation():
    """Test generating a small batch of features"""
    print("\n" + "=" * 60)
    print("TEST: FEATURE GENERATION (10 features)")
    print("=" * 60)

    # Check API keys
    config = get_config()
    env_check = config.validate_env_vars()

    if not env_check.get("OPENAI_API_KEY"):
        print("\n⚠️  OpenAI API key not set. Skipping LLM test.")
        return False

    print("\n✓ API keys validated")

    # Initialize
    llm_router = LLMRouter(config.get_all())
    feature_gen = FeatureGenerator(llm_router)

    # Generate small batch
    print("\nGenerating 10 features for 'data_operations/file_io'...")
    features = feature_gen.generate_features_for_domain(
        domain_name="data_operations",
        subdomain="file_io",
        num_features=10,
    )

    print(f"\n✓ Generated {len(features)} features")

    # Show sample features
    print(f"\nSample features:")
    for i, feat in enumerate(features[:3]):
        print(f"\n{i+1}. {feat.name}")
        print(f"   Description: {feat.description}")
        print(f"   Complexity: {feat.complexity.value}")
        print(f"   Keywords: {', '.join(feat.keywords)}")

    # Show stats
    stats = llm_router.get_stats()
    print(f"\n💰 Cost for 10 features: ${stats['total_cost']:.4f}")

    return True


def test_embeddings():
    """Test embedding generation"""
    print("\n" + "=" * 60)
    print("TEST: EMBEDDINGS")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OpenAI API key not set. Skipping embedding test.")
        return False

    from src.stage1.embedding_generator import EmbeddingGenerator

    embedding_gen = EmbeddingGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-large",
    )

    # Test single embedding
    text = "Load CSV file and parse data"
    print(f"\nGenerating embedding for: '{text}'")

    embedding = embedding_gen.generate_embedding(text)

    print(f"\n✓ Generated embedding:")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   Cost: ${embedding_gen.total_cost:.4f}")

    return True


def test_vector_store():
    """Test Pinecone connection"""
    print("\n" + "=" * 60)
    print("TEST: VECTOR STORE")
    print("=" * 60)

    if not os.getenv("PINECONE_API_KEY"):
        print("\n⚠️  Pinecone API key not set. Skipping vector store test.")
        return False

    from src.stage1.vector_store import VectorStore

    vector_store = VectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="blueprint-features-test",
        dimension=1536,
    )

    # Test connection
    print("\nConnecting to Pinecone...")
    try:
        vector_store.create_index(delete_if_exists=False)
        print("✓ Connected to Pinecone")

        # Get stats
        stats = vector_store.get_stats()
        print(f"\n✓ Index stats:")
        print(f"   Name: {stats.get('index_name', 'N/A')}")
        print(f"   Vectors: {stats.get('total_vectors', 0)}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("BLUEPRINT FEATURE TREE - COMPONENT TESTS")
    print("=" * 60)

    tests = [
        ("Taxonomy", test_taxonomy),
        ("Feature Generation", test_feature_generation),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL/SKIP"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! Ready to build feature tree.")
        print("\nNext step: python build_feature_tree.py")
        return 0
    else:
        print(f"\n⚠️  Some tests failed/skipped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
