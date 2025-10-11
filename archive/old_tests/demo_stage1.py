import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.core.llm_router import LLMRouter
from src.stage1.user_input_processor import UserInputProcessor
from src.models.taxonomy import get_taxonomy_stats


def demo_user_input_processing():
    print("=" * 70)
    print("DEMO: USER INPUT PROCESSING")
    print("=" * 70)

    config = get_config()

    # Check API key
    env_check = config.validate_env_vars()
    if not env_check.get("OPENAI_API_KEY"):
        print("\n⚠️  OpenAI API key required for this demo")
        print("This component parses user descriptions using LLM")
        return

    # Initialize
    llm_router = LLMRouter(config.get_all())
    processor = UserInputProcessor(llm_router)

    # Example descriptions
    examples = [
        "Build a REST API for blog management with authentication",
        "Create a CLI tool for data processing and visualization",
        "Machine learning library for time series forecasting",
    ]

    print("\nWill process these example descriptions:")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")

    response = input("\nSelect example (1-3) or enter your own: ").strip()

    if response in ["1", "2", "3"]:
        description = examples[int(response) - 1]
    else:
        description = response if response else examples[0]

    print(f"\n✓ Processing: {description}")
    print("\nCalling LLM to analyze description...")

    # Process
    request = processor.process(description)

    # Display results
    print("\n" + "-" * 70)
    print("PARSED REQUEST")
    print("-" * 70)
    print(f"\nRepository Type: {request.repo_type}")
    print(f"Primary Domain: {request.primary_domain}")
    print(f"Complexity: {request.complexity_estimate}")

    print(f"\nSubdomains ({len(request.subdomains)}):")
    for sd in request.subdomains[:5]:
        print(f"  - {sd}")

    print(f"\nExplicit Requirements ({len(request.explicit_requirements)}):")
    for req in request.explicit_requirements[:5]:
        print(f"  - {req}")

    print(f"\nImplicit Requirements ({len(request.implicit_requirements)}):")
    for req in request.implicit_requirements[:5]:
        print(f"  - {req}")

    print(f"\nRecommended Features ({len(request.recommended_features)}):")
    for feat in request.recommended_features[:5]:
        print(f"  - {feat}")

    # Target features
    target = processor.get_target_feature_count(request)
    print(f"\nTarget Features: {target}")

    # Cost
    stats = llm_router.get_stats()
    print(f"\nCost: ${stats['total_cost']:.4f}")

    print("\n✅ Demo complete!")


def demo_taxonomy():
    """Demo taxonomy"""
    print("\n" + "=" * 70)
    print("DEMO: UNIVERSAL TAXONOMY")
    print("=" * 70)

    from src.models.taxonomy import UNIVERSAL_TAXONOMY

    stats = get_taxonomy_stats()

    print(f"\nTaxonomy Statistics:")
    print(f"  Total domains: {stats['total_domains']}")
    print(f"  Total subdomains: {stats['total_subdomains']}")
    print(f"  Avg subdomains/domain: {stats['avg_subdomains_per_domain']:.1f}")

    print(f"\nAll {stats['total_domains']} Domains:")
    for i, (key, domain) in enumerate(UNIVERSAL_TAXONOMY.items(), 1):
        print(f"\n{i}. {domain.name} ({key})")
        print(f"   Description: {domain.description}")
        print(f"   Subdomains ({len(domain.subdomains)}): {', '.join(domain.subdomains[:4])}...")
        print(f"   Keywords: {', '.join(domain.keywords)}")


def main():
    """Run demos"""
    print("=" * 70)
    print("BLUEPRINT STAGE 1 - COMPONENT DEMOS")
    print("=" * 70)

    print("\nAvailable demos:")
    print("1. Universal Taxonomy (no API needed)")
    print("2. User Input Processing (requires API key)")

    response = input("\nSelect demo (1-2): ").strip()

    if response == "1":
        demo_taxonomy()
    elif response == "2":
        demo_user_input_processing()
    else:
        print("\nRunning both demos...\n")
        demo_taxonomy()
        print("\n")
        demo_user_input_processing()

    print("\n" + "=" * 70)
    print("For full Stage 1: python run_stage1.py")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
