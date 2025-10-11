import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.core.llm_router import LLMRouter


def test_api_call():
    print("Testing LLM Router with actual API call...")
    print("=" * 60)

    try:
        # Load config
        config = get_config()

        # Check API keys
        env_check = config.validate_env_vars()
        missing = [k for k, v in env_check.items() if not v]

        if missing:
            print("⚠️  Missing API keys:")
            for key in missing:
                print(f"   - {key}")
            print("\nPlease add your API keys to .env file")
            return False

        # Initialize router
        router = LLMRouter(config.get_all())
        print("✓ LLM Router initialized")

        # Test simple call
        prompt = "Say 'Hello from BLUEPRINT!' and nothing else."
        print(f"\nSending test prompt: {prompt}")
        print("Calling random LLM provider...")

        response = router.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=50
        )

        print("\n" + "=" * 60)
        print("RESPONSE:")
        print(f"Provider: {response.provider.value}")
        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Latency: {response.latency:.2f}s")
        print("=" * 60)

        # Get stats
        stats = router.get_stats()
        print("\nRouter Statistics:")
        print(f"Total calls: {stats['api_calls']}")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Total tokens: {stats['total_tokens']}")

        print("\n✅ API test successful!")
        return True

    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_api_call()
    sys.exit(0 if success else 1)
