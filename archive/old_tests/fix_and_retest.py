import yaml
config = {
    'stage1': {
        'iterations': 30,
        'top_k': 10,
        'exploit_ratio_early': 0.7,
        'exploit_ratio_mid': 0.5,
        'exploit_ratio_late': 0.3,
        'target_features': None,
        'save_checkpoints': True
    },
    'stage2': {
        'signature_batch_size': 5,
        'create_base_classes': True,
        'validate_before_stage3': True
    },
    'stage3': {
        'max_debug_attempts': 8,
        'test_timeout': 60,
        'checkpoint_interval': 10
    },
    'llm': {
        'default_temperature': 0.7,
        'max_tokens': 4000,
        'retry_attempts': 3,
        'retry_delay': 2
    },
    'docker': {
        'timeout': 300,
        'memory_limit': "1g",
        'cpu_limit': 1.0,
        'base_image': "python:3.11-slim"
    },
    'vector_db': {
        'index_name': "BLUEPRINT-features",
        'dimension': 1024,
        'metric': "cosine",
        'environment': "us-east-1"
    },
    'feature_tree': {
        'total_features': 5000,
        'features_per_subdomain': 25,
        'batch_size': 96
    },
    'logging': {
        'level': "INFO",
        'format': "json",
        'output_file': "logs/BLUEPRINT.log"
    },
    'performance': {
        'parallel_requests': 5,
        'cache_embeddings': True,
        'cache_llm_responses': False
    },
    'cost': {
        'track_costs': True,
        'warn_threshold': 50.0,
        'stop_threshold': 100.0
    }
}

# Write updated config
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Updated config.yaml with all required keys")

# Now run tests
print("\n🔥 Running comprehensive tests...\n")
import subprocess
result = subprocess.run(['python', 'comprehensive_test.py'], capture_output=False)
exit(result.returncode)
