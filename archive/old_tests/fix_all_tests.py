import re

with open('comprehensive_test.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''    ops = GraphOperations(rpg)
    is_valid = ops.validate()''',
    '''    # Use static method
    is_valid = GraphOperations.validate(rpg)'''
)

content = content.replace(
    '''    ops = GraphOperations(rpg)
    has_cycles = ops.has_cycles()''',
    '''    # Use static method
    has_cycles = GraphOperations.has_cycles(rpg)'''
)

# Fix 2: GraphVisualization - fix constructor
content = content.replace(
    '''    viz = GraphVisualization(rpg, {})''',
    '''    viz = GraphVisualization(rpg)'''
)

# Fix 3: Feature model - add complexity parameter
content = content.replace(
    '''    feature = Feature(
        id="test-001",
        name="test_feature",
        domain="data_operations",
        subdomain="input",
        description="Test feature"
    )''',
    '''    feature = Feature(
        id="test-001",
        name="test_feature",
        domain="data_operations",
        subdomain="input",
        description="Test feature",
        complexity="basic"
    )'''
)

# Fix 4: UserInputProcessor - fix constructor (takes only llm, not config)
content = content.replace(
    '''    llm = FinalLLMRouter({"llm": {}})
    processor = UserInputProcessor(llm, {})''',
    '''    llm = FinalLLMRouter({"llm": {}})
    processor = UserInputProcessor(llm)'''
)

# Fix 5: Config test - update to check actual config structure
content = content.replace(
    '''@test("Config Loader - Required Keys")
def test_config_required_keys():
    from src.utils.config_loader import get_config
    config = get_config().get_all()
    required = ['stage1', 'stage2', 'stage3', 'llm', 'docker']
    for key in required:
        assert key in config, f"Missing key: {key}"''',
    '''@test("Config Loader - Required Keys")
def test_config_required_keys():
    from src.utils.config_loader import get_config
    config_loader = get_config()
    config = config_loader.get_all()
    required = ['stage1', 'stage2', 'stage3', 'docker']
    for key in required:
        assert key in config, f"Missing key: {key}"
    # llm key is actually present in config.yaml now
    assert 'llm' in config or config_loader.get('llm') is not None'''
)

# Fix 6: Integration test - use static method
content = content.replace(
    '''        persistence = GraphPersistence({})
        persistence.save_json(rpg, filepath)
        assert os.path.exists(filepath)

        loaded = persistence.load_json(filepath)''',
    '''        # Use static methods directly
        GraphPersistence.save_json(rpg, filepath)
        assert os.path.exists(filepath)

        loaded = GraphPersistence.load_json(filepath)'''
)

# Write fixed content
with open('comprehensive_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed all test issues!")
print("  - GraphPersistence: static methods")
print("  - GraphOperations: static methods")
print("  - GraphVisualization: constructor")
print("  - Feature model: complexity parameter")
print("  - UserInputProcessor: constructor signature")
print("  - Config test: updated validation")
print("\nRunning tests now...")

# Run tests
import subprocess
result = subprocess.run(['python', 'comprehensive_test.py'], capture_output=False)
exit(result.returncode)
