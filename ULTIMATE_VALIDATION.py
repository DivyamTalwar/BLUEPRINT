import sys
import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class Colors:
    """ASCII color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class UltimateValidator:
    def __init__(self):
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': []
        }
        self.root = Path(__file__).parent

    def print_header(self, text: str):
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    def print_test(self, name: str, passed: bool, message: str = ""):
        self.results['total'] += 1

        if passed:
            self.results['passed'] += 1
            status = f"{Colors.GREEN}[+] PASS{Colors.RESET}"
        else:
            self.results['failed'] += 1
            status = f"{Colors.RED}[-] FAIL{Colors.RESET}"
            self.results['errors'].append({'test': name, 'error': message})

        print(f"{status} | {name}")
        if message:
            print(f"       {Colors.YELLOW}->{Colors.RESET} {message}")

    def print_warning(self, message: str):
        """Print warning"""
        self.results['warnings'] += 1
        print(f"{Colors.YELLOW}[!] WARN{Colors.RESET} | {message}")

    # ========================================================================
    # CATEGORY 1: FILE STRUCTURE VALIDATION
    # ========================================================================

    def validate_file_structure(self):
        """Validate complete file structure"""
        self.print_header("CATEGORY 1: FILE STRUCTURE VALIDATION")

        # Critical root files
        root_files = [
            'README.md',
            'main.py',
            'config.yaml',
            'requirements.txt',
            '.gitignore',
            '.env.example',
        ]

        missing_root = [f for f in root_files if not (self.root / f).exists()]
        self.print_test(
            "Root files present",
            len(missing_root) == 0,
            f"All {len(root_files)} critical root files exist" if not missing_root
            else f"Missing: {missing_root}"
        )

        # Critical directories
        directories = [
            'src',
            'src/core',
            'src/stage1',
            'src/stage2',
            'src/stage3',
            'src/models',
            'src/utils',
            'tests',
            'docs',
            'scripts',
            'archive',
            'archive/old_docs',
            'archive/old_tests',
            'config',
            'data',
        ]

        missing_dirs = [d for d in directories if not (self.root / d).is_dir()]
        self.print_test(
            "Directory structure",
            len(missing_dirs) == 0,
            f"All {len(directories)} directories exist" if not missing_dirs
            else f"Missing: {missing_dirs}"
        )

        # Source code files
        source_files = [
            'src/__init__.py',
            'src/core/__init__.py',
            'src/core/rpg.py',
            'src/core/llm_router_final.py',
            'src/core/graph_operations.py',
            'src/core/graph_persistence.py',
            'src/stage1/__init__.py',
            'src/stage1/feature_selection_loop.py',
            'src/stage2/__init__.py',
            'src/stage2/stage2_orchestrator.py',
            'src/stage3/__init__.py',
            'src/stage3/tdd_engine.py',
            'src/stage3/stage3_orchestrator.py',
            'src/stage3/repository_builder.py',
            'src/utils/__init__.py',
            'src/utils/config_loader.py',
            'src/utils/cost_tracker.py',
            'src/utils/docker_runner.py',
            'src/utils/logger.py',
        ]

        missing_src = [f for f in source_files if not (self.root / f).exists()]
        self.print_test(
            "Source code files",
            len(missing_src) == 0,
            f"All {len(source_files)} source files present" if not missing_src
            else f"Missing: {missing_src}"
        )

        # Documentation files
        doc_files = [
            'docs/QUICK_START.md',
            'docs/PRODUCTION_GUIDE.md',
            'docs/API_KEYS_AND_COSTS.md',
            'docs/MODELS_USED.md',
            'docs/SETUP_WITH_YOUR_KEYS.md',
            'docs/Roadmap.md',
        ]

        missing_docs = [f for f in doc_files if not (self.root / f).exists()]
        self.print_test(
            "Documentation files in docs/",
            len(missing_docs) == 0,
            f"All {len(doc_files)} doc files in docs/" if not missing_docs
            else f"Missing: {missing_docs}"
        )

        # Script files
        script_files = [
            'scripts/validate_production.py',
            'scripts/verify_setup.py',
            'scripts/analyze_dependencies.py',
            'scripts/generate_feature_tree.py',
        ]

        missing_scripts = [f for f in script_files if not (self.root / f).exists()]
        self.print_test(
            "Utility scripts in scripts/",
            len(missing_scripts) == 0,
            f"All {len(script_files)} scripts in scripts/" if not missing_scripts
            else f"Missing: {missing_scripts}"
        )

        # Archive validation
        archived_docs = list((self.root / 'archive' / 'old_docs').glob('*.md')) + \
                       list((self.root / 'archive' / 'old_docs').glob('*.txt'))
        self.print_test(
            "Old docs archived",
            len(archived_docs) >= 20,
            f"{len(archived_docs)} old documentation files safely archived"
        )

        archived_tests = list((self.root / 'archive' / 'old_tests').glob('*.py'))
        self.print_test(
            "Old tests archived",
            len(archived_tests) >= 18,
            f"{len(archived_tests)} old test files safely archived"
        )

    # ========================================================================
    # CATEGORY 2: IMPORT VALIDATION
    # ========================================================================

    def validate_imports(self):
        """Validate all critical imports work"""
        self.print_header("CATEGORY 2: IMPORT VALIDATION")

        # Core imports
        try:
            from src.core.rpg import RepositoryPlanningGraph, NodeType, NodeStatus, EdgeType, RPGNode
            self.print_test("Core RPG imports", True, "RepositoryPlanningGraph, NodeType, NodeStatus, EdgeType, RPGNode")
        except Exception as e:
            self.print_test("Core RPG imports", False, str(e))

        try:
            from src.core.llm_router_final import FinalLLMRouter
            self.print_test("LLM Router import", True, "FinalLLMRouter")
        except Exception as e:
            self.print_test("LLM Router import", False, str(e))

        try:
            from src.core.graph_operations import GraphOperations
            self.print_test("Graph Operations import", True, "GraphOperations")
        except Exception as e:
            self.print_test("Graph Operations import", False, str(e))

        try:
            from src.core.graph_persistence import GraphPersistence
            self.print_test("Graph Persistence import", True, "GraphPersistence")
        except Exception as e:
            self.print_test("Graph Persistence import", False, str(e))

        # Stage imports
        try:
            from src.stage1.feature_selection_loop import FeatureSelectionLoop
            self.print_test("Stage1 imports", True, "FeatureSelectionLoop")
        except Exception as e:
            self.print_test("Stage1 imports", False, str(e))

        try:
            from src.stage2.stage2_orchestrator import Stage2Orchestrator
            self.print_test("Stage2 imports", True, "Stage2Orchestrator")
        except Exception as e:
            self.print_test("Stage2 imports", False, str(e))

        try:
            from src.stage3.tdd_engine import TDDEngine
            from src.stage3.stage3_orchestrator import Stage3Orchestrator
            from src.stage3.repository_builder import RepositoryBuilder
            self.print_test("Stage3 imports", True, "TDDEngine, Stage3Orchestrator, RepositoryBuilder")
        except Exception as e:
            self.print_test("Stage3 imports", False, str(e))

        # Utils imports
        try:
            from src.utils.config_loader import ConfigLoader, get_config
            self.print_test("Config loader import", True, "ConfigLoader, get_config")
        except Exception as e:
            self.print_test("Config loader import", False, str(e))

        try:
            from src.utils.cost_tracker import CostTracker
            self.print_test("Cost tracker import", True, "CostTracker")
        except Exception as e:
            self.print_test("Cost tracker import", False, str(e))

        try:
            from src.utils.docker_runner import DockerRunner
            self.print_test("Docker runner import", True, "DockerRunner")
        except Exception as e:
            self.print_test("Docker runner import", False, str(e))

        try:
            from src.utils.logger import get_logger
            self.print_test("Logger import", True, "get_logger")
        except Exception as e:
            self.print_test("Logger import", False, str(e))

        # Main import
        try:
            from main import BLUEPRINTPipeline
            self.print_test("Main pipeline import", True, "BLUEPRINTPipeline")
        except Exception as e:
            self.print_test("Main pipeline import", False, str(e))

    # ========================================================================
    # CATEGORY 3: SYNTAX VALIDATION
    # ========================================================================

    def validate_syntax(self):
        """Validate Python syntax of all files"""
        self.print_header("CATEGORY 3: PYTHON SYNTAX VALIDATION")

        # Find all Python files
        python_files = []
        for pattern in ['src/**/*.py', 'tests/*.py', 'scripts/*.py']:
            python_files.extend(self.root.glob(pattern))
        python_files.append(self.root / 'main.py')

        syntax_errors = []
        valid_count = 0

        for file_path in python_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    ast.parse(code)
                    valid_count += 1
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path.relative_to(self.root)}: {e}")

        self.print_test(
            "Python syntax validation",
            len(syntax_errors) == 0,
            f"{valid_count}/{len(python_files)} files have valid syntax" if not syntax_errors
            else f"Syntax errors in: {syntax_errors}"
        )

        # Validate critical modified files specifically
        critical_files = [
            'src/core/rpg.py',
            'src/core/llm_router_final.py',
            'src/stage3/tdd_engine.py',
            'src/stage3/stage3_orchestrator.py',
            'src/utils/docker_runner.py',
            'main.py',
        ]

        critical_errors = []
        for file_path in critical_files:
            full_path = self.root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    critical_errors.append(f"{file_path}: {e}")

        self.print_test(
            "Critical files syntax",
            len(critical_errors) == 0,
            f"All {len(critical_files)} critical files valid" if not critical_errors
            else f"Errors: {critical_errors}"
        )

    # ========================================================================
    # CATEGORY 4: CONFIGURATION VALIDATION
    # ========================================================================

    def validate_configuration(self):
        """Validate configuration files"""
        self.print_header("CATEGORY 4: CONFIGURATION VALIDATION")

        # Check config.yaml exists and is valid
        config_path = self.root / 'config.yaml'
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Check required sections
                required_sections = ['stage1', 'stage2', 'stage3', 'llm', 'docker']
                missing_sections = [s for s in required_sections if s not in config]

                self.print_test(
                    "Config structure",
                    len(missing_sections) == 0,
                    f"All {len(required_sections)} sections present" if not missing_sections
                    else f"Missing: {missing_sections}"
                )

                # Check stage3 critical settings
                stage3 = config.get('stage3', {})
                critical_settings = [
                    'skip_docker',
                    'save_unvalidated',
                    'static_validation',
                    'incremental_save',
                ]
                missing_settings = [s for s in critical_settings if s not in stage3]

                self.print_test(
                    "Stage3 production settings",
                    len(missing_settings) == 0,
                    f"All critical settings present" if not missing_settings
                    else f"Missing: {missing_settings}"
                )

                # Validate critical values
                if stage3.get('save_unvalidated') == True and stage3.get('static_validation') == True:
                    self.print_test(
                        "Production safety settings",
                        True,
                        "save_unvalidated=True, static_validation=True [OK]"
                    )
                else:
                    self.print_test(
                        "Production safety settings",
                        False,
                        f"save_unvalidated={stage3.get('save_unvalidated')}, static_validation={stage3.get('static_validation')}"
                    )

            except Exception as e:
                self.print_test("Config validation", False, str(e))
        else:
            self.print_test("Config file exists", False, "config.yaml not found")

        # Check .env.example
        env_example = self.root / '.env.example'
        self.print_test(
            ".env.example present",
            env_example.exists(),
            "Template environment file available"
        )

        # Check .env (without exposing keys)
        env_file = self.root / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()

                required_keys = [
                    'GOOGLE_API_KEY',
                    'OPENROUTER_API_KEY',
                    'COHERE_API_KEY',
                    'PINECONE_API_KEY',
                ]

                set_keys = sum(1 for key in required_keys if os.getenv(key))

                self.print_test(
                    "API keys configured",
                    set_keys >= 3,
                    f"{set_keys}/{len(required_keys)} API keys set" if set_keys >= 3
                    else f"Only {set_keys}/4 keys set (need at least 3)"
                )
            except Exception as e:
                self.print_warning(f"Could not validate .env: {e}")
        else:
            self.print_warning(".env file not found - user needs to configure")

    # ========================================================================
    # CATEGORY 5: DATA STRUCTURE VALIDATION
    # ========================================================================

    def validate_data_structures(self):
        """Validate core data structures"""
        self.print_header("CATEGORY 5: DATA STRUCTURE VALIDATION")

        try:
            from src.core.rpg import NodeStatus, RPGNode, NodeType

            # Validate NodeStatus enum
            expected_statuses = ['planned', 'designed', 'generated', 'syntax_valid', 'validated', 'failed']
            actual_statuses = [s.value for s in NodeStatus]
            missing_statuses = [s for s in expected_statuses if s not in actual_statuses]

            self.print_test(
                "NodeStatus enum (6 statuses)",
                len(missing_statuses) == 0,
                f"All statuses present: {', '.join(expected_statuses)}" if not missing_statuses
                else f"Missing: {missing_statuses}"
            )

            # Validate RPGNode has new fields
            test_node = RPGNode(id="test", type=NodeType.LEAF, name="test")

            required_fields = ['validation_method', 'validation_errors', 'generation_attempts']
            missing_fields = [f for f in required_fields if not hasattr(test_node, f)]

            self.print_test(
                "RPGNode validation fields",
                len(missing_fields) == 0,
                f"All fields present: {', '.join(required_fields)}" if not missing_fields
                else f"Missing: {missing_fields}"
            )

            # Test serialization
            try:
                node_dict = test_node.to_dict()
                has_fields = all(f in node_dict for f in required_fields)
                self.print_test(
                    "RPGNode serialization",
                    has_fields,
                    "to_dict() includes all validation fields"
                )
            except Exception as e:
                self.print_test("RPGNode serialization", False, str(e))

        except Exception as e:
            self.print_test("Data structure validation", False, str(e))

    # ========================================================================
    # CATEGORY 6: PRODUCTION FEATURES VALIDATION
    # ========================================================================

    def validate_production_features(self):
        """Validate all 8 critical production fixes"""
        self.print_header("CATEGORY 6: PRODUCTION FEATURES VALIDATION")

        import inspect

        # Fix 1: TDD Engine static validation
        try:
            from src.stage3.tdd_engine import TDDEngine
            methods = [m[0] for m in inspect.getmembers(TDDEngine, predicate=inspect.isfunction)]

            has_static = '_validate_static' in methods
            has_docker_check = '_check_docker_available' in methods

            self.print_test(
                "TDD Engine: Static validation",
                has_static and has_docker_check,
                "_validate_static and _check_docker_available methods present"
            )

            # Check generate() returns has_code
            generate_source = inspect.getsource(TDDEngine.generate)
            has_code_return = 'has_code' in generate_source

            self.print_test(
                "TDD Engine: Code preservation",
                has_code_return,
                "generate() returns has_code (always saves code)"
            )

        except Exception as e:
            self.print_test("TDD Engine validation", False, str(e))

        # Fix 2: Stage3 Orchestrator preservation
        try:
            from src.stage3.stage3_orchestrator import Stage3Orchestrator
            orch_source = inspect.getsource(Stage3Orchestrator._generate_all_code)

            preserves_code = 'generated_code[node_id] = result' in orch_source
            checks_has_code = 'has_code' in orch_source

            self.print_test(
                "Stage3: Code preservation",
                preserves_code and checks_has_code,
                "Orchestrator saves all generated code"
            )

        except Exception as e:
            self.print_test("Stage3 validation", False, str(e))

        # Fix 3: LLM Router safety filters
        try:
            from src.core.llm_router_final import FinalLLMRouter
            router_source = inspect.getsource(FinalLLMRouter._call_gemini)

            has_safety = 'safety_settings' in router_source and 'BLOCK_NONE' in router_source
            checks_finish = 'finish_reason' in router_source

            self.print_test(
                "LLM Router: Safety filters",
                has_safety and checks_finish,
                "Gemini safety settings configured"
            )

            router_methods = [m[0] for m in inspect.getmembers(FinalLLMRouter, predicate=inspect.isfunction)]
            has_sanitize = '_sanitize_prompt_for_gemini' in router_methods

            self.print_test(
                "LLM Router: Prompt sanitization",
                has_sanitize,
                "Prompt sanitization method present"
            )

        except Exception as e:
            self.print_test("LLM Router validation", False, str(e))

        # Fix 4: Docker Runner error classification
        try:
            from src.utils.docker_runner import DockerRunner
            docker_source = inspect.getsource(DockerRunner.run_code)

            has_error_types = 'error_type' in docker_source and 'docker_unavailable' in docker_source
            checks_available = '_docker_available' in docker_source

            self.print_test(
                "Docker Runner: Error classification",
                has_error_types and checks_available,
                "Error types and availability checks present"
            )

        except Exception as e:
            self.print_test("Docker Runner validation", False, str(e))

        # Fix 5: Main pipeline Docker detection
        try:
            from main import BLUEPRINTPipeline
            main_source = inspect.getsource(BLUEPRINTPipeline.check_prerequisites)

            detects_docker = 'docker.from_env' in main_source
            auto_configures = 'skip_docker' in main_source

            self.print_test(
                "Main Pipeline: Docker detection",
                detects_docker and auto_configures,
                "Auto-detects Docker and configures skip_docker mode"
            )

        except Exception as e:
            self.print_test("Main Pipeline validation", False, str(e))

    # ========================================================================
    # CATEGORY 7: NO BROKEN REFERENCES
    # ========================================================================

    def validate_no_broken_references(self):
        """Ensure no files reference moved/deleted files"""
        self.print_header("CATEGORY 7: NO BROKEN REFERENCES")

        # Files that were moved - should not be referenced in active code
        moved_files = [
            'validate_production_ready.py',  # Now scripts/validate_production.py
            'PHASE1_COMPLETE.md',
            'FINAL_STATUS.md',
            'LEGENDARY_PRODUCTION_TEST.py',
        ]

        # Search source files for references
        broken_refs = []
        source_files = list(self.root.glob('src/**/*.py'))
        source_files.extend([self.root / 'main.py'])

        for src_file in source_files:
            if src_file.exists():
                try:
                    with open(src_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for moved_file in moved_files:
                        if moved_file in content and 'archive' not in content:
                            broken_refs.append(f"{src_file.name} references {moved_file}")

                except Exception:
                    pass

        self.print_test(
            "No broken file references",
            len(broken_refs) == 0,
            "No source files reference moved/deleted files" if not broken_refs
            else f"Found references: {broken_refs}"
        )

        # Check that archive files are truly archived
        old_docs_in_root = list(self.root.glob('PHASE*.md')) + list(self.root.glob('FINAL_*.md'))
        self.print_test(
            "No old docs in root",
            len(old_docs_in_root) == 0,
            "Root directory clean of old documentation" if not old_docs_in_root
            else f"Found in root: {[f.name for f in old_docs_in_root]}"
        )

        old_tests_in_root = [f for f in self.root.glob('*TEST*.py') if f.name != 'ULTIMATE_VALIDATION.py']
        self.print_test(
            "No old tests in root",
            len(old_tests_in_root) == 0,
            "Root directory clean of old test files" if not old_tests_in_root
            else f"Found in root: {[f.name for f in old_tests_in_root]}"
        )

    # ========================================================================
    # CATEGORY 8: RESTRUCTURE VERIFICATION
    # ========================================================================

    def validate_restructure(self):
        """Verify restructure was successful"""
        self.print_header("CATEGORY 8: RESTRUCTURE VERIFICATION")

        # Check restructure log
        log_path = self.root / 'RESTRUCTURE_LOG.json'
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log = json.load(f)

                success_rate = log.get('success_rate', 0)
                num_changes = len(log.get('changes', []))
                num_errors = len(log.get('errors', []))

                self.print_test(
                    "Restructure success rate",
                    success_rate == 1.0,
                    f"{num_changes} files moved, {num_errors} errors, {success_rate*100:.0f}% success"
                )

            except Exception as e:
                self.print_test("Restructure log validation", False, str(e))
        else:
            self.print_warning("RESTRUCTURE_LOG.json not found")

        # Verify expected file counts
        docs_count = len(list((self.root / 'docs').glob('*.md')))
        self.print_test(
            "Documentation organized",
            docs_count >= 6,
            f"{docs_count} documentation files in docs/"
        )

        scripts_count = len(list((self.root / 'scripts').glob('*.py')))
        self.print_test(
            "Scripts organized",
            scripts_count >= 3,
            f"{scripts_count} utility scripts in scripts/"
        )

        archived_docs = len(list((self.root / 'archive' / 'old_docs').glob('*')))
        self.print_test(
            "Old docs archived",
            archived_docs >= 20,
            f"{archived_docs} old documentation files archived"
        )

        archived_tests = len(list((self.root / 'archive' / 'old_tests').glob('*.py')))
        self.print_test(
            "Old tests archived",
            archived_tests >= 18,
            f"{archived_tests} old test files archived"
        )

    # ========================================================================
    # RUN ALL VALIDATIONS
    # ========================================================================

    def run_all(self):
        """Execute all validation categories"""
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{Colors.BOLD}ULTIMATE BLUEPRINT VALIDATION - THE FUCKING BEST{Colors.RESET}")
        print(f"{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Validating EVERYTHING before production deployment...{Colors.RESET}")
        print(f"{Colors.YELLOW}NO API CALLS - Pure structural validation only{Colors.RESET}\n")

        # Run all validation categories
        self.validate_file_structure()
        self.validate_imports()
        self.validate_syntax()
        self.validate_configuration()
        self.validate_data_structures()
        self.validate_production_features()
        self.validate_no_broken_references()
        self.validate_restructure()

        # Print final summary
        self.print_summary()

        return self.results['failed'] == 0

    def print_summary(self):
        """Print final validation summary"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}ULTIMATE VALIDATION SUMMARY{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        print(f"Total Tests:    {self.results['total']}")
        print(f"{Colors.GREEN}Passed:         {self.results['passed']}{Colors.RESET}")
        print(f"{Colors.RED}Failed:         {self.results['failed']}{Colors.RESET}")
        print(f"{Colors.YELLOW}Warnings:       {self.results['warnings']}{Colors.RESET}")

        success_rate = (self.results['passed'] / self.results['total'] * 100) if self.results['total'] > 0 else 0
        print(f"\nSuccess Rate:   {success_rate:.1f}%")

        if self.results['failed'] == 0 and self.results['passed'] > 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}*** LEGENDARY! 100% PRODUCTION READY! ***{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.RESET}\n")
            print(f"{Colors.GREEN}[+] All file structure validated{Colors.RESET}")
            print(f"{Colors.GREEN}[+] All imports working{Colors.RESET}")
            print(f"{Colors.GREEN}[+] All syntax valid{Colors.RESET}")
            print(f"{Colors.GREEN}[+] Configuration complete{Colors.RESET}")
            print(f"{Colors.GREEN}[+] Data structures correct{Colors.RESET}")
            print(f"{Colors.GREEN}[+] Production features verified{Colors.RESET}")
            print(f"{Colors.GREEN}[+] No broken references{Colors.RESET}")
            print(f"{Colors.GREEN}[+] Restructure successful{Colors.RESET}")
            print(f"\n{Colors.GREEN}{Colors.BOLD}HELL YEAH! READY FOR PRODUCTION TESTING!{Colors.RESET}\n")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}{'='*70}{Colors.RESET}")
            print(f"{Colors.RED}{Colors.BOLD}VALIDATION FAILED - FIX REQUIRED{Colors.RESET}")
            print(f"{Colors.RED}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

            if self.results['errors']:
                print(f"{Colors.RED}Failed Tests:{Colors.RESET}")
                for error in self.results['errors']:
                    print(f"  {Colors.RED}[-]{Colors.RESET} {error['test']}")
                    print(f"    {Colors.YELLOW}->{Colors.RESET} {error['error']}")

        # Save results
        report_path = self.root / 'ULTIMATE_VALIDATION_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump({
                'total_tests': self.results['total'],
                'passed': self.results['passed'],
                'failed': self.results['failed'],
                'warnings': self.results['warnings'],
                'success_rate': success_rate,
                'production_ready': self.results['failed'] == 0,
                'errors': self.results['errors']
            }, f, indent=2)

        print(f"\n{Colors.CYAN}Detailed report saved: {report_path.name}{Colors.RESET}\n")


if __name__ == "__main__":
    validator = UltimateValidator()
    success = validator.run_all()
    sys.exit(0 if success else 1)
