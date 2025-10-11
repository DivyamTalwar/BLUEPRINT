import json
from typing import Dict, Any, Optional, Tuple
from src.core.rpg import RepositoryPlanningGraph
from src.core.llm_router_final import FinalLLMRouter
from src.utils.logger import StructuredLogger

logger = StructuredLogger("tdd_engine")


class TDDEngine:

    def __init__(self, llm_router: FinalLLMRouter, docker_runner, config: Dict[str, Any]):
        self.llm = llm_router
        self.docker = docker_runner
        self.config = config
        self.logger = logger

        self.max_fix_attempts = config.get("max_debug_attempts", 8)
        self.skip_docker = config.get("skip_docker", False)
        self.save_unvalidated = config.get("save_unvalidated", True)
        self.static_validation = config.get("static_validation", True)

        # Check Docker availability once
        self._docker_available = self._check_docker_available()

    def generate(self, rpg: RepositoryPlanningGraph, node_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate code for a single node using TDD.

        Args:
            rpg: Complete RPG with signatures
            node_id: Node to generate

        Returns:
            (has_code, result) where result contains:
                - test_code: Generated test
                - implementation: Generated code
                - test_output: Test execution output
                - attempts: Number of attempts taken
                - status: "validated", "generated", "syntax_valid", or "failed"
                - validation_method: "docker", "static", or "none"
                - errors: List of validation errors
        """
        node_data = rpg.graph.nodes[node_id]

        self.logger.info(f"Generating code for: {node_data.get('name')}")

        # Step 1: Generate test
        test_code = self._generate_test(node_data, rpg)

        if not test_code:
            return False, {
                "error": "Failed to generate test",
                "status": "failed",
                "validation_method": "none"
            }

        # Step 2: Generate initial implementation
        implementation = self._generate_implementation(node_data, rpg, test_code)

        if not implementation:
            return False, {
                "error": "Failed to generate implementation",
                "status": "failed",
                "validation_method": "none"
            }

        # CRITICAL: Code is generated - we WILL return it regardless of test results
        result = {
            "test_code": test_code,
            "implementation": implementation,
            "test_output": "",
            "attempts": 0,
            "status": "generated",  # Default to "generated" status
            "validation_method": "none",
            "errors": []
        }

        # Step 3: Validate code (Docker TDD loop OR static validation)
        if self.skip_docker or not self._docker_available:
            # Use static validation instead of Docker
            if self.static_validation:
                validation_result = self._validate_static(implementation, test_code, node_data)
                result["status"] = validation_result["status"]
                result["validation_method"] = "static"
                result["errors"] = validation_result["errors"]
                self.logger.info(f"Static validation: {result['status']}")
            else:
                self.logger.warning("Skipping validation - code generated but not validated")
                result["status"] = "generated"
                result["validation_method"] = "none"
        else:
            # Run TDD loop with Docker
            tdd_success, final_impl, test_output, attempts = self._tdd_loop(
                node_data=node_data,
                test_code=test_code,
                initial_implementation=implementation,
                rpg=rpg
            )

            result["implementation"] = final_impl
            result["test_output"] = test_output
            result["attempts"] = attempts
            result["status"] = "validated" if tdd_success else "generated"
            result["validation_method"] = "docker"

        # Always return True if code was generated (even if not validated)
        has_code = bool(result["implementation"])
        return has_code, result

    def _generate_test(self, node_data: Dict[str, Any], rpg: RepositoryPlanningGraph) -> Optional[str]:
        """
        Generate pytest test from function signature and docstring.
        """
        signature = node_data.get("signature")
        docstring = node_data.get("docstring", "")
        name = node_data.get("name")
        functionality = node_data.get("functionality", "")

        if not signature:
            self.logger.error(f"No signature found for {name}")
            return None

        prompt = f"""Write a comprehensive pytest test for this function.

Function signature:
{signature}

Docstring:
{docstring}

Functionality:
{functionality}

Requirements:
1. Test the main functionality
2. Test edge cases
3. Test error handling
4. Use realistic test data
5. Include assertions with clear messages

Output ONLY the test code, no explanations.
Format:
```python
import pytest
# other imports as needed

def test_{name}():
    # Test implementation
    pass

def test_{name}_edge_cases():
    # Edge case tests
    pass
```

Generate the test code:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000
            )

            # Extract code from response
            test_code = self._extract_code_block(response.content)

            self.logger.debug(f"Test generated for {name}", lines=len(test_code.split('\n')))
            return test_code

        except Exception as e:
            self.logger.error(f"Failed to generate test for {name}", error=str(e))
            return None

    def _generate_implementation(self, node_data: Dict[str, Any], rpg: RepositoryPlanningGraph,
                                 test_code: str = "") -> Optional[str]:
        """
        Generate implementation code to pass the test.
        """
        signature = node_data.get("signature")
        docstring = node_data.get("docstring", "")
        name = node_data.get("name")
        functionality = node_data.get("functionality", "")
        code_type = node_data.get("code_type")

        # Get base class if inherits
        inherits_from = node_data.get("inherits_from")
        base_class_code = ""
        if inherits_from:
            base_class_code = self._get_base_class_code(rpg, inherits_from)

        prompt = f"""Implement this function/method to pass the test.

Signature:
{signature}

Docstring:
{docstring}

Functionality:
{functionality}

Test code:
{test_code[:1000]}...

{"Base class to inherit from:\n" + base_class_code if base_class_code else ""}

Requirements:
1. Implement the function to pass the test
2. Handle edge cases properly
3. Include error handling
4. Use type hints
5. Follow Python best practices

Output ONLY the implementation code, no explanations.

Generate the implementation:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=3000
            )

            impl_code = self._extract_code_block(response.content)

            self.logger.debug(f"Implementation generated for {name}", lines=len(impl_code.split('\n')))
            return impl_code

        except Exception as e:
            self.logger.error(f"Failed to generate implementation for {name}", error=str(e))
            return None

    def _tdd_loop(self, node_data: Dict, test_code: str, initial_implementation: str,
                  rpg: RepositoryPlanningGraph) -> Tuple[bool, str, str, int]:
        """
        Main TDD loop: test → analyze → fix → retest.

        Returns:
            (success, final_implementation, test_output, attempts)
        """
        current_impl = initial_implementation
        attempts = 0

        while attempts < self.max_fix_attempts:
            attempts += 1

            # Run test
            test_passed, test_output = self._run_test(
                implementation=current_impl,
                test_code=test_code,
                node_data=node_data
            )

            if test_passed:
                self.logger.info(f"[OK] Test passed on attempt {attempts}")
                return True, current_impl, test_output, attempts

            # Test failed - analyze and fix
            self.logger.warning(f"[FAIL] Test failed on attempt {attempts}")

            if attempts >= self.max_fix_attempts:
                self.logger.error("Max fix attempts reached")
                return False, current_impl, test_output, attempts

            # Generate fix
            fixed_impl = self._generate_fix(
                current_implementation=current_impl,
                test_code=test_code,
                error_output=test_output,
                node_data=node_data
            )

            if not fixed_impl or fixed_impl == current_impl:
                self.logger.warning("Fix generation failed or no change")
                return False, current_impl, test_output, attempts

            current_impl = fixed_impl

        return False, current_impl, test_output, attempts

    def _run_test(self, implementation: str, test_code: str, node_data: Dict) -> Tuple[bool, str]:
        """
        Run test via Docker sandbox.

        Returns:
            (test_passed, output)
        """
        try:
            # Combine implementation and test into single file
            full_code = f"""{implementation}

{test_code}
"""

            # Run via Docker
            result = self.docker.run_code(
                code=full_code,
                test_mode=True,
                timeout=self.config.get("test_timeout", 60)
            )

            passed = result.get("success", False) and "PASSED" in result.get("output", "")
            output = result.get("output", "") + "\n" + result.get("error", "")

            return passed, output

        except Exception as e:
            self.logger.error("Failed to run test", error=str(e))
            return False, str(e)

    def _generate_fix(self, current_implementation: str, test_code: str,
                     error_output: str, node_data: Dict) -> Optional[str]:
        """
        Analyze error and generate fixed implementation.
        """
        name = node_data.get("name")

        prompt = f"""This code failed its test. Analyze the error and provide a fix.

Current implementation:
{current_implementation}

Test code:
{test_code[:800]}...

Error output:
{error_output[:1000]}

Analyze the error and provide corrected implementation.
Output ONLY the fixed code, no explanations.

Fixed implementation:"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=3000
            )

            fixed_code = self._extract_code_block(response.content)

            self.logger.debug(f"Fix generated for {name}")
            return fixed_code

        except Exception as e:
            self.logger.error(f"Failed to generate fix for {name}", error=str(e))
            return None

    def _get_base_class_code(self, rpg: RepositoryPlanningGraph, class_name: str) -> str:
        """Get base class code if available."""
        for node_id, node_data in rpg.graph.nodes(data=True):
            if (node_data.get("is_base_class") and
                node_data.get("name") == class_name):
                # Return the design as code
                design = node_data.get("base_class_design", {})
                return self._format_base_class(design)
        return ""

    def _format_base_class(self, design: Dict) -> str:
        """Format base class design as code."""
        class_name = design.get("class_name", "BaseClass")
        docstring = design.get("docstring", "")
        methods = design.get("methods", [])

        code = f"class {class_name}:\n"
        code += f'    """{docstring}"""\n\n'

        for method in methods:
            sig = method.get("signature", "")
            doc = method.get("docstring", "")
            is_abstract = method.get("is_abstract", False)

            code += f"    {sig}:\n"
            code += f'        """{doc}"""\n'

            if is_abstract:
                code += "        raise NotImplementedError\n"
            else:
                code += "        pass\n"

            code += "\n"

        return code

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        # Remove markdown code fences
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()

        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()

        # No code block found, return as-is
        return text.strip()

    def _check_docker_available(self) -> bool:
        """
        Check if Docker is available for test execution.

        Returns:
            True if Docker is available and working, False otherwise
        """
        if self.skip_docker:
            return False

        try:
            if self.docker is None:
                return False

            # Try to ping Docker
            import docker
            client = docker.from_env()
            client.ping()
            return True

        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            return False

    def _validate_static(self, implementation: str, test_code: str, node_data: Dict) -> Dict[str, Any]:
        """
        Perform static validation on generated code (without executing).

        Checks:
        1. Python syntax validity
        2. Import availability
        3. Basic structure (has functions/classes)
        4. Type hint usage

        Returns:
            dict with "status" and "errors"
        """
        import ast
        import sys

        errors = []
        name = node_data.get("name", "unknown")

        # 1. Syntax check
        try:
            ast.parse(implementation)
            self.logger.debug(f"[OK] Syntax valid for {name}")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return {"status": "failed", "errors": errors}

        # 2. Check structure - should have at least one function or class
        try:
            tree = ast.parse(implementation)
            has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            has_class = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))

            if not has_function and not has_class:
                errors.append("No functions or classes defined")

        except Exception as e:
            errors.append(f"Structure check failed: {e}")

        # 3. Extract and check imports (basic check only)
        try:
            tree = ast.parse(implementation)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check if import is in stdlib or common packages
                        if alias.name not in sys.modules and alias.name not in [
                            "pytest", "typing", "dataclasses", "enum", "json",
                            "os", "sys", "pathlib", "datetime", "time"
                        ]:
                            errors.append(f"Potentially missing dependency: {alias.name}")

        except Exception as e:
            errors.append(f"Import check failed: {e}")

        # 4. Check test code syntax too
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"Test syntax error: {e}")

        # Determine status
        if len(errors) == 0:
            status = "syntax_valid"
            self.logger.info(f"[OK] Static validation passed for {name}")
        elif any("Syntax error" in err for err in errors):
            status = "failed"
            self.logger.error(f"[FAIL] Syntax errors in {name}")
        else:
            # Minor issues, but code is syntactically valid
            status = "syntax_valid"
            self.logger.warning(f"[WARNING] Minor validation issues for {name}")

        return {
            "status": status,
            "errors": errors
        }
