from typing import Dict, Any
from jinja2 import Template, Environment, FileSystemLoader
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptTemplates:
    """Manage and render prompt templates"""

    def __init__(self, templates_dir: str = None):
        """
        Initialize prompt templates

        Args:
            templates_dir: Directory containing template files
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent.parent / "prompts"

        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Create Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))

    def render(self, template_name: str, **kwargs) -> str:
        """
        Render template with variables

        Args:
            template_name: Template file name
            **kwargs: Template variables

        Returns:
            Rendered template string
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**kwargs)
            logger.debug("Rendered template: %s", template_name)
            return rendered
        except Exception as e:
            logger.error("Error rendering template %s: %s", template_name, str(e))
            raise

    def render_string(self, template_string: str, **kwargs) -> str:
        """
        Render template from string

        Args:
            template_string: Template string
            **kwargs: Template variables

        Returns:
            Rendered string
        """
        try:
            template = Template(template_string)
            rendered = template.render(**kwargs)
            return rendered
        except Exception as e:
            logger.error("Error rendering template string: %s", str(e))
            raise

    @staticmethod
    def get_stage1_user_input_prompt() -> str:
        """Get prompt for analyzing user input in Stage 1"""
        return """Analyze this repository request and extract structured information.

User Request: {{ user_request }}

Extract the following:
1. Repository type (web/CLI/library/API/ML/data/other)
2. Primary domain and subdomains
3. Explicit requirements (stated by user)
4. Implicit requirements (obviously needed but not stated)
5. Recommended additional features

Output as JSON:
{
  "repo_type": "...",
  "primary_domain": "...",
  "subdomains": [...],
  "explicit_requirements": [...],
  "implicit_requirements": [...],
  "recommended_features": [...]
}
"""

    @staticmethod
    def get_stage1_exploit_filter_prompt() -> str:
        """Get prompt for filtering exploit features"""
        return """Given this repository goal and candidate features, filter for relevance.

Repository Goal: {{ repo_goal }}

Candidate Features (from vector search):
{{ features }}

Task: Select the top {{ top_k }} most essential features for this repository.

Consider:
- Direct relevance to repository goal
- Must-have vs nice-to-have
- Avoid redundancy

Output as JSON:
{
  "selected_features": [
    {
      "feature_id": "...",
      "description": "...",
      "relevance_score": 0.0-1.0,
      "reasoning": "why this is essential"
    }
  ]
}
"""

    @staticmethod
    def get_stage1_explore_suggest_prompt() -> str:
        """Get prompt for exploring diverse features"""
        return """Given this repository goal and already selected features, suggest non-obvious but useful features.

Repository Goal: {{ repo_goal }}

Already Selected Features:
{{ selected_features }}

Visited Domains: {{ visited_domains }}

Task: Suggest {{ top_k }} diverse, complementary features from UNVISITED domains.

Focus on:
- Cross-domain functionality
- Supporting infrastructure
- Quality/testing features
- Non-obvious but valuable additions

Output as JSON:
{
  "suggested_features": [
    {
      "description": "...",
      "domain": "...",
      "reasoning": "why this adds value"
    }
  ]
}
"""

    @staticmethod
    def get_stage1_refactor_prompt() -> str:
        """Get prompt for refactoring features into modules"""
        return """Organize these features into a hierarchical module structure.

Features to Organize:
{{ features }}

Task: Create a modular hierarchy with:
- 4-8 top-level modules
- Sub-modules within each
- Feature assignment to modules

Principles:
- High cohesion (related features together)
- Low coupling (minimal cross-module dependencies)
- Clear responsibilities

Output as JSON:
{
  "modules": [
    {
      "name": "...",
      "description": "...",
      "sub_modules": [
        {
          "name": "...",
          "features": [...]
        }
      ]
    }
  ]
}
"""

    @staticmethod
    def get_stage2_file_structure_prompt() -> str:
        """Get prompt for encoding file structure"""
        return """Map this functionality graph to folder and file structure.

Functionality Graph:
{{ functionality_graph }}

Task: Assign each module to folders/files following best practices for {{ language }}.

Consider:
- Standard project layout
- Domain conventions
- Scalability
- File size targets (~200-500 LOC)

Output as JSON:
{
  "structure": [
    {
      "module": "...",
      "folder": "...",
      "files": [
        {
          "name": "...",
          "contains": [...]
        }
      ]
    }
  ]
}
"""

    @staticmethod
    def get_stage2_data_flow_prompt() -> str:
        """Get prompt for encoding data flows"""
        return """Define data flows between these modules.

Modules:
{{ modules }}

Task: Specify how data flows from module to module.

For each flow, define:
- Source module
- Destination module
- Data type
- Transformation (if any)

Output as JSON:
{
  "flows": [
    {
      "from": "...",
      "to": "...",
      "data": "...",
      "type": "...",
      "transformation": "..."
    }
  ]
}
"""

    @staticmethod
    def get_stage2_base_class_prompt() -> str:
        """Get prompt for creating base classes"""
        return """Identify patterns and create base classes.

Features:
{{ features }}

Task: Find features with similar signatures and extract base classes.

Output as JSON:
{
  "base_classes": [
    {
      "name": "...",
      "description": "...",
      "methods": [
        {
          "name": "...",
          "signature": "..."
        }
      ],
      "used_by": [...]
    }
  ]
}
"""

    @staticmethod
    def get_stage2_signature_prompt() -> str:
        """Get prompt for designing function signatures"""
        return """Design function signature for this feature.

Feature: {{ feature_description }}

Context:
- Input data type: {{ input_type }}
- Output data type: {{ output_type }}
- Related features: {{ related_features }}

Task: Create complete function signature with:
- Parameter types (with type hints)
- Return type
- Docstring (Google style)

Output as JSON:
{
  "signature": "...",
  "docstring": "...",
  "is_class": true/false,
  "class_methods": [...]
}
"""

    @staticmethod
    def get_stage3_test_generation_prompt() -> str:
        """Get prompt for generating tests"""
        return """Write pytest test for this function.

Function Signature:
{{ signature }}

Docstring:
{{ docstring }}

Task: Create comprehensive test covering:
- Basic functionality
- Edge cases
- Error handling

Output as Python code (test function only).
"""

    @staticmethod
    def get_stage3_implementation_prompt() -> str:
        """Get prompt for generating implementation"""
        return """Implement this function to pass the test.

Function Signature:
{{ signature }}

Docstring:
{{ docstring }}

Test Code:
{{ test_code }}

Dependencies Available:
{{ dependencies }}

Task: Write the implementation.

Output as Python code (function/class only).
"""

    @staticmethod
    def get_stage3_error_fix_prompt() -> str:
        """Get prompt for fixing errors"""
        return """Fix this code that failed testing.

Current Code:
{{ current_code }}

Test:
{{ test_code }}

Error:
{{ error }}

Task: Analyze the error and provide corrected code.

Output as JSON:
{
  "error_type": "...",
  "root_cause": "...",
  "fix": "corrected code here"
}
"""

    @staticmethod
    def get_integration_test_prompt() -> str:
        """Get prompt for integration test generation"""
        return """Generate integration test for these components.

Components:
{{ components }}

Data Flow:
{{ data_flow }}

Task: Create integration test that:
1. Tests components together
2. Validates data flow
3. Checks interfaces

Output as Python code (test function).
"""
