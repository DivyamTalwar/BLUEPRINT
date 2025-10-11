import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.core.llm_router import LLMRouter
from src.models.taxonomy import UNIVERSAL_TAXONOMY, get_all_domains
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepositoryRequest:
    """Structured repository request"""

    raw_description: str
    repo_type: str  # web/CLI/library/API/ML/data/other
    primary_domain: str
    subdomains: List[str] = field(default_factory=list)
    explicit_requirements: List[str] = field(default_factory=list)
    implicit_requirements: List[str] = field(default_factory=list)
    recommended_features: List[str] = field(default_factory=list)
    complexity_estimate: str = "intermediate"  # basic/intermediate/advanced/expert
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "raw_description": self.raw_description,
            "repo_type": self.repo_type,
            "primary_domain": self.primary_domain,
            "subdomains": self.subdomains,
            "explicit_requirements": self.explicit_requirements,
            "implicit_requirements": self.implicit_requirements,
            "recommended_features": self.recommended_features,
            "complexity_estimate": self.complexity_estimate,
            "metadata": self.metadata,
        }


class UserInputProcessor:
    """Process and analyze user's repository description"""

    def __init__(self, llm_router: LLMRouter):
        """
        Initialize processor

        Args:
            llm_router: LLM router for API calls
        """
        self.llm_router = llm_router

    def process(self, user_description: str) -> RepositoryRequest:
        """
        Process user input and extract structured information

        Args:
            user_description: User's natural language description

        Returns:
            Structured repository request
        """
        logger.info("Processing user input: %s", user_description[:100])

        # Build prompt
        prompt = self._build_analysis_prompt(user_description)

        # Call LLM
        response = self.llm_router.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temp for more consistent parsing
            max_tokens=2000,
            json_mode=True,
        )

        # Parse response
        try:
            data = json.loads(response.content)

            request = RepositoryRequest(
                raw_description=user_description,
                repo_type=data.get("repo_type", "other"),
                primary_domain=data.get("primary_domain", ""),
                subdomains=data.get("subdomains", []),
                explicit_requirements=data.get("explicit_requirements", []),
                implicit_requirements=data.get("implicit_requirements", []),
                recommended_features=data.get("recommended_features", []),
                complexity_estimate=data.get("complexity_estimate", "intermediate"),
                metadata={
                    "analysis_model": response.model,
                    "analysis_provider": response.provider if isinstance(response.provider, str) else response.provider.value,
                    "analysis_cost": response.cost,
                },
            )

            logger.info(
                "Parsed request: type=%s, domain=%s, %d requirements",
                request.repo_type,
                request.primary_domain,
                len(request.explicit_requirements) + len(request.implicit_requirements),
            )

            return request

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response: %s", str(e))
            # Return basic request
            return RepositoryRequest(
                raw_description=user_description,
                repo_type="other",
                primary_domain="unknown",
            )

    def _build_analysis_prompt(self, user_description: str) -> str:
        """Build analysis prompt"""

        domains_list = "\n".join([f"- {name}: {domain.description}"
                                  for name, domain in list(UNIVERSAL_TAXONOMY.items())[:15]])

        prompt = f"""Analyze this repository request and extract structured information.

User Request:
\"\"\"{user_description}\"\"\"

Available Domains (select most relevant):
{domains_list}

Task: Extract the following information:

1. **Repository Type**: Classify as one of:
   - web: Web application (frontend/backend/full-stack)
   - cli: Command-line tool
   - library: Reusable library/package
   - api: REST/GraphQL API service
   - ml: Machine learning project
   - data: Data processing/analysis
   - other: Other type

2. **Primary Domain**: The MAIN domain from the list above (use exact key like "data_operations", "api_web", etc.)

3. **Subdomains**: Related domains that will also be needed (list of domain keys)

4. **Explicit Requirements**: Features/functionality explicitly mentioned by user

5. **Implicit Requirements**: Features NOT mentioned but obviously needed
   - Think: authentication, error handling, logging, testing, configuration, etc.
   - What does EVERY repository of this type need?

6. **Recommended Features**: Nice-to-have features that enhance the repository

7. **Complexity Estimate**: Overall project complexity (basic/intermediate/advanced/expert)

Output as JSON:
{{
  "repo_type": "web|cli|library|api|ml|data|other",
  "primary_domain": "domain_key",
  "subdomains": ["domain_key1", "domain_key2"],
  "explicit_requirements": ["requirement 1", "requirement 2", ...],
  "implicit_requirements": ["requirement 1", "requirement 2", ...],
  "recommended_features": ["feature 1", "feature 2", ...],
  "complexity_estimate": "basic|intermediate|advanced|expert"
}}

Be thorough with implicit requirements - these are critical for a complete repository!"""

        return prompt

    def enhance_request(self, request: RepositoryRequest) -> RepositoryRequest:
        """
        Enhance request with additional analysis

        Args:
            request: Initial repository request

        Returns:
            Enhanced request
        """
        logger.info("Enhancing repository request")

        # Build enhancement prompt
        prompt = f"""Given this repository request, suggest additional features that would make it production-ready.

Original Request:
- Type: {request.repo_type}
- Domain: {request.primary_domain}
- Explicit Requirements: {', '.join(request.explicit_requirements[:10])}

Already Identified Implicit Requirements:
{', '.join(request.implicit_requirements[:10])}

Task: Suggest 5-10 additional features for production-readiness:
- Testing & quality assurance
- Observability (logging, monitoring)
- Performance optimization
- Security hardening
- Developer experience
- Documentation
- Deployment & operations

Output as JSON:
{{
  "additional_features": ["feature 1", "feature 2", ...],
  "rationale": "Brief explanation of why these are important"
}}"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=1000,
                json_mode=True,
            )

            data = json.loads(response.content)
            additional = data.get("additional_features", [])

            # Add to recommended features
            request.recommended_features.extend(additional)
            request.metadata["enhancement_rationale"] = data.get("rationale", "")

            logger.info("Added %d enhancement features", len(additional))

        except Exception as e:
            logger.error("Enhancement failed: %s", str(e))

        return request

    def validate_request(self, request: RepositoryRequest) -> tuple[bool, List[str]]:
        """
        Validate repository request

        Args:
            request: Repository request

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check basic fields
        if not request.raw_description:
            errors.append("Missing raw description")

        if not request.primary_domain:
            errors.append("No primary domain identified")

        if not request.explicit_requirements and not request.implicit_requirements:
            errors.append("No requirements identified")

        # Check domain validity
        if request.primary_domain not in get_all_domains():
            errors.append(f"Invalid primary domain: {request.primary_domain}")

        for subdomain in request.subdomains:
            if subdomain not in get_all_domains():
                errors.append(f"Invalid subdomain: {subdomain}")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning("Request validation failed: %s", errors)

        return is_valid, errors

    def get_target_feature_count(self, request: RepositoryRequest) -> int:
        """
        Estimate target number of features based on complexity

        Args:
            request: Repository request

        Returns:
            Target feature count
        """
        base_counts = {
            "basic": 50,
            "intermediate": 100,
            "advanced": 150,
            "expert": 200,
        }

        base = base_counts.get(request.complexity_estimate, 100)

        # Adjust based on explicit requirements
        req_count = len(request.explicit_requirements) + len(request.implicit_requirements)
        adjustment = min(50, req_count * 2)  # Each requirement might need ~2 features

        target = base + adjustment

        logger.info(
            "Target feature count: %d (base=%d, adjustment=%d)",
            target,
            base,
            adjustment,
        )

        return target
