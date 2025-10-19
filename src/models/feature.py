from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ComplexityLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Feature:
    """Represents a single feature in the feature tree"""

    id: str
    domain: str
    subdomain: str
    name: str
    description: str
    complexity: ComplexityLevel
    keywords: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity.value,
            "keywords": self.keywords,
            "use_cases": self.use_cases,
            "dependencies": self.dependencies,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        required_fields = ["id", "domain", "subdomain", "name", "description", "complexity"]
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"Missing required field: {field}")

        try:
            complexity = ComplexityLevel(data["complexity"])
        except ValueError:
            raise ValueError(f"Invalid complexity level: {data['complexity']}. Must be one of: {[c.value for c in ComplexityLevel]}")

        return cls(
            id=data["id"],
            domain=data["domain"],
            subdomain=data["subdomain"],
            name=data["name"],
            description=data["description"],
            complexity=complexity,
            keywords=data.get("keywords", []),
            use_cases=data.get("use_cases", []),
            dependencies=data.get("dependencies", []),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )

    def get_search_text(self) -> str:
        """Get text for embedding/search"""
        parts = [
            f"Domain: {self.domain}",
            f"Subdomain: {self.subdomain}",
            f"Feature: {self.name}",
            f"Description: {self.description}",
            f"Keywords: {', '.join(self.keywords)}",
            f"Complexity: {self.complexity.value}",
        ]
        return " | ".join(parts)


@dataclass
class Domain:
    """Represents a domain in the taxonomy"""

    name: str
    description: str
    subdomains: List[str]
    keywords: List[str] = field(default_factory=list)
    typical_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "subdomains": self.subdomains,
            "keywords": self.keywords,
            "typical_features": self.typical_features,
        }
