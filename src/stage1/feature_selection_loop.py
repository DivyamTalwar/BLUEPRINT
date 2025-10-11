from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

from src.stage1.user_input_processor import RepositoryRequest
from src.stage1.exploit_strategy import ExploitStrategy
from src.stage1.explore_strategy import ExploreStrategy
from src.models.feature import Feature
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SelectionIteration:
    """Single iteration of feature selection"""

    iteration: int
    exploit_ratio: float
    explore_ratio: float
    exploit_features: List[Feature] = field(default_factory=list)
    explore_features: List[Feature] = field(default_factory=list)
    total_selected: int = 0
    visited_domains: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureSelectionLoop:
    """30-iteration feature selection loop with exploit-explore"""

    def __init__(
        self,
        exploit_strategy: ExploitStrategy,
        explore_strategy: ExploreStrategy,
        num_iterations: int = 30,
    ):
        """
        Initialize selection loop

        Args:
            exploit_strategy: Exploit strategy
            explore_strategy: Explore strategy
            num_iterations: Number of iterations (default: 30)
        """
        self.exploit_strategy = exploit_strategy
        self.explore_strategy = explore_strategy
        self.num_iterations = num_iterations

        # Track state
        self.selected_features: List[Feature] = []
        self.visited_domains: Set[str] = set()
        self.iterations: List[SelectionIteration] = []
        self.seen_feature_ids: Set[str] = set()

    def run(
        self,
        request: RepositoryRequest,
        target_features: int = 150,
    ) -> List[Feature]:
        """
        Run the complete feature selection loop

        Args:
            request: Repository request
            target_features: Target number of features (default: 150)

        Returns:
            Selected features
        """
        logger.info(
            "Starting %d-iteration feature selection (target: %d features)",
            self.num_iterations,
            target_features,
        )

        features_per_iteration = target_features // self.num_iterations

        for i in range(self.num_iterations):
            iteration_num = i + 1

            # Get exploit-explore ratio for this iteration
            exploit_ratio, explore_ratio = self._get_ratio(iteration_num)

            # Calculate features to select
            num_exploit = int(features_per_iteration * exploit_ratio)
            num_explore = int(features_per_iteration * explore_ratio)

            logger.info(
                "Iteration %d/%d: exploit=%.2f (%d), explore=%.2f (%d)",
                iteration_num,
                self.num_iterations,
                exploit_ratio,
                num_exploit,
                explore_ratio,
                num_explore,
            )

            # Run iteration
            iteration = self._run_iteration(
                iteration_num=iteration_num,
                request=request,
                num_exploit=num_exploit,
                num_explore=num_explore,
                exploit_ratio=exploit_ratio,
                explore_ratio=explore_ratio,
            )

            self.iterations.append(iteration)

            # Early stopping if target reached
            if len(self.selected_features) >= target_features:
                logger.info("Target features reached at iteration %d", iteration_num)
                break

        logger.info(
            "Selection complete: %d features selected, %d domains visited",
            len(self.selected_features),
            len(self.visited_domains),
        )

        return self.selected_features

    def _run_iteration(
        self,
        iteration_num: int,
        request: RepositoryRequest,
        num_exploit: int,
        num_explore: int,
        exploit_ratio: float,
        explore_ratio: float,
    ) -> SelectionIteration:
        """Run single iteration"""

        iteration = SelectionIteration(
            iteration=iteration_num,
            exploit_ratio=exploit_ratio,
            explore_ratio=explore_ratio,
        )

        # EXPLOIT: Focused retrieval
        if num_exploit > 0:
            exploit_features = self._exploit_phase(request, num_exploit)
            iteration.exploit_features = exploit_features

            # Add to selected (deduplicated)
            self._add_features(exploit_features)

        # EXPLORE: Diverse discovery
        if num_explore > 0:
            explore_features = self._explore_phase(request, num_explore)
            iteration.explore_features = explore_features

            # Add to selected (deduplicated)
            self._add_features(explore_features)

        # Update iteration stats
        iteration.total_selected = len(self.selected_features)
        iteration.visited_domains = self.visited_domains.copy()

        logger.info(
            "Iteration %d: +%d exploit, +%d explore, total=%d",
            iteration_num,
            len(iteration.exploit_features),
            len(iteration.explore_features),
            iteration.total_selected,
        )

        return iteration

    def _exploit_phase(
        self,
        request: RepositoryRequest,
        num_features: int,
    ) -> List[Feature]:
        """Run exploit phase"""

        # Strategy varies by iteration phase
        iteration_num = len(self.iterations) + 1

        if iteration_num <= 5:
            # Early: Focus on must-have features
            features = self.exploit_strategy.retrieve_must_have_features(
                request=request,
                top_k=num_features,
            )

        elif iteration_num <= 15:
            # Mid: Focus on explicit requirements
            features = self.exploit_strategy.retrieve_for_requirements(
                requirements=request.explicit_requirements,
                top_k_per_requirement=max(1, num_features // len(request.explicit_requirements))
                if request.explicit_requirements else 1,
            )

        else:
            # Late: General focused retrieval
            features = self.exploit_strategy.retrieve_focused_features(
                request=request,
                top_k=num_features,
                use_domain_filter=False,  # Broaden search
            )

        # Track visited domains
        for feature in features:
            self.visited_domains.add(feature.domain)

        return features[:num_features]

    def _explore_phase(
        self,
        request: RepositoryRequest,
        num_features: int,
    ) -> List[Feature]:
        """Run explore phase"""

        iteration_num = len(self.iterations) + 1

        if iteration_num <= 10:
            # Early: Discover from unvisited domains
            features = self.explore_strategy.discover_diverse_features(
                request=request,
                visited_domains=self.visited_domains,
                top_k=num_features,
            )

        elif iteration_num <= 20:
            # Mid: Suggest complementary features
            features = self.explore_strategy.suggest_complementary_features(
                request=request,
                selected_features=self.selected_features,
                top_k=num_features,
            )

        else:
            # Late: Cross-domain features
            features = self.explore_strategy.discover_cross_domain_features(
                request=request,
                primary_domain=request.primary_domain,
                top_k=num_features,
            )

        # Track visited domains
        for feature in features:
            self.visited_domains.add(feature.domain)

        return features[:num_features]

    def _add_features(self, features: List[Feature]):
        """Add features with deduplication"""
        for feature in features:
            if feature.id not in self.seen_feature_ids:
                self.selected_features.append(feature)
                self.seen_feature_ids.add(feature.id)

    def _get_ratio(self, iteration: int) -> Tuple[float, float]:
        """
        Get exploit-explore ratio for iteration

        Early iterations: More exploit (focused)
        Mid iterations: Balanced
        Late iterations: More explore (diverse)

        Args:
            iteration: Iteration number (1-based)

        Returns:
            Tuple of (exploit_ratio, explore_ratio)
        """
        # Define phases
        if iteration <= 10:
            # Early: 70% exploit, 30% explore
            exploit_ratio = 0.7
            explore_ratio = 0.3

        elif iteration <= 20:
            # Mid: 50% exploit, 50% explore
            exploit_ratio = 0.5
            explore_ratio = 0.5

        else:
            # Late: 30% exploit, 70% explore
            exploit_ratio = 0.3
            explore_ratio = 0.7

        return exploit_ratio, explore_ratio

    def get_stats(self) -> Dict[str, Any]:
        """Get selection statistics"""

        # Features by domain
        by_domain = {}
        for feature in self.selected_features:
            domain = feature.domain
            if domain not in by_domain:
                by_domain[domain] = 0
            by_domain[domain] += 1

        # Features by complexity
        by_complexity = {}
        for feature in self.selected_features:
            complexity = feature.complexity.value
            if complexity not in by_complexity:
                by_complexity[complexity] = 0
            by_complexity[complexity] += 1

        # Iteration stats
        exploit_total = sum(len(it.exploit_features) for it in self.iterations)
        explore_total = sum(len(it.explore_features) for it in self.iterations)

        return {
            "total_features": len(self.selected_features),
            "total_iterations": len(self.iterations),
            "domains_visited": len(self.visited_domains),
            "features_by_domain": by_domain,
            "features_by_complexity": by_complexity,
            "exploit_total": exploit_total,
            "explore_total": explore_total,
            "avg_features_per_iteration": (
                len(self.selected_features) / len(self.iterations)
                if self.iterations else 0
            ),
        }

    def export_selection_log(self) -> List[Dict[str, Any]]:
        """Export detailed selection log"""
        log = []

        for iteration in self.iterations:
            log.append({
                "iteration": iteration.iteration,
                "exploit_ratio": iteration.exploit_ratio,
                "explore_ratio": iteration.explore_ratio,
                "exploit_features": [f.name for f in iteration.exploit_features],
                "explore_features": [f.name for f in iteration.explore_features],
                "total_selected": iteration.total_selected,
                "domains_visited": list(iteration.visited_domains),
            })

        return log
