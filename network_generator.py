import random
from typing import List, Dict, Tuple, Any, Union
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field


# Symbols for our L-system, representing network construction operations
class Symbol(Enum):
    CREATE_TORUS = auto()
    CONNECT_TOROIDS = auto()  # Does this need more than one paramter to work with for us to define inter-toroid connections? E.g. do we need a mask or range of indices that get copied into the destination, or....??????)
    START_BRANCH = auto()
    END_BRANCH = auto()
    TRANSFORM = auto()

    @property
    def parameter(self):
        # Each symbol can have an associated random parameter
        # This allows for more varied networks in pathway/network construction
        return np.random.randint(0, 100)


@dataclass
class Rule:
    predecessor: Symbol
    successor: List[Any]
    weight: float = 1.0  # Rule probability weight


@dataclass
class NetworkGeneratorConfig:
    # Configuration for the L-system network generator
    # This allows us to control network complexity and characteristics
    max_depth: int = 5  # Maximum depth of the network tree
    balance_factor: float = 0.5  # Controls network complexity. Higher values (approaching 1.0) produce more linear networks, lower values produce more branching
    branching_factor: int = 2  # Maximum number of branches at each branching point
    create_torus_weights: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    connect_toroids_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    seed: int = None  # Optional seed for consistent network generation, critical for repeatable maze generation


class NetworkGenerator:
    def __init__(self, config: NetworkGeneratorConfig = NetworkGeneratorConfig()):
        self.config = config
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        # Here we define the grammar of our L-system
        # These rules will dictate how our network "grows" during generation
        self.axiom: List[Symbol] = [Symbol.CREATE_TORUS]
        self.rules: Dict[Symbol, List[Rule]] = {
            Symbol.CREATE_TORUS: [
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS, Symbol.CONNECT_TOROIDS], weight=self.config.create_torus_weights[0]),
                Rule(Symbol.CREATE_TORUS, [[Symbol.CREATE_TORUS, Symbol.TRANSFORM], [Symbol.CREATE_TORUS]], weight=self.config.create_torus_weights[1]),
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS], weight=self.config.create_torus_weights[2])
            ],
            Symbol.CONNECT_TOROIDS: [
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.CONNECT_TOROIDS, Symbol.TRANSFORM], weight=self.config.connect_toroids_weights[0]),
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.TRANSFORM], weight=self.config.connect_toroids_weights[1])
            ]
        }

        # Re-normalize in case non-default values were provided
        self.config.create_torus_weights = self.normalize_weights(self.config.create_torus_weights, "create_torus_weights")
        self.config.connect_toroids_weights = self.normalize_weights(self.config.connect_toroids_weights, "connect_toroids_weights")

    @staticmethod
    def normalize_weights(weights: List[float], name: str) -> List[float]:
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            print(f"Warning: {name} add up to {total:.4f}, not 1.0. Auto-normalizing.")
            return [w / total for w in weights]
        return weights

    def select_rule(self, symbol: Symbol) -> Rule:
        candidates = self.rules.get(symbol, [])
        if not candidates:
            return Rule(symbol, [symbol])

        weights = [rule.weight for rule in candidates]
        normalized_weights = [w/sum(weights) for w in weights]
        return np.random.choice(candidates, p=normalized_weights)

    def produce(self, sequence: Union[List[Any], Symbol], depth: int) -> List[Any]:
        if depth <= 0:
            return [sequence] if isinstance(sequence, Symbol) else sequence

        # Convert single Symbol to list for uniform processing
        sequence = [sequence] if isinstance(sequence, Symbol) else sequence

        result = []
        for item in sequence:
            if isinstance(item, Symbol):
                rule = self.select_rule(item)
                successor = rule.successor if isinstance(rule.successor, list) else [rule.successor]
                result.extend(self.produce(successor, depth - 1))
            elif isinstance(item, list) and np.random.rand() > self.config.balance_factor:
                if len(item) >= self.config.branching_factor:
                    branch_count = self.config.branching_factor
                    branches = [self.produce(branch, depth - 1) for branch in
                                np.random.choice(item, branch_count, replace=False)]
                else:
                    branches = [self.produce(branch, depth - 1) for branch in item]
                result.append(branches)
            else:
                result.append(item)

        return result

    def generate(self) -> List[Any]:
        return self.produce(self.axiom, depth=self.config.max_depth)

    @staticmethod
    def render_sequence(sequence: List[Any], depth: int = 0):
        for item in sequence:
            if isinstance(item, Symbol):
                print("  " * depth + f"{item.name}(p={item.parameter})")
            elif isinstance(item, list):
                print("  " * depth + "\\ BRANCH:")  # ⎡
                NetworkGenerator.render_sequence(item, depth + 1)
                print("  " * depth + "/ END BRANCH")  # ⎣


if __name__ == "__main__":
    config = NetworkGeneratorConfig(max_depth=5, balance_factor=0.7, branching_factor=2,
                                    create_torus_weights=[0.6, 0.3, 0.1], connect_toroids_weights=[0.7, 0.3])
    generator = NetworkGenerator(config)
    result = generator.generate()  # Generate the "recipe" for our toroidal network
    print("Tree-based L-System result:")
    generator.render_sequence(result)


"""
New Symbols: INPUT and OUTPUT
Represent data entry/exit points
Connect to n-dimensional slices of tori (n ≤ torus dimensions)
Placement rules vary based on loop configuration
At least one of each required; configurable maximum

Connection Rules
Enforce TORUS-to-TORUS and TRANSFORM-to-TORUS connections (CONNECT_TOROIDS)
INPUT connects to TORUS; OUTPUT from TORUS or TRANSFORM
No connection required TORUS-to-TRANSFORM (TRANSFORM is a mutation of TORUS, its basically a TORUS *and* a mutation rule... its torus takes on the transformed values of the original)

Post-processing
Prune empty branches and unnecessary depth

TRANSFORM Symbol
Parameter determines transform type
Interpretation handled at implementation level

Loop Functionality
Configurable enable/disable option
Directional connections when enabled
Various structures allowed (within/across branches and depths)
Prevent orphaned flow areas
All loops require both "input" and "output" (can be symbols or connections)

Dimensionality and Size Constraints
INPUT/OUTPUT dimensions must be compatible with connected TORUS
Sizes limited by TORUS dimensions
"""