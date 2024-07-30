from typing import List, Dict, Tuple, Any, Union
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field


class Symbol(Enum):
    CREATE_TORUS = auto()
    CONNECT_TOROIDS = auto()
    START_BRANCH = auto()
    END_BRANCH = auto()
    TRANSFORM = auto()

    @property
    def parameter(self):
        return np.random.randint(0, 100)


@dataclass
class Rule:
    predecessor: Symbol
    successor: List[Any]
    weight: float = 1.0  # Rule probability weight


@dataclass
class NetworkGeneratorConfig:
    max_depth: int = 5
    balance_factor: float = 0.5  # Higher means less branching
    branching_factor: int = 2  # Max number of branches
    create_torus_weights: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    connect_toroids_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])


class NetworkGenerator:
    def __init__(self, config: NetworkGeneratorConfig = NetworkGeneratorConfig()):
        self.config = config
        self.axiom: List[Symbol] = [Symbol.CREATE_TORUS]
        self.rules: Dict[Symbol, List[Rule]] = {
            Symbol.CREATE_TORUS: [
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS, Symbol.CONNECT_TOROIDS],
                     weight=self.config.create_torus_weights[0]),
                Rule(Symbol.CREATE_TORUS, [[Symbol.CREATE_TORUS, Symbol.TRANSFORM], [Symbol.CREATE_TORUS]],
                     weight=self.config.create_torus_weights[1]),
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS], weight=self.config.create_torus_weights[2])
            ],
            Symbol.CONNECT_TOROIDS: [
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.CONNECT_TOROIDS, Symbol.TRANSFORM],
                     weight=self.config.connect_toroids_weights[0]),
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.TRANSFORM], weight=self.config.connect_toroids_weights[1])
            ]
        }

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
                print("  " * depth + "⎡ BRANCH:")
                NetworkGenerator.render_sequence(item, depth + 1)
                print("  " * depth + "⎣ END BRANCH")


if __name__ == "__main__":
    config = NetworkGeneratorConfig(max_depth=5, balance_factor=0.7, branching_factor=2,
                                    create_torus_weights=[0.6, 0.3, 0.1], connect_toroids_weights=[0.7, 0.3])
    generator = NetworkGenerator(config)
    result = generator.generate()  # No iteration count
    print("Tree-based L-System result:")
    generator.render_sequence(result)