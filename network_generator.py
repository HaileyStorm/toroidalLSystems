from typing import List, Dict, Tuple, Any
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
    successor: List[Any]  # Can be Symbol or List[Symbol] for branching
    probability: float = 1.0

    def __post_init__(self):
        assert 0 <= self.probability <= 1, f"Invalid probability: {self.probability}"


class NetworkGenerator:
    def __init__(self):
        self.axiom: List[Symbol] = [Symbol.CREATE_TORUS]
        self.rules: Dict[Symbol, List[Rule]] = {
            Symbol.CREATE_TORUS: [
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS, Symbol.CONNECT_TOROIDS], 0.6),
                Rule(Symbol.CREATE_TORUS, [[Symbol.CREATE_TORUS, Symbol.TRANSFORM], [Symbol.CREATE_TORUS]], 0.3),
                # Multiple branches
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS], 0.1)  # Identity-like rule
            ],
            Symbol.CONNECT_TOROIDS: [
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.CONNECT_TOROIDS, Symbol.TRANSFORM], 0.7),
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.TRANSFORM], 0.3)
            ]
        }

    def select_rule(self, symbol: Symbol) -> Rule:
        candidates = self.rules.get(symbol, [])
        if not candidates:
            return Rule(symbol, [symbol])  # Identity rule

        probabilities = [rule.probability for rule in candidates]
        normalized_probabilities = [p / sum(probabilities) for p in probabilities]
        return np.random.choice(candidates, p=normalized_probabilities)

    def produce(self, sequence: List[Any], depth: int = 3) -> List[Any]:
        if depth <= 0:
            return sequence

        result = []
        for item in sequence:
            if isinstance(item, Symbol):
                rule = self.select_rule(item)
                result.extend(self.produce(rule.successor, depth - 1))
            elif isinstance(item, list):
                # If we get a list, we have a branch: produce it recursively
                result.append(self.produce(item, depth - 1))
            else:
                result.append(item)

        return result

    def generate(self, iterations: int = 3) -> List[Any]:
        return self.produce(self.axiom, depth=iterations)

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
    generator = NetworkGenerator()
    result = generator.generate(iterations=5)
    print("Tree-based L-System result:")
    generator.render_sequence(result)