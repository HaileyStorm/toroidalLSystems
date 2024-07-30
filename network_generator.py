import numpy as np
from enum import Enum, auto
from typing import List, Dict, NamedTuple
from dataclasses import dataclass


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
    successor: List[Symbol]
    probability: float = 1.0

    def __post_init__(self):
        assert 0 <= self.probability <= 1, "Probability must be between 0 and 1"


class NetworkGenerator:
    def __init__(self):
        self.axiom: List[Symbol] = [Symbol.CREATE_TORUS]
        self.rules: Dict[Symbol, List[Rule]] = {
            Symbol.CREATE_TORUS: [
                Rule(Symbol.CREATE_TORUS, [Symbol.CREATE_TORUS, Symbol.CONNECT_TOROIDS], 0.7),
                Rule(Symbol.CREATE_TORUS, [Symbol.START_BRANCH, Symbol.CREATE_TORUS, Symbol.END_BRANCH], 0.3)
            ],
            Symbol.CONNECT_TOROIDS: [
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.CONNECT_TOROIDS, Symbol.TRANSFORM], 0.6),
                Rule(Symbol.CONNECT_TOROIDS, [Symbol.TRANSFORM], 0.4)
            ]
        }
        self._normalize_probabilities()

    def select_rule(self, symbol: Symbol) -> Rule:
        candidates = self.rules.get(symbol, [])
        if not candidates:
            return Rule(symbol, [symbol])  # Identity rule

        probabilities = [rule.probability for rule in candidates]
        normalized_probabilities = [p / sum(probabilities) for p in probabilities]
        return np.random.choice(candidates, p=normalized_probabilities)

    def generate(self, iterations: int = 3) -> List[List[Symbol]]:
        result = [self.axiom.copy()]

        for _ in range(iterations):
            new_sequence = []
            for branch in result:
                new_branch = []
                i = 0
                while i < len(branch):
                    symbol = branch[i]
                    if symbol == Symbol.START_BRANCH:
                        sub_branch, i = self._generate_sub_branch(branch, i + 1)
                        new_branch.append(sub_branch)
                    else:
                        rule = self.select_rule(symbol)
                        new_branch.extend(rule.successor)
                    i += 1
                new_sequence.append(new_branch)
            result = new_sequence

        return result[0]  # Return the main branch for now

    def _generate_sub_branch(self, sequence, start):
        depth = 1
        end = start
        while depth > 0:
            if sequence[end] == Symbol.START_BRANCH:
                depth += 1
            elif sequence[end] == Symbol.END_BRANCH:
                depth -= 1
            end += 1
        return self.generate(1)[0], end  # Recursively generate sub-branch

    @staticmethod
    def interpret_sequence(sequence: List[Symbol]) -> str:
        """Debug helper: text-interpretation of generated sequence"""
        message = []
        indent = 0
        for i, symbol in enumerate(sequence):
            if symbol == Symbol.START_BRANCH:
                indent += 1
            elif symbol == Symbol.END_BRANCH:
                indent -= 1

            msg = f"{'  ' * indent}{symbol.name}(P={symbol.parameter})"
            message.append(msg)

        return "\n".join(message)

    def _normalize_probabilities(self):
        for symbol, rules in self.rules.items():
            total_prob = sum(rule.probability for rule in rules)
            if total_prob != 1:
                print(f"Warning: Total probability for {symbol} is {total_prob}, normalizing...")
                factor = 1 / total_prob
                for rule in rules:
                    rule.probability *= factor


if __name__ == "__main__":
    #np.random.seed(1137)  # For reproducibility
    generator = NetworkGenerator()
    sequence = generator.generate(iterations=5)

    print(f"Generated {len(sequence)} symbols!")
    print("L-system sequence interpretation:")
    print(generator.interpret_sequence(sequence))