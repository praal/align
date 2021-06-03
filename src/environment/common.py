from typing import Iterable, Tuple, Union

from .craft import Craft, CraftState
from .environment import ActionId, RewardFn


CraftOffice = Union[Craft]
CraftOfficeState = Union[CraftState]


class ReachFacts(RewardFn):
    target: Tuple[int, ...]

    def __init__(self, environment: CraftOffice, facts: Iterable[int], notfacts: Iterable[int], problem_mood):
        super().__init__(environment)
        self.target = tuple(facts)
        self.nottargets = tuple(notfacts)
        self.problem_mood = problem_mood


    def __call__(self, s0: CraftOfficeState, a: ActionId,
                 s1: CraftOfficeState) -> Tuple[float, bool]:
        cost = self.environment.cost(s0, a, s1)
        for fact in self.target:
            if not s1.facts[fact]:
                return -cost, False
        for fact in self.nottargets:
            if s1.facts[fact]:
                return -cost, False
        if self.problem_mood == 1:
            if s1.x == 1 and s1.y == 1:
                return -cost, True
            elif s1.x == self.environment.width - 2 and s1.y == self.environment.height - 2:
                return -cost, True
            return -cost, False
        elif self.problem_mood == 2:
            return -cost, True

    def reset(self):
        pass
