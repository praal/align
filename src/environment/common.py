from typing import Iterable, Tuple, Union

from .craft import Craft, CraftState
from .environment import ActionId, RewardFn
from .farm import Farm, FarmState
from .office import Office, OfficeState

CraftOffice = Union[Craft, Farm, Office]
CraftOfficeState = Union[CraftState, FarmState, OfficeState]


class ReachFacts(RewardFn):
    target: Tuple[int, ...]

    def __init__(self, environment: CraftOffice, facts: Iterable[int], notfacts: Iterable[int]):
        super().__init__(environment)
        self.target = tuple(facts)
        self.nottargets = tuple(notfacts)

    def __call__(self, s0: CraftOfficeState, a: ActionId,
                 s1: CraftOfficeState) -> Tuple[float, bool]:
        cost = self.environment.cost(s0, a, s1)
        for fact in self.target:
            if not s1.facts[fact]:
                return -cost, False
        for fact in self.nottargets:
            if s1.facts[fact]:
                return -cost, False

        return -cost, True

    def reset(self):
        pass
