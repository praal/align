import logging
from random import Random
from typing import Iterable, FrozenSet, List, Optional, Set, Tuple

from .environment import ActionId, Environment, Observation, State

WIDTH = 12
HEIGHT = 9

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # up
    (0, -1),  # down
    (-1, 0),  # left
    (1, 0),   # right
]

WALLS = set([
    ((2, 0), (3, 0)), ((3, 0), (2, 0)),
    ((5, 0), (6, 0)), ((6, 0), (5, 0)),
    ((8, 0), (9, 0)), ((9, 0), (8, 0)),

    ((2, 2), (3, 2)), ((3, 2), (2, 2)),
    ((5, 2), (6, 2)), ((6, 2), (5, 2)),
    ((8, 2), (9, 2)), ((9, 2), (8, 2)),

    ((2, 3), (3, 3)), ((3, 3), (2, 3)),
    ((5, 3), (6, 3)), ((6, 3), (5, 3)),
    ((8, 3), (9, 3)), ((9, 3), (8, 3)),

    ((2, 4), (3, 4)), ((3, 4), (2, 4)),
    ((5, 4), (6, 4)), ((6, 4), (5, 4)),
    ((8, 4), (9, 4)), ((9, 4), (8, 4)),

    ((2, 5), (3, 5)), ((3, 5), (2, 5)),
    ((5, 5), (6, 5)), ((6, 5), (5, 5)),
    ((8, 5), (9, 5)), ((9, 5), (8, 5)),

    ((2, 6), (3, 6)), ((3, 6), (2, 6)),
    ((5, 6), (6, 6)), ((6, 6), (5, 6)),
    ((8, 6), (9, 6)), ((9, 6), (8, 6)),

    ((2, 8), (3, 8)), ((3, 8), (2, 8)),
    ((5, 8), (6, 8)), ((6, 8), (5, 8)),
    ((8, 8), (9, 8)), ((9, 8), (8, 8)),

    ((0, 2), (0, 3)), ((0, 3), (0, 2)),
    ((0, 5), (0, 6)), ((0, 6), (0, 5)),

    ((2, 2), (2, 3)), ((2, 3), (2, 2)),
    ((2, 5), (2, 6)), ((2, 6), (2, 5)),

    ((3, 2), (3, 3)), ((3, 3), (3, 2)),
    ((3, 5), (3, 6)), ((3, 6), (3, 5)),

    ((4, 2), (4, 3)), ((4, 3), (4, 2)),

    ((5, 2), (5, 3)), ((5, 3), (5, 2)),
    ((5, 5), (5, 6)), ((5, 6), (5, 5)),

    ((6, 2), (6, 3)), ((6, 3), (6, 2)),
    ((6, 5), (6, 6)), ((6, 6), (6, 5)),

    ((7, 2), (7, 3)), ((7, 3), (7, 2)),

    ((8, 2), (8, 3)), ((8, 3), (8, 2)),
    ((8, 5), (8, 6)), ((8, 6), (8, 5)),

    ((9, 2), (9, 3)), ((9, 3), (9, 2)),
    ((9, 5), (9, 6)), ((9, 6), (9, 5)),

    ((11, 2), (11, 3)), ((11, 3), (11, 2)),
    ((11, 5), (11, 6)), ((11, 6), (11, 5)),
])

EMPTY: Observation = frozenset()
OBJECTS = {
    (1, 1):  frozenset('a'),
    (10, 1): frozenset('b'),
    (10, 7): frozenset('c'),
    (1, 7):  frozenset('d'),
    (7, 4):  frozenset('e'),  # mail
    (3, 6):  frozenset('f'),  # coffee
    (8, 2):  frozenset('f'),  # coffee
    (4, 4):  frozenset('g'),  # office
    (4, 1):  frozenset('n'),  # plant
    (7, 1):  frozenset('n'),  # plant
    (4, 7):  frozenset('n'),  # plant
    (7, 7):  frozenset('n'),  # plant
    (1, 4):  frozenset('n'),  # plant
    (10, 4): frozenset('n'),  # plant
}


def update_facts(facts: Tuple[bool, ...],
                 objects: FrozenSet[str]) -> Set[int]:
    fact_indices = set([i for i, v in enumerate(facts) if v])
    if 'a' in objects:
        fact_indices.add(0)
    if 'b' in objects:
        fact_indices.add(1)
    if 'c' in objects:
        fact_indices.add(2)
    if 'd' in objects:
        fact_indices.add(3)
    if 'e' in objects:
        fact_indices.add(4)
    if 'f' in objects:
        fact_indices.add(5)
    if 'g' in objects:
        fact_indices.add(6)
        if facts[4]:
            fact_indices.remove(4)
            fact_indices.add(7)
        if facts[5]:
            fact_indices.remove(5)
            fact_indices.add(8)
    return fact_indices


class OfficeState(State):
    facts: Tuple[bool, ...]

    def __init__(self, x: int, y: int, facts: Iterable[int]):
        self.x = x
        self.y = y

        fact_list = [False] * 9
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.uid = (self.x, self.y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(rng: Random) -> 'OfficeState':
        # return OfficeState(0, 0, [])
        while True:
            x = rng.randrange(WIDTH)
            y = rng.randrange(HEIGHT)
            if (x, y) not in OBJECTS:
                return OfficeState(x, y, [])


class Office(Environment[OfficeState]):
    num_actions = 4

    def __init__(self, rng: Random):
        self.rng = rng
        super().__init__(OfficeState.random(self.rng))

    def apply_action(self, a: ActionId):
        old = (self.state.x, self.state.y)
        x, y = old[0] + ACTIONS[a][0], old[1] + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT or \
                (old, (x, y)) in WALLS:
            return

        objects = OBJECTS.get((x, y), frozenset())
        new_facts = update_facts(self.state.facts, objects)

        self.state = OfficeState(x, y, new_facts)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: OfficeState, a: ActionId, s1: OfficeState) -> float:
        c = 1.0
        if s0 == s1:
            c += 10.0
        if 'n' in self.observe(s1):
            c += 10.0
        return c

    def observe(self, state: OfficeState) -> Observation:
        return OBJECTS.get((state.x, state.y), EMPTY)

    def reset(self, state: Optional[OfficeState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = OfficeState.random(self.rng)

    @staticmethod
    def label(state: OfficeState) -> FrozenSet[int]:
        return frozenset([i for i in range(9) if state.facts[i]])
