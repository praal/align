import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import ActionId, Environment, Observation, State

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
]


OBJECTS = dict([(v, k) for k, v in enumerate(
    ["wood", "iron", "axe", "factory", "gem"])])  # , "wall"])])


def update_facts(facts: Sequence[bool], objects: Observation, is_iron) -> Set[int]:
    state = set([i for i, v in enumerate(facts) if v])
    for o in objects:
        # if o != "gold" and o != "gem" and o in OBJECTS:
        #     state.add(OBJECTS[o])
        # elif o == "gem" and OBJECTS["axe"] in state:
        #     state.add(OBJECTS[o])
        #     state.remove(OBJECTS["axe"])
        if o == "iron" and OBJECTS["iron"] in state:
            state.remove(OBJECTS[o])

        elif o == "iron" and is_iron:
            state.add(OBJECTS[o])
        elif o != "iron" and o!= "gem" and o in OBJECTS:
            state.add(OBJECTS[o])
        elif o == "gem" and OBJECTS["axe"] in state:
            state.add(OBJECTS[o])

    if "factory" in objects:
        if OBJECTS["wood"] in state and OBJECTS["iron"] in state:
            if OBJECTS["axe"] not in state:
                state.add(OBJECTS["axe"])


    return state


class CraftState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]

    def __init__(self, x: int, y: int, facts: Set[int], is_iron=True):
        self.x = x
        self.y = y

        fact_list = [False] * len(OBJECTS)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.uid = (self.x, self.y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(rng: Random,
               map_data: Sequence[Sequence[Observation]]) -> 'CraftState':
        # return CraftState(5, 5, set())
        while True:
            y = rng.randrange(len(map_data))
            x = rng.randrange(len(map_data[0]))
            if "wall" not in map_data[y][x]:
                return CraftState(x, y, update_facts((), map_data[y][x], True))


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
    'X': frozenset(["wall"]),
    'a': frozenset(["wood"]),
    'b': frozenset(["toolshed"]),
    'c': frozenset(["workbench"]),
    'd': frozenset(["grass"]),
    'e': frozenset(["factory"]),
    'f': frozenset(["iron"]),
    'g': frozenset(["gold"]),
    'h': frozenset(["gem"]),
    ' ': frozenset(),
    }


def load_map(map_fn: str) -> Tuple[Tuple[Observation, ...], ...]:
    with open(map_fn) as map_file:
        array = []
        for l in map_file:
            if len(l.rstrip()) == 0:
                continue

            row = []
            for cell in l.rstrip():
                row.append(MAPPING[cell])
            array.append(tuple(row))

    return tuple(array)


class Craft(Environment[CraftState]):
    map_data: Tuple[Tuple[Observation, ...], ...]
    num_actions = 4

    def __init__(self, map_fn: str, rng: Random):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        super().__init__(CraftState.random(self.rng, self.map_data))

    def apply_action(self, a: ActionId, is_iron):
        x, y = self.state.x + ACTIONS[a][0], self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x]:
            return

        objects = self.map_data[y][x]
        new_facts = update_facts(self.state.facts, objects, is_iron)

        self.state = CraftState(x, y, new_facts, is_iron)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: CraftState, a: ActionId, s1: CraftState) -> float:
        return 1.0

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[CraftState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = CraftState.random(self.rng, self.map_data)

    @staticmethod
    def label(state: CraftState) -> FrozenSet[int]:
        return frozenset([i for i in range(len(OBJECTS)) if state.facts[i]])
