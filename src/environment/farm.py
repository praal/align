import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import ActionId, Environment, Observation, State

ACTIONS: List[Tuple[float, float]] = [
    (0.0, 1.0),   # down
    (0.0, -1.0),  # up
    (-1.0, 0.0),  # left
    (1.0, 0.0),   # right
]


OBJECTS = dict([(v, k) for k, v in enumerate(
    ["wood", "grass", "iron", "gold", "gem",  # "toolshed", "workbench",
     "factory", "plank", "rope", "axe", "bow", "stick", "saw", "bed", "shears",
     "cloth", "bridge", "goldware", "ring"])])  # , "wall"])])


def update_facts(facts: Sequence[bool], objects: Observation) -> Set[int]:
    state = set([i for i, v in enumerate(facts) if v])
    for o in objects:
        if o != "gold" and o != "gem" and o in OBJECTS:
            state.add(OBJECTS[o])
        elif o == "gold" and OBJECTS["bridge"] in state:
            state.add(OBJECTS[o])
        elif o == "gem" and OBJECTS["axe"] in state:
            state.add(OBJECTS[o])
    if "toolshed" in objects:
        if OBJECTS["wood"] in state:
            state.add(OBJECTS["plank"])
            state.remove(OBJECTS["wood"])
        if OBJECTS["grass"] in state:
            state.add(OBJECTS["rope"])
            state.remove(OBJECTS["grass"])
        if OBJECTS["stick"] in state and OBJECTS["iron"] in state:
            state.add(OBJECTS["axe"])
            state.remove(OBJECTS["stick"])
            state.remove(OBJECTS["iron"])
        if OBJECTS["rope"] in state and OBJECTS["stick"] in state:
            state.add(OBJECTS["bow"])
            state.remove(OBJECTS["rope"])
            state.remove(OBJECTS["stick"])
    if "workbench" in objects:
        if OBJECTS["wood"] in state:
            state.add(OBJECTS["stick"])
            state.remove(OBJECTS["wood"])
        if OBJECTS["stick"] in state and OBJECTS["iron"] in state:
            state.add(OBJECTS["shears"])
            state.remove(OBJECTS["stick"])
            state.remove(OBJECTS["iron"])
        if OBJECTS["iron"] in state:
            state.add(OBJECTS["saw"])
            state.remove(OBJECTS["iron"])
        if OBJECTS["plank"] in state and OBJECTS["grass"] in state:
            state.add(OBJECTS["bed"])
            state.remove(OBJECTS["plank"])
            state.remove(OBJECTS["grass"])
    if "factory" in objects:
        if OBJECTS["grass"] in state:
            state.add(OBJECTS["cloth"])
            state.remove(OBJECTS["grass"])
        if OBJECTS["gold"] in state:
            state.add(OBJECTS["goldware"])
            state.remove(OBJECTS["gold"])
        if OBJECTS["gem"] in state:
            state.add(OBJECTS["ring"])
            state.remove(OBJECTS["gem"])
        if OBJECTS["iron"] in state and OBJECTS["wood"] in state:
            state.add(OBJECTS["bridge"])
            state.remove(OBJECTS["iron"])
            state.remove(OBJECTS["wood"])

    return state


class FarmState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]

    def __init__(self, x: float, y: float, dx: float, dy: float,
                 facts: Set[int]):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

        fact_list = [False] * len(OBJECTS)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.uid = (self.x, self.y, self.dx, self.dy, self.facts)

    def __str__(self) -> str:
        return "({:2f}, {:2f}, {:2f}, {:2f}, {})".format(self.x, self.y,
                                                         self.dx, self.dy,
                                                         self.facts)

    @staticmethod
    def random(rng: Random,
               map_data: Sequence[Sequence[Observation]]) -> 'FarmState':
        return FarmState(5.5, 5.5, 0.0, 0.0, set())


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
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
            row = []
            for cell in l.rstrip('\n'):
                row.append(MAPPING[cell])
            array.append(tuple(row))

    return tuple(array)


class Farm(Environment[FarmState]):
    map_data: Tuple[Tuple[Observation, ...], ...]
    num_actions = 4

    def __init__(self, map_fn: str, rng: Random):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        super().__init__(FarmState.random(self.rng, self.map_data))

    def apply_action(self, a: ActionId):
        dx = self.state.dx + ACTIONS[a][0] * self.rng.random()
        dy = self.state.dy + ACTIONS[a][1] * self.rng.random()
        logging.debug("applying action %s:%s", a, ACTIONS[a])

        x, y = self.state.x + dx, self.state.y + dy
        x = x % self.width
        y = y % self.height

        objects = self.map_data[int(y) % self.height][int(x) % self.width]
        new_facts = update_facts(self.state.facts, objects)

        self.state = FarmState(x, y, dx, dy, new_facts)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: FarmState, a: ActionId, s1: FarmState) -> float:
        return 1.0

    def observe(self, state: FarmState) -> Observation:
        return self.map_data[int(self.state.y)][int(self.state.x)]

    def reset(self, state: Optional[FarmState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = FarmState.random(self.rng, self.map_data)

    @staticmethod
    def label(state: FarmState) -> FrozenSet[int]:
        return frozenset([i for i in range(len(OBJECTS)) if state.facts[i]])
