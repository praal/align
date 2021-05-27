import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import ActionId, Environment, Observation, State

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
    (0, 0),
]

key_IND = 0

OBJECTS = dict([(v, k) for k, v in enumerate(
    ["key", "wood", "hammer", "box", "extra", "extrawood"])])


def update_facts(facts: Sequence[bool], objects: Observation, is_key,tool_in_fac=False, wood_in_fac = False, put_extra=False):
    state = set([i for i, v in enumerate(facts) if v])
    key_change = 0
    if tool_in_fac:
        state.add(OBJECTS["extra"])

    if wood_in_fac:
        state.add(OBJECTS["extrawood"])
    for o in objects:

        if o == "key":
            if OBJECTS["key"] in state:
                state.remove(OBJECTS["key"])
                key_change = 1

            elif is_key:
                state.add(OBJECTS[o])
                key_change = -1
        elif o in OBJECTS:
            if o == "wood" and put_extra:
                state.add(OBJECTS["extrawood"])
            state.add(OBJECTS[o])

    if "warehouse" in objects:
        if "hammer" in OBJECTS:
            state.add((OBJECTS["hammer"]))
        if put_extra:
            state.add(OBJECTS["extra"])

    if "factory" in objects:
        if OBJECTS["wood"] in state and OBJECTS["key"] in state and OBJECTS["hammer"] in state:
            state.add(OBJECTS["box"])
        if OBJECTS["wood"] in state and OBJECTS["key"] in state and OBJECTS["extra"] in state:
            state.add(OBJECTS["box"])
        if OBJECTS["extrawood"] in state and OBJECTS["key"] in state and OBJECTS["hammer"] in state:
            state.add(OBJECTS["box"])
        if OBJECTS["extrawood"] in state and OBJECTS["key"] in state and OBJECTS["extra"] in state:
            state.add(OBJECTS["box"])

    return state, key_change


class CraftState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]


    def __init__(self, x: int, y: int, key_x, key_y,  facts: Set[int], default_x, default_y):
        self.x = x
        self.y = y
        self.default_x = default_x
        self.default_y = default_y
        fact_list = [False] * len(OBJECTS)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.key_x = key_x
        self.key_y = key_y
        self.uid = (self.x, self.y, key_x, key_y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {:2d}, {:2d}, {})".format(self.x, self.y, self.key_x, self.key_y, self.facts)

    @staticmethod
    def random(rng: Random,
               map_data: Sequence[Sequence[Observation]], key_locations: List[List[int]], default_x, default_y, tool_in_fact, wood_in_fact) -> 'CraftState':

        while True:

            x = default_x
            y = default_y
            ind = rng.randrange(len(key_locations))
            key_y = key_locations[ind][0]
            key_x = key_locations[ind][1]
            if "wall" not in map_data[y][x] and "wall" not in map_data[key_y][key_x]:
                next_tool = False
                next_wood = False
                if tool_in_fact:
                    if rng.random() < 0.5:
                        next_tool = True
                if wood_in_fact:
                    if rng.random() < 0.5:
                        next_wood = True
                facts, _ = update_facts((), map_data[y][x], 0, next_tool, next_wood)
                return CraftState(x, y, key_x, key_y, facts, default_x, default_y)


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
    'X': frozenset(["wall"]),
    'a': frozenset(["wood"]),
    'b': frozenset(["toolshed"]),
    'c': frozenset(["workbench"]),
    'd': frozenset(["grass"]),
    'e': frozenset(["factory"]),
    'f': frozenset(["iron"]),
    'k': frozenset(["key"]),
    'g': frozenset(["gold"]),
    'w': frozenset(["warehouse"]),
    'D': frozenset(["door"]),
    'h': frozenset(["gem"]),
    ' ': frozenset(),
    }


def load_map(map_fn: str):
    with open(map_fn) as map_file:
        array = []
        for l in map_file:
            if len(l.rstrip()) == 0:
                continue

            row = []
            for cell in l.rstrip():
                row.append(MAPPING[cell])
            array.append(row)

    return array


class Craft(Environment[CraftState]):
    map_data = [[]]
    num_actions = 4

    def __init__(self, map_fn: str, rng: Random, default_x, default_y, tool_in_fact = False, wood_in_fact = False,noise = 0.0):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.key_locations = self.get_all_item()
        self.default_x = default_x
        self.default_y = default_y
        self.noise = noise
        self.tool_in_fact_default = tool_in_fact
        self.wood_in_fact_default = wood_in_fact
        super().__init__(CraftState.random(self.rng, self.map_data, self.key_locations, default_x, default_y, self.tool_in_fact_default, self.wood_in_fact_default))


    def get_all_item(self, item="key"):
        ans = []
        for y in range(self.height):
            for x in range(self.width):
                if item in self.map_data[y][x]:
                    ans.append([y, x])
        return ans

    def apply_action(self, a: ActionId):

        if self.rng.random() < self.noise:
            a = self.rng.randrange(self.num_actions)

        x, y = self.state.x + ACTIONS[a][0], self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x]:
            return
        objects = self.map_data[y][x]

        is_key = False
        if x == self.state.key_x and y == self.state.key_y:
            is_key = True

        put_extra = False
        if a == 4:
            put_extra = True
        new_facts, key_change = update_facts(self.state.facts, objects, is_key, False, False, put_extra)

        new_key_x = self.state.key_x
        new_key_y = self.state.key_y

        if key_change == 1:

            new_key_x = x
            new_key_y = y


        elif key_change == -1:

            new_key_x = -1
            new_key_y = -1

        self.state = CraftState(x, y, new_key_x, new_key_y, new_facts, self.default_x, self.default_y)

        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: CraftState, a: ActionId, s1: CraftState) -> float:
        if not s0.facts[OBJECTS["extrawood"]] and s1.facts[OBJECTS["extrawood"]] and a == 4:
            return 12.0
        return 1.0

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[CraftState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = CraftState.random(self.rng, self.map_data, self.key_locations, self.default_x, self.default_y, self.tool_in_fact_default, self.wood_in_fact_default)
    @staticmethod
    def label(state: CraftState) -> FrozenSet[int]:
        return frozenset([i for i in range(len(OBJECTS)) if state.facts[i]])
