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

IRON_IND = 0

OBJECTS = dict([(v, k) for k, v in enumerate(
    ["iron", "wood", "gold", "axe", "gem", "ring"])])  # , "wall"])])
#    ["iron", "wood", "axe", "factory", "gem", "gold", "ring", "bridge"])])  # , "wall"])])



def update_facts(facts: Sequence[bool], objects: Observation, is_iron):
    state = set([i for i, v in enumerate(facts) if v])
    iron_change = 0
    for o in objects:

        if o == "iron":
            if OBJECTS["iron"] in state:
                state.remove(OBJECTS["iron"])
                iron_change = 1

            elif is_iron:
                state.add(OBJECTS[o])
                iron_change = -1
        elif o != "gold" and o != "gem" and o in OBJECTS:
             state.add(OBJECTS[o])
        elif o == "gem" and OBJECTS["axe"] in state:
             state.add(OBJECTS[o])

        elif o == "gold" and OBJECTS["axe"] in state:
            state.add(OBJECTS[o])


    if "factory" in objects:
        if OBJECTS["iron"] in state and OBJECTS["wood"] in state:

            if OBJECTS["axe"] not in state:
                state.add(OBJECTS["axe"])


        if OBJECTS["gold"] in state and OBJECTS["gem"] in state:
            state.add(OBJECTS["ring"])


    return state, iron_change


class CraftState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]


    def __init__(self, x: int, y: int, iron_x, iron_y,  facts: Set[int]):
        self.x = x
        self.y = y

        fact_list = [False] * len(OBJECTS)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.iron_x = iron_x
        self.iron_y = iron_y
        self.uid = (self.x, self.y, iron_x, iron_y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {:2d}, {:2d}, {})".format(self.x, self.y, self.iron_x, self.iron_y, self.facts)

    @staticmethod
    def random(rng: Random,
               map_data: Sequence[Sequence[Observation]], iron_locations: List[List[int]]) -> 'CraftState':
        # return CraftState(5, 5, set())
        while True:
            #y = rng.randrange(len(map_data))
            #x = rng.randrange(len(map_data[0]))
            x = 1
            y = 1
            ind = rng.randrange(len(iron_locations))
            iron_y = iron_locations[ind][0]
            iron_x = iron_locations[ind][1]
            if "wall" not in map_data[y][x] and "wall" not in map_data[iron_y][iron_x]:
                facts, _ = update_facts((), map_data[y][x], 0)
                return CraftState(x, y, iron_x, iron_y, facts)


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

    def __init__(self, map_fn: str, rng: Random):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.iron_locations = self.get_all_item()
        super().__init__(CraftState.random(self.rng, self.map_data, self.iron_locations))


    def get_all_item(self, item="iron"):
        ans = []
        for y in range(self.height):
            for x in range(self.height):
                if item in self.map_data[y][x]:
                    ans.append([y, x])
        return ans

    def delete_iron(self):
        x = self.state.iron_x
        y = self.state.iron_y
        self.map_data[y][x] = frozenset()

    def add_iron(self):
        x = self.state.iron_x
        y = self.state.iron_y
        self.map_data[y][x] = frozenset(["iron"])

    def apply_action(self, a: ActionId):
        x, y = self.state.x + ACTIONS[a][0], self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x]:
            return
        objects = self.map_data[y][x]

        is_iron = False
        if x == self.state.iron_x and y == self.state.iron_y:
            is_iron = True

        new_facts, iron_change = update_facts(self.state.facts, objects, is_iron)

        new_iron_x = self.state.iron_x
        new_iron_y = self.state.iron_y

        if iron_change == 1:
          #  print("here", iron_change, self.state)
            new_iron_x = x
            new_iron_y = y
        #    print("########", self.map_data[y][x])

        elif iron_change == -1:

         #   print("here", self.state, self.map_data[y][x])
            new_iron_x = -1
            new_iron_y = -1
         #   self.map_data[y][x] = frozenset()
         #   print("####", self.map_data[y][x])

        self.state = CraftState(x, y, new_iron_x, new_iron_y, new_facts)
      #  if iron_change == 1:
        #    print("ttttthere", iron_change, self.state)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: CraftState, a: ActionId, s1: CraftState) -> float:
        return 1.0

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[CraftState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = CraftState.random(self.rng, self.map_data, self.iron_locations)
    @staticmethod
    def label(state: CraftState) -> FrozenSet[int]:
        return frozenset([i for i in range(len(OBJECTS)) if state.facts[i]])
