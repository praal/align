from random import Random
from typing import Dict, Hashable, List, Optional, Tuple

from environment import ActionId, State
from environment.craft import OBJECTS, update_facts
from .rl import Policy

class Empathic(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, epsilon: float, default_q: float,
                 num_actions: int, rng: Random, others_q, penalty: int, others_alpha: float, others_dist = [1.0], our_alpha=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_q = default_q, False
        self.num_actions = num_actions
        self.rng = rng
        self.Q = {}
        self.others_q = others_q
        self.penalty = penalty
        self.others_alpha = others_alpha
        self.our_alpha = our_alpha
        self.others_dist = others_dist

    def clear(self):
        self.Q = {}

    def estimate(self, state: State) -> float:
        max_q = self.Q.get((state.uid, 0), self.default_q)
        for action in range(1, self.num_actions):
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:
                max_q = q
        return max_q[0]

    def get_best_action(self, state: State,
                        restrict: Optional[List[int]] = None) -> ActionId:
        if restrict is None:
            restrict = list(range(self.num_actions))
        max_q = self.Q.get((state.uid, restrict[0]), self.default_q)
        best_actions = [restrict[0]]
        for action in restrict[1:]:
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:  # or (self.evaluation and q[1] and not max_q[1]):
                max_q = q
                best_actions = [action]
            elif q == max_q:
                best_actions.append(action)
        return self.rng.choice(best_actions)

    def get_policy(self):
        ans = {}
        for state,_ in self.Q:
            max_q = self.Q.get((state, 0), self.default_q)
            max_action = 0
            for action in range(1, self.num_actions):
                q = self.Q.get((state, action), self.default_q)
                if q > max_q:
                    max_q = q
                    max_action = action
            ans[state] = max_action
        return ans

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None, restrict_locations = [[8, 1], [6, 5]]) -> ActionId:
        if self.rng.random() < self.epsilon:
            if restrict is None:
                restrict = list(range(self.num_actions - 1))
                for i in range(len(restrict_locations)):
                    if state.x == restrict_locations[i][0] and state.y == restrict_locations[i][1]:
                        restrict = list(range(self.num_actions) )

            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)

    def estimate_other(self, state, init_x=1, init_y=1):
        ans = 0
        for other in range(len(self.others_q)):
            if state.key_x == -1:
                ans += self.others_dist[other] * self.penalty
                continue
            fact_list = [False] * len(OBJECTS)
            if state.facts[OBJECTS["extra"]]:
                fact_list[4] = True
            if state.facts[OBJECTS["extrawood"]]:
                fact_list[5] = True

            facts = tuple(fact_list)

            uid = (init_x, init_y, state.key_x, state.key_y, facts)
            max_q = self.others_q[other].get((uid, 0), (-1000, False))
            for action in range(1, self.num_actions):
                q = self.others_q[other].get((uid, action), (-1000, False))
                if q > max_q:
                    max_q = q
            if max_q[0] == 0:
                print("zero!", uid)

            ans += self.others_dist[other] * max_q[0]
        return ans


    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool, p = [1], init_x = [[1, 1]]):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
            others = 0
            for i in range(len(p)):
                others += p[i] * self.estimate_other(s1, init_x[i][0], init_x[i][1])
            q += self.alpha * (self.our_alpha * r + self.others_alpha * others)
        else:
            q += self.alpha * (self.our_alpha * r + self.gamma * (self.estimate(s1)))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))




