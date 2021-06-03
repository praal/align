from random import Random
from typing import Dict, Hashable, List, Optional, Tuple

from environment import ActionId, State
from environment.craft import OBJECTS1
from .rl import Policy

class Empathic(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, epsilon: float, default_q: float,
                 num_actions: int, rng: Random, others_q, penalty: int, others_alpha, objects, problem_mood, others_dist = [1.0], others_init = [[1, 1]], our_alpha=1.0, caring_func = "sum", restricted = []):
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
        self.caring_func = caring_func
        self.objects = objects
        self.others_init = others_init
        self.problem_mood = problem_mood
        self.default_key_x = 2
        self.default_key_y = 3
        self.restricted = restricted


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
                if self.problem_mood == 1:
                    for i in range(len(restrict_locations)):
                        if state.x == restrict_locations[i][0] and state.y == restrict_locations[i][1]:
                            restrict = list(range(self.num_actions) )
                elif self.problem_mood == 2:
                    restrict = list(range(self.num_actions))

            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)

    def get_max_q(self, uid, qval):
        max_q = qval.get((uid, 0), (-1000, False))
        for action in range(1, self.num_actions):
            q = qval.get((uid, action), (-1000, False))
            if q > max_q:
                max_q = q
        if max_q[0] == 0:
            print("zero!", uid)
        return max_q[0]

    def estimate_other(self, state):
        ans = 0
        first = True
        for other in range(len(self.others_q)):
            if state.key_x == -1:
                ans += self.others_dist[other] * self.penalty
                continue
            fact_list = [False] * len(self.objects)
            if "extra" in self.objects and state.facts[OBJECTS1["extra"]]:
                fact_list[4] = True
            if "extrawood" in self.objects and state.facts[OBJECTS1["extrawood"]]:
                fact_list[5] = True

            facts = tuple(fact_list)

            uid = (self.others_init[other][0], self.others_init[other][1], state.key_x, state.key_y, facts)
            max_q = self.get_max_q(uid, self.others_q[other])
                       
            if self.caring_func == "sum":
                ans += self.others_dist[other] * max_q

            elif self.caring_func == "min":
                if first:
                    ans = max_q
                    first = False
                else:
                    ans = min(ans, max_q)
            elif self.caring_func == "neg":
                uid2 = (self.others_init[other][0], self.others_init[other][1], self.default_key_x, self.default_key_y, facts)
                max_q2 = self.get_max_q(uid2, self.others_q[other])
                ans += self.others_dist[other] * min(max_q2, max_q)

            else:
                print("Error!")
        return ans


    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
            others = 0
            for i in range(len(self.others_dist)):
                others += self.others_alpha[i] * self.estimate_other(s1)
            q += self.alpha * (self.our_alpha * r + others)
        else:
            q += self.alpha * (self.our_alpha * r + self.gamma * (self.estimate(s1)))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))




