from random import Random
from typing import Dict, Hashable, List, Optional, Tuple

from environment import ActionId, State

from .rl import Policy

class Empathic(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, epsilon: float, default_q: float,
                 num_actions: int, rng: Random, others_q, penalty: int, others_alpha: float):
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
                         restrict: Optional[List[int]] = None) -> ActionId:
        if self.rng.random() < self.epsilon:
            if restrict is None:
                restrict = list(range(self.num_actions))
            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)

    def estimate_other(self, state, init_x=1, init_y=1):
       # print(state, state.iron_x,state.iron_y)
        if state.iron_x == -1:
            return self.penalty
       # print(state.iron_x,state.iron_y)
        return -1.0 * (state.iron_x - 1 + state.iron_y - 1)
        uid = (init_x, init_y, state.iron_x, state.iron_y, tuple([False, False]))

        max_q = self.others_q.get((uid, 0), self.default_q)
        for action in range(1, self.num_actions):
            q = self.Q.get((uid, action), self.default_q)
            if q > max_q:
                max_q = q
        #print(max_q, uid)
        if max_q[0] == 0:
            print(uid)
        return max_q[0]

        #print(max_q, uid)

        uid2 = (init_x, init_y, 2, 1, tuple([False, False]))
        max_q2 = self.others_q.get((uid2, 0), self.default_q)
        for action in range(1, self.num_actions):
            q = self.Q.get((uid, action), self.default_q)
            if q > max_q2:
                max_q2 = q
        #print(max_q, max_q2, state.iron_x, state.iron_y)
        return max_q[0] - max_q2[0]

    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
           # print("hereeee")
            q += self.alpha * (r + self.others_alpha * self.estimate_other(s1))
        else:
            q += self.alpha * (r + self.gamma * (self.estimate(s1)))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))




