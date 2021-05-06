import sys  # noqa
from os import path as p  # noqa
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa

import logging
from os import path
from random import Random
from time import time
from typing import List, Tuple

import tensorflow as tf

from environment import Farm, FarmState, ReachFacts
from rl import Agent, EpsilonGreedy, HLAction, HRL, MonPOP, MonSeq, Option, \
        POPlan, SeqPlan
from rl.dqn import EpsilonDQN
from utils import SequenceReport

from environment.farm import OBJECTS

DEFAULT_Q = 0.0
TOTAL_STEPS = 10000000
EPISODE_LENGTH = 1000
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 0

logging.basicConfig(level=logging.INFO)

here = path.dirname(__file__)
map_fn = path.join(here, "craft/farm.map")

init = FarmState(5.5, 5.5, 0.0, 0.0, set())

rng = Random(2019)
env = Farm(map_fn, rng)

tasks = [[OBJECTS["plank"]],
         [OBJECTS["stick"]],
         [OBJECTS["cloth"]],
         [OBJECTS["rope"]],
         [OBJECTS["bridge"]],
         [OBJECTS["bed"]],
         [OBJECTS["axe"]],
         [OBJECTS["shears"]],
         [OBJECTS["gold"]],
         [OBJECTS["gem"]]]
tasks = tasks[START_TASK:END_TASK+1]

sess = tf.Session()

with open("tmp/farm-dqn.csv", "w") as csvfile:
    print("dqn: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("dqn: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policy = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                            num_features=22, num_actions=4, num_neurons=64,
                            num_hidden_layers=6, batch_size=32, buffer_size=32,
                            rng=rng, sess=sess, start=1000, frequency=100,
                            name="dqn", epsilon=0.1)
        agent = Agent(env, policy, reward, rng)
        try:
            start = time()
            agent.train(steps=TOTAL_STEPS,
                        steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("dqn: Ran task {} for {} seconds.".format(j + START_TASK,
                                                           end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("dqn: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)

del policy
del agent

sess = tf.Session()

# go to wood
policyA = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionA", epsilon=0.1)
optionA = Option(env, policyA,
                 lambda s: "wood" in env.observe(s))  # type: ignore

# go to toolshed
policyB = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionB", epsilon=0.1)
optionB = Option(env, policyB,
                 lambda s: "toolshed" in env.observe(s))  # type: ignore

# go to workbench
policyC = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionC", epsilon=0.1)
optionC = Option(env, policyC,
                 lambda s: "workbench" in env.observe(s))  # type: ignore

# go to grass
policyD = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionD", epsilon=0.1)
optionD = Option(env, policyD,
                 lambda s: "grass" in env.observe(s))  # type: ignore

# go to factory
policyE = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionE", epsilon=0.1)
optionE = Option(env, policyE,
                 lambda s: "factory" in env.observe(s))  # type: ignore

# go to iron
policyF = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionF", epsilon=0.1)
optionF = Option(env, policyF,
                 lambda s: "iron" in env.observe(s))  # type: ignore

# go to gold
policyG = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionG", epsilon=0.1)
optionG = Option(env, policyG,
                 lambda s: "gold" in env.observe(s))  # type: ignore

# go to gem
policyH = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                     num_features=22, num_actions=4, num_neurons=64,
                     num_hidden_layers=6, batch_size=32, buffer_size=32,
                     rng=rng, sess=sess, start=1000, frequency=100,
                     name="optionH", epsilon=0.1)
optionH = Option(env, policyH,
                 lambda s: "gem" in env.observe(s))  # type: ignore

options = [optionA, optionB, optionC, optionD, optionE, optionF, optionG,
           optionH]

with open("tmp/farm-hrl.csv", "w") as csvfile:
    print("hrl: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("hrl: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policyMeta = EpsilonDQN(alpha=0.00001, gamma=1.0, default_q=DEFAULT_Q,
                                num_features=22, num_actions=len(options),
                                num_neurons=64, num_hidden_layers=6,
                                batch_size=32, buffer_size=32, rng=rng,
                                sess=sess, start=1000, frequency=100,
                                name="HRL", epsilon=0.1)
        policyHRL = HRL(policyMeta, options, 1.0, rng)
        agentHRL = Agent(env, policyHRL, reward, rng)
        try:
            start = time()
            agentHRL.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
                           report=report)
            end = time()
            print("hrl: Ran task {} for {} seconds.".format(j + START_TASK,
                                                            end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("hrl: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)

del policyMeta
del policyHRL
del agentHRL


# raw materials:
get_wood = HLAction([], [], [OBJECTS["wood"]], 0)
get_grass = HLAction([], [], [OBJECTS["grass"]], 3)
get_iron = HLAction([], [], [OBJECTS["iron"]], 5)
get_gold = HLAction([OBJECTS["bridge"]], [], [OBJECTS["gold"]], 6)
get_gem = HLAction([OBJECTS["axe"]], [], [OBJECTS["gem"]], 7)

# at toolshed:
make_plank = HLAction([OBJECTS["wood"]], [OBJECTS["wood"]], [OBJECTS["plank"]],
                      1)
make_rope = HLAction([OBJECTS["grass"]], [OBJECTS["grass"]], [OBJECTS["rope"]],
                     1)
make_axe = HLAction([OBJECTS["stick"], OBJECTS["iron"]],
                    [OBJECTS["stick"], OBJECTS["iron"]], [OBJECTS["axe"]], 1)
make_bow = HLAction([OBJECTS["rope"], OBJECTS["stick"]],
                    [OBJECTS["rope"], OBJECTS["stick"]], [OBJECTS["bow"]], 1)

# at workbench:
make_stick = HLAction([OBJECTS["wood"]], [OBJECTS["wood"]], [OBJECTS["stick"]],
                      2)
make_saw = HLAction([OBJECTS["iron"]], [OBJECTS["iron"]], [OBJECTS["saw"]], 2)
make_bed = HLAction([OBJECTS["plank"], OBJECTS["grass"]],
                    [OBJECTS["plank"], OBJECTS["grass"]], [OBJECTS["bed"]], 2)
make_shears = HLAction([OBJECTS["stick"], OBJECTS["iron"]],
                       [OBJECTS["stick"], OBJECTS["iron"]],
                       [OBJECTS["shears"]], 2)

# at factory:
make_cloth = HLAction([OBJECTS["grass"]], [OBJECTS["grass"]],
                      [OBJECTS["cloth"]], 4)
make_bridge = HLAction([OBJECTS["iron"], OBJECTS["wood"]],
                       [OBJECTS["iron"], OBJECTS["wood"]], [OBJECTS["bridge"]],
                       4)
make_goldware = HLAction([OBJECTS["gold"]], [OBJECTS["gold"]],
                         [OBJECTS["goldware"]], 4)
make_ring = HLAction([OBJECTS["gem"]], [OBJECTS["gem"]], [OBJECTS["ring"]], 4)


plan0 = [get_wood, make_plank]
plan1 = [get_wood, make_stick]
plan2 = [get_grass, make_cloth]
plan3 = [get_grass, make_rope]
plan4 = [get_iron, get_wood, make_bridge]
plan5 = [get_wood, make_plank, get_grass, make_bed]
plan6 = [get_wood, make_stick, get_iron, make_axe]
plan7 = [get_wood, make_stick, get_iron, make_shears]
plan8 = [get_iron, get_wood, make_bridge, get_gold]
plan9 = [get_wood, make_stick, get_iron, make_axe, get_gem]

plans = [plan0, plan1, plan2, plan3, plan4, plan5, plan6, plan7, plan8, plan9]
plans = plans[START_TASK:END_TASK+1]

with open("tmp/farm-seq.csv", "w") as csvfile:
    print("seq: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("seq: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policySeq = SeqPlan(plans[j], options, rng)
        agentSeq = Agent(env, policySeq, reward, rng)
        try:
            start = time()
            agentSeq.train(steps=TOTAL_STEPS,
                           steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("seq: Ran task {} for {} seconds.".format(j + START_TASK,
                                                            end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("seq: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)

del policySeq
del agentSeq

for option in options:
    option.policy.clear()


with open("tmp/farm-monseq.csv", "w") as csvfile:
    print("monseq: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("monseq: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policyMonSeq = MonSeq(plans[j], env.label, goal, options, rng)
        agentMonSeq = Agent(env, policyMonSeq, reward, rng)
        try:
            start = time()
            agentMonSeq.train(steps=TOTAL_STEPS,
                              steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("monseq: Ran task {} for {} seconds.".format(j + START_TASK,
                                                            end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("monseq: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)

del policyMonSeq
del agentMonSeq

for option in options:
    option.policy.clear()


pops: List[Tuple[List[HLAction], List[Tuple[int, int]]]] = \
    [(plan0, [(0, 1)]), (plan1, [(0, 1)]), (plan2, [(0, 1)]),
     (plan3, [(0, 1)]), (plan4, [(0, 2), (1, 2)]),
     (plan5, [(0, 1), (1, 2), (2, 3)]), (plan6, [(0, 1), (1, 3), (2, 3)]),
     (plan7, [(0, 1), (1, 3), (2, 3)]), (plan8, [(0, 2), (1, 2), (2, 3)]),
    (plan9, [(0, 1), (1, 3), (2, 3), (3, 4)])]
pops = pops[START_TASK:END_TASK+1]

with open("tmp/farm-pop.csv", "w") as csvfile:
    print("pop: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("pop: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policyPOP = POPlan(pops[j],
                           EpsilonDQN(alpha=0.00001, gamma=1.0,
                                      default_q=DEFAULT_Q, num_features=22,
                                      num_actions=len(options), num_neurons=64,
                                      num_hidden_layers=6, batch_size=32,
                                      buffer_size=32, rng=rng, sess=sess,
                                      start=1000, frequency=100, name="POP",
                                      epsilon=0.1),
                           options, 1.0, rng)
        agentPOP = Agent(env, policyPOP, reward, rng)
        try:
            start = time()
            agentPOP.train(steps=TOTAL_STEPS,
                           steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("pop: Ran task {} for {} seconds.".format(j + START_TASK,
                                                            end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("pop: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)

del policyPOP
del agentPOP

for option in options:
    option.policy.clear()

with open("tmp/craft-monpop.csv", "w") as csvfile:
    print("monpop: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, [init], EPISODE_LENGTH, TRIALS)

    for j, goal in enumerate(tasks):
        print("monpop: begin task {}".format(j + START_TASK))
        rng.seed(2019 + j)

        reward = ReachFacts(env, goal)
        policyMonPOP = MonPOP(pops[j], env.label, goal,
                              EpsilonDQN(alpha=0.00001, gamma=1.0,
                                         default_q=DEFAULT_Q, num_features=22,
                                         num_actions=len(options),
                                         num_neurons=64, num_hidden_layers=6,
                                         batch_size=32, buffer_size=32,
                                         rng=rng, sess=sess, start=1000,
                                         frequency=100, name="MonPOP",
                                         epsilon=0.1),
                              options, 1.0, rng)
        agentMonPOP = Agent(env, policyMonPOP, reward, rng)
        try:
            start = time()
            agentMonPOP.train(steps=TOTAL_STEPS,
                              steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("monpop: Ran task {} for {} seconds.".format(j + START_TASK,
                                                               end - start))
        except KeyboardInterrupt:
            end = time()
            logging.warning("monpop: interrupted task %s after %s seconds!",
                            j + START_TASK, end - start)

        report.increment(TOTAL_STEPS)
