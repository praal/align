import craftexpcausal, craftexp, craftseq, craftpop, crafthrl
import numpy as np
import matplotlib.pyplot as plt

start = 0
num_exp = 5

rl = 1000
pc = 2000
ical = 3000
def generate_data():
    for i in range(start, num_exp):
        print(i, "******")
        craftexpcausal.run( f"../../data/final/causal_bed_hmin_18_{i}.csv", i)
       ## craftexp.run(f"../../data/final/ql_bed_18_{i}.csv", i)
       # craftseq.run( f"../../data/final/seq_bed_18_{i}.csv", i)
       # craftpop.run( f"../../data/final/pop_bed_18_{i}.csv", i)
       # crafthrl.run(f"../../data/final/hrl_bed_18_{i}.csv", i)



def visualize_results(output_dir, results_paths, length, alg_names):

    num_alg = len(results_paths)
    results = []
    for i in range(num_alg):
        ans = []
        for j in range(num_exp):
            tmp = np.genfromtxt(results_paths[i][j])
            ans.append(tmp[:length, 1])

        results.append(ans)

    for i in range(num_alg):
        means = np.mean(results[i], axis=0)
        std = np.std(results[i], axis=0)
        plt.plot(range(length), means, label=alg_names[i])
        ci = 1.96 * std / np.sqrt(num_exp)
        plt.fill_between(range(length), (means - ci), (means + ci), alpha=.1)


    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward")
    plt.legend(loc='best')

    plt.savefig(output_dir + "fig1.png")
    plt.show()

#
datasets = list()
ql_paths = [f"../../data/final/ql_{i}.csv" for i in range(num_exp)]
causal_paths = [f"../../data/final/causal_18_{i}.csv" for i in range(num_exp)]
hrl_paths = [f"../../data/final/hrl_bed_18_{i}.csv" for i in range(num_exp)]
seq_paths = [f"../../data/final/seq_bed_18_{i}.csv" for i in range(num_exp)]
pop_paths = [f"../../data/final/pop_bed_18_{i}.csv" for i in range(num_exp)]

pc_paths = [f"../../data/final/causal_pc_18_{i}.csv" for i in range(num_exp)]
rl_paths = [f"../../data/final/causal_rl_18_{i}.csv" for i in range(num_exp)]

h2_paths = [f"../../data/final/causal_h2_18_{i}.csv" for i in range(num_exp)]
hmin_paths = [f"../../data/final/causal_hmin_18_{i}.csv" for i in range(num_exp)]

datasets.append(ql_paths)
datasets.append(causal_paths)
#datasets.append(h2_paths)
datasets.append(hmin_paths)
datasets.append(pc_paths)
datasets.append(rl_paths)

#datasets.append(hrl_paths)
#datasets.append(seq_paths)
#datasets.append(pop_paths)
visualize_results("../../data/final/", datasets , 190, ["q-Learning","all-sum", "max-path", "pc", "rl"])
#generate_data()