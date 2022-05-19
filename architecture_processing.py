from genotypes import PRIMITIVES, PRIMITIVES_DARTS, Genotype_opt, Genotype_nested, ResNet18, Xception, residual_layer_simple, ResNet50
import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import plotly.express as px
import json
from train import TrainArgs, TrainNetwork
from scipy.stats import describe
import time

def hausdorff_metric(u, v, seed=0):
    '''
    Turns Hausdorff distance into a metric by enforcing symmetry.
    '''
    return max(global_hausdorff_distance(u, v, seed), global_hausdorff_distance(v, u, seed))

def cell_hausdorff_distance(c1, c2, seed=0, stats_file_path="op_stats.json"):
    '''
    Computes Hausdorff distance between two cells based on operation performance stats as weights of Hamming distance rather than standard Euclidian distance.
    '''
    with open(stats_file_path) as f:
        op_stats = np.array(list(json.load(f).values()))

    cmax = cmin = d = 0
    N1 = c1.shape[0]
    N2 = c2.shape[0]
    i_store = j_store = i_ret = j_ret = 0

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    rng = np.random.RandomState(seed)
    resort1 = np.arange(N1, dtype=np.int64)
    resort2 = np.arange(N2, dtype=np.int64)
    rng.shuffle(resort1)
    rng.shuffle(resort2)
    ar1 = np.asarray(c1)[resort1]
    ar2 = np.asarray(c2)[resort2]

    cmax = 0
    for i in range(N1):
        cmin = np.inf
        for j in range(N2):
            d = hamming(ar1[i], ar2[j], w=op_stats)
            if d < cmax: # break out of `for j` loop
                break

            if d < cmin: # always true on first iteration of for-j loop
                cmin = d
                i_store = i
                j_store = j

        # always true on first iteration of for-j loop, after that only
        # if d >= cmax
        if cmin >= cmax and d >= cmax:
            cmax = cmin
            i_ret = i_store
            j_ret = j_store

    return cmax
    


def deserialize_architecture_to_alphas(genotype, parsing_method="threshold"):
    '''
    Deserialize an architecture from a genotype to alphas weights.
    '''
    prims = PRIMITIVES if isinstance(genotype, Genotype_opt) else PRIMITIVES_DARTS
    if parsing_method != "threshold":
        raise "Only threshold parsing method is supported for now."
    steps = genotype.concat[-1] - 1
    k = sum(1 for i in range(steps) for n in range(i+2))
    alphas = np.zeros((len(genotype.genes), k, len(prims)))
    for i, cell in enumerate(genotype.genes):
        for op, to, f in cell:
            offset = to - 2
            pos = sum(1 for i in range(offset) for n in range(i+2))
            alphas[i][pos+f][prims.index(op)] = 10.0
    return alphas

def show_genotype_stats(g, save_path):
    '''
    Show the statistical dispersion of operations in a genotype and save a pie chart to the disk.
    '''
    prims = PRIMITIVES if isinstance(g, Genotype_opt) else PRIMITIVES_DARTS
    glob_stats = {p: 0 for p in prims}
    cell_stats = []
    for i, c in enumerate(g.genes):
        stats = {p: 0 for p in prims}
        for op in c:
            stats[op[0]] += 1
            glob_stats[op[0]] += 1
        cell_stats.append(stats)
    #fig = go.Figure(data=[go.Pie(labels=list(glob_stats.keys()), values=list(glob_stats.values()))])
    #fig.write_image(save_path)

def architectural_distance_metric(g1: Genotype_nested, g2: Genotype_nested, save_path: str = None):
    a1 = deserialize_architecture_to_alphas(g1)
    a2 = deserialize_architecture_to_alphas(g2)
    min_shape, max_shape = np.sort([a1.shape[0], a2.shape[0]])
    cell_dists = []
    for c1, c2 in zip(a1[:min_shape], a2[:min_shape]):        
        cell_dists.append(hausdorff_metric(c1, c2))
    for _ in range(max_shape-min_shape):
        cell_dists.append(1.0)
    if save_path:
        colors = ['lightgray']*(len(cell_dists))
        colors[np.argmax(cell_dists)] = 'crimson'
        fig = px.Figure(data=[px.Bar(x=[f"Cell {i}" for i in range(len(cell_dists))], y=cell_dists, marker_color=colors)])
        fig.update_xaxes(title_text="Cell")
        fig.update_yaxes(title_text="Hausdorff Distance", automargin=True)
        fig.write_image(save_path)
    return cell_dists 

def global_hausdorff_distance(g1: Genotype_nested, g2: Genotype_nested, seed: int = 0):
    g1 = deserialize_architecture_to_alphas(g1)
    g2 = deserialize_architecture_to_alphas(g2)

    distances = []
    for c1, c2 in zip(g1, g2):
        distances.append(max(cell_hausdorff_distance(c1, c2, seed), cell_hausdorff_distance(c2, c1, seed)))
    
    return np.mean(distances)


def benchmark_operations(num_epochs: int, num_runs: int, dataset: str = "cifar10", num_layers: int = 2, gpu: int = 0, dartopti: bool = True):
    prims = PRIMITIVES if dartopti else PRIMITIVES_DARTS
    test_arch = Genotype_nested(genes=[residual_layer_simple]*num_layers, concat=range(2,6), reductions=range(1, num_layers))
    perfs = {}
    stats = {}
    for l in range(num_layers):
        perfs[f"cell_{l}"] = {}
        stats[f"cell_{l}"] = {}
        for p in range(len(residual_layer_simple)):
            perfs[f"cell_{l}"][f"position_{p}"] = {}
            stats[f"cell_{l}"][f"position_{p}"] = {}
            arch = test_arch
            for i, op in enumerate(prims):
                print(f"Benchmarking operation {op} ({i+1}/{len(prims)})")
                _, to, fr = arch.genes[l][p]
                arch.genes[l][p] = (op, to, fr)
                results = []
                args = TrainArgs(test_arch, num_epochs, dataset, 64, num_layers, gpu)
                for r in range(num_runs):
                    print(f"Run {r}/{num_runs}")
                    trainer = TrainNetwork(args)
                    results.append(trainer.run())
                perfs[f"cell_{l}"][f"position_{p}"][op] = max(results)
            stats[f"cell_{l}"][f"position_{p}"] = describe(list(perfs[f"cell_{l}"][f"position_{p}"].values()))
    print(perfs)
    print(stats)
    perfs["stats"] = stats
    with open('op_perfs.json', 'w') as fp:
        json.dump(perfs, fp)

def compute_op_stats_from_perfs(perf_file_path: str):
    with open(perf_file_path) as f:
        perfs = json.load(f)
    perfs.pop("stats")
    op_list = list(perfs['cell_0']['position_0'].keys())
    op_stats = {op: [] for op in op_list}
    for c in perfs:
        for p in perfs[c]:
            for op in op_list:
                op_stats[op].append(perfs[c][p][op])
    stds = {op: np.std(op_stats[op]) for op in op_list}
    print(stds)
    op_stats = {op: np.median(op_stats[op]) for op in op_list}
    with open('op_stats_2.json', 'w') as fp:
        json.dump(op_stats, fp)

def plot_distance_heatmap():
    distances = pd.read_csv("distances.csv")
    fig = px.density_heatmap(distances, x="arch1", y="arch2", z="distance")
    fig.write_image("distance_heatmap.pdf", engine="orca")


if __name__ == "__main__":
    start = time.time()
    benchmark_operations(10, 4, num_layers=3, gpu=2)
    end = time.time()
    print(f"Execution time in s: {end-start}")