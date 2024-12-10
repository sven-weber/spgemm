import os
import pandas as pd
from typing import Dict, List, Tuple
from os.path import join
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

# change this!
RUNS_DIR1 = "measurements/HV15R"
RUNS_DIR2 = "measurements/HV15R"

TITLE = "diocane title"
MEASUREMENT_FILE_NAME               = "measurements"
OMP_ALGOS               = [
    "comb", "drop_at_once_parallel", "drop_parallel"
]
FUNCTION = "gemm"
SCALING_FACTOR = 10**9
plt.rcParams['font.family'] = 'Computer Modern Roman'
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
STYLES = {
    # color, marker, offst, name
    "comb": ('dodgerblue', 'D', (-80, -40), "\\textsc{CombBLAS}", None),
    "drop_at_once_parallel-no-shuffling": ('deeppink', 'o', (-50, -20), "\\textsc{DropAtOnceNS}", None),
    "todo2": ('goldenrod', 'h', (-60, -20), "\\textsc{NHWC}", None),
    "todo3": ('brown', '^', (-210, -135), "\\textsc{Tensor Macro}", 26),
    "todo4": ('darkcyan', 'H', (-75, -15), "\\textsc{Merged}", None),
    "todo5": ('darkolivegreen', 's', (-120, -118), "\\textsc{Merged+Blocked}", None),
}
OUTPUT_FILE               = "plot1.pdf"

def get_subfolders(folder_path):
    subfolders = []
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")
    subfolders.sort()
    return subfolders

def load_algorithm(folder_path: str) -> str:
    with open(join(folder_path, "algo"), "r") as f:
        return f.read().strip()

def load_nodes_mpi(folder_path: str) -> Tuple[int, int]:
    parts = folder_path.split('-')
    return int(parts[-2]), int(parts[-1])

def load_matrix(folder_path: str) -> str:
    with open(join(folder_path, "matrix"), "r") as f:
        return f.read().strip()

# Returns a dict of dataframes
def load_timings(folder_path: str):
    dataframes = dict()
    for file in os.listdir(folder_path):
        if file.startswith(MEASUREMENT_FILE_NAME) and file.endswith(".csv"):
            node_id = int(file.split("_")[1].split(".")[0])
            df = pd.read_csv(join(folder_path, file))
            dataframes[node_id] = df
    return dataframes

# Because there may be multiple function names in the same file
def load_duration_as_numeric(dataframes: Dict[str, pd.DataFrame]):
    for node_id, df in dataframes.items():
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        dataframes[node_id] = df

# Groups the runs of all nodes together to get the maximu
# for each line in the CSV (the slowest node)
def group_runs(node_dict: dict) -> pd.DataFrame:
    # Find all the functions that exists in the files
    functions = []
    for node_id in node_dict:
        functions += node_dict[node_id]["func"].unique().tolist()
    functions = list(set(functions))

    # For each of the functions, do a grouping!
    result = []
    for function in functions:
        groups = {}
        for node_id in node_dict:
            df = node_dict[node_id][node_dict[node_id]["func"] == function]
            # This ensures the functions are in the same order, no matter
            # which order they where in in the original file
            groups[node_id] = df.reset_index(drop=True)
        
        # Group by the same index!
        result.append(pd.concat(groups).groupby(level=1).max())
        
    # Return new dataframe with all the functions grouped!
    return pd.concat(result)

def graph_multiple_runs(folders: List[str]) -> Tuple[str, Dict[str, pd.DataFrame]]:
    timings_per_algo = {}
    for folder_path in folders:
        # Load algo and initialize
        algo = load_algorithm(folder_path)
        if not algo in timings_per_algo:
            timings_per_algo[algo] = pd.DataFrame()

        # Load DFs
        node_dict = load_timings(folder_path)
        load_duration_as_numeric(node_dict)
        grouped_df = group_runs(node_dict)

        nodes, mpi = load_nodes_mpi(folder_path)
        grouped_df['nodes'] = nodes
        grouped_df['mpi'] = mpi
        timings_per_algo[algo] = pd.concat([timings_per_algo[algo], grouped_df], axis=0)
    
    matrix = load_matrix(folders[-1])
    return (matrix, timings_per_algo)

def create_plots(ax: Axes, matrix: str, benchmarks: Dict[str, pd.DataFrame], main: bool) -> None:
    for algo in benchmarks:
        timings = benchmarks[algo]
        func_data = timings[timings["func"] == FUNCTION]

        # Get the average, min and max of the function execution
        timing_data = pd.DataFrame({
            "nodes": [],
            "avg": [],
            "min": [],
            "max": [],
        })

        grouped = func_data.groupby("nodes")
        for i, (key, item) in enumerate(grouped):
            value = item["duration"]
            avg_time = value.mean()/SCALING_FACTOR
            min_time = value.min()/SCALING_FACTOR
            max_time = value.max()/SCALING_FACTOR
            timing_data.loc[i] = [int(key), avg_time, min_time, max_time]

        # Calculate the error bars
        eb = [
            (timing_data['avg'] - timing_data['min']),
            (timing_data['max'] - timing_data['avg'])
        ]

        color, marker, offst, name, rotation = STYLES[algo]

        nodes = timing_data["nodes"]
        values = timing_data["avg"]
        print(nodes, values)
        if main:
            ax.annotate(name, (nodes.iloc[-1], values.iloc[-1]), color=color, xytext=offst, textcoords='offset points', fontsize='x-large', rotation=rotation or 0)
        ax.errorbar(nodes, values, yerr=eb, fmt='-o', label=algo, marker=marker, color=color)

    if main:
        ax.set_xlabel("Cores (MPI Rank $\\times$ OpenMP Threads)")
    else:
        ax.set_xlabel("MPI Rank")
    if main:
        ax.set_ylabel('Time [s]',
                rotation='horizontal',
                loc='top',
                labelpad=-50)

    ax.tick_params(axis='both', direction='in', which='major', pad=5)
    ax.grid(which='major', axis='y', linewidth=.5, dashes=(3,3))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    if not main:
        ax.yaxis.tick_right()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    if main:
        ax.set_title(f"TITLE: {matrix}", fontsize=15)
        
folders1 = get_subfolders(RUNS_DIR1)
folders2 = get_subfolders(RUNS_DIR2)

matrix1, timings1 = graph_multiple_runs(folders1)
matrix2, timings2 = graph_multiple_runs(folders2)
assert matrix1 == matrix2

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
sub_ax = inset_axes(
    parent_axes=ax,
    loc='lower left',
    width="35%",
    height="50%",
    borderpad=1
)
sub_ax.set_facecolor('whitesmoke')

create_plots(ax, matrix1, timings1, True)
create_plots(sub_ax, matrix1, timings2, False)

fig.savefig(OUTPUT_FILE, bbox_inches='tight')
