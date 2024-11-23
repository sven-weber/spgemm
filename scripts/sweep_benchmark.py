from datetime import datetime
from typing import Dict, List
from os.path import join
import pandas as pd
import subprocess
import pathlib
import argparse
import os
import math
import matplotlib.pyplot as plt

CMD             = "./build/dphpc"
RUNS_DIR        = "runs"  # for plotting: "measurements/viscoplastic2/euler-5-40"
N_WARMUP        = 5
N_RUNS          = 10
N_SECTIONS      = 1
MAXIMUM_MEMORY  = 128
FILE_NAME       = "measurements"
DataFrames      = Dict[int, pd.DataFrame]
COLOR_MAP       = {
    "comb": "orange", 
    "advanced": "black",
    "baseline": "blue"
}

def should_skip_run(impl: str, matrix: str, nodes: int) -> bool:
    if impl == "comb" and matrix == "viscoplastic2" and nodes == 9:
        # Segfaults for unknown reason
        return True
    if impl == "comb" and nodes == 1:
        # Does not work on one node!
        return True
    
    return False

# Does a run of the CMD with mpi using `nodes` nodes and returns
# the run folder.
def run_mpi(impl: str, matrix: str, nodes: int, euler: bool = False, daint: bool = False) -> str:
    if should_skip_run(impl, matrix, nodes):
        print(f"CAUTION: Skipping run with {nodes} nodes!!!")
        return ""
    
    print(f"Running with {nodes} cores")
    date = datetime.now().strftime("%Y-%m-%d-%T")
    id = f"{date}-{nodes}"
    folder = join(RUNS_DIR, id)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    if euler:
        mem_per_core = int(math.floor(MAXIMUM_MEMORY/nodes))
        print(f"Running with {mem_per_core} GB memory per core")
        cmd = [
            "sbatch",
            "--wait",
            #"--nodefile=euler_nodes.txt",
            f"--mem-per-cpu={mem_per_core}G",
            "--constraint=EPYC_7742",
            "-n", str(nodes),
            "-N", "1", # 1 node
            "--wrap",
            f"mpirun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {N_SECTIONS}"
        ]
    elif daint:
        # Running on broadwell cluster with 2 sockets - 18 cores each per machine 
        number_of_nodes = int(math.ceil(nodes / 36.0))
        mem_per_core = 1
        print(f"Running on {number_of_nodes} machines with {mem_per_core}G memory per core.")
        cmd = [
            "sbatch",
            "--wait", 
            "--constraint=mc", # Constraint to XC40
            "-n", str(nodes), # Number of cores
            "--ntasks-per-core=1",
            "--switches=1", # Make sure we are in the same electircal group
            f"--mem-per-cpu={mem_per_core}G",
            "-N", str(number_of_nodes), # 1 node
            "-A", "g34", # The project we use
            "--wrap",
            f"srun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {N_SECTIONS}"
        ]
    else:
        cmd = ["mpirun", "-n", str(nodes), CMD, impl, matrix, folder, str(N_RUNS), str(N_WARMUP), str(N_SECTIONS)]
    result = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()
    error = result.stderr.strip()
    if output != "":
        print(output)
    if error != "":
        print(error)

    return_code = result.returncode
    assert return_code == 0, f"Slurm job execution with {nodes} nodes failed" if euler else f"Running {CMD} with {nodes} nodes failed"
    return folder

def load_timings(folder_path: str) -> DataFrames:
    dataframes = dict()
    for file in os.listdir(folder_path):
        if file.startswith(FILE_NAME) and file.endswith(".csv"):
            node_id = int(file.split("_")[1].split(".")[0])
            df = pd.read_csv(join(folder_path, file))
            dataframes[node_id] = df
    return dataframes

def load_algorithm(folder_path: str) -> str:
    with open(join(folder_path, "algo"), "r") as f:
        return f.read().strip()

def load_matrix(folder_path: str) -> str:
    with open(join(folder_path, "matrix"), "r") as f:
        return f.read().strip()

# Because there may be multiple function names in the same file
def load_duration_as_numeric(dataframes: DataFrames):
    for node_id, df in dataframes.items():
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        dataframes[node_id] = df

# Groups the runs of all nodes together to get the maximu
# for each line in the CSV (the slowest node)
def group_runs(dataframes: DataFrames) -> pd.DataFrame:
    return pd.concat(dataframes).groupby(level=1).max()

def graph_multiple_runs(folders: List[str]) -> Dict[str, pd.DataFrame]:
    timings_per_algo = {}
    for folder_path in folders:
        # Load algo and initialize
        algo = load_algorithm(folder_path)
        if not algo in timings_per_algo:
            timings_per_algo[algo] = pd.DataFrame()

        # Load DFs
        dataframes = load_timings(folder_path)
        load_duration_as_numeric(dataframes)

        # Aggregate horizontally timings related to same function
        grouped_df = group_runs(dataframes)        
        grouped_df['nodes'] = len(dataframes)
        timings_per_algo[algo] = pd.concat([timings_per_algo[algo], grouped_df], axis=0)
    
    #Print the results:
    for algo in timings_per_algo:
        print(f"Aggregated data for {algo}")
        print(timings_per_algo[algo])
    
    matrix = load_matrix(folders[-1])
    return (matrix, timings_per_algo)

#property: Literal["duration", "bytes"]
def plot_increasingnodes(
    ax, function: str, property, 
    scaling_factor, data: Dict[str, pd.DataFrame], linear: bool
):
    for algo in data:
        timings = data[algo]
        func_data = timings[timings["func"] == function]
        
        # Get the average, min and max of the function execution
        timing_data = pd.DataFrame({
            "nodes": [],
            "avg": [],
            "min": [],
            "max": [],
        })

        grouped = func_data.groupby("nodes")
        for i, (key, item) in enumerate(grouped):
            value = grouped.get_group(key)[property]
            scaling = scaling_factor(key)
            avg_time = value.mean()/scaling
            min_time = value.min()/scaling
            max_time = value.max()/scaling
            timing_data.loc[i] = [int(key), avg_time, min_time, max_time]

        # Calculate the error bars
        eb = [
            (timing_data['avg'] - timing_data['min']),
            (timing_data['max'] - timing_data['avg'])
        ]

        color = None
        if algo in COLOR_MAP:
            color = COLOR_MAP[algo]

        ax.errorbar(timing_data["nodes"], timing_data["avg"],  yerr=eb, fmt='-o', label=algo, color=color)

        if (linear):
            # Calculate a linear progression based on the first timing point
            initial_time = timing_data.iloc[0]["avg"]  # Timing for the first number of nodes
            linear_progression = initial_time * (1 / (timing_data["nodes"] / timing_data["nodes"].iloc[0]))

            # Plot the linear progression line
            ax.plot(timing_data["nodes"], linear_progression, linestyle='--', color=color, label="Linear Speedup")

def plot_timings_increasingnodes(data: Dict[str, pd.DataFrame], matrix: str, linear: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "gemm", "duration", lambda _: 10**6, data, linear)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Time vs Nodes on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "timings_plot.png"))

def plot_bytes_increasingnodes(data: Dict[str, pd.DataFrame], matrix: str):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "bytes", "bytes", lambda _: 1, data, False)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("bytes")
    ax.set_title(f"Maximum bytes send per multiplication on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "bytes_plot.png"))

def plot_waiting_times(data: Dict[str, pd.DataFrame], matrix: str):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "wait_all", "duration", lambda _: 10**6, data, False)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Maximum wait time per Node and communication round on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "waiting_plot.png"))

def do_plots(data: Dict[str, pd.DataFrame], matrix: str, linear: bool):
    plot_timings_increasingnodes(timings, matrix, args.plot_linear)
    plot_bytes_increasingnodes(timings, matrix)
    plot_waiting_times(timings, matrix)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='sweep_benchmark',
                    description='Generate graphs from benchmarks')
    parser.add_argument('--impl', type=str, required=False, default='baseline')
    parser.add_argument('--matrix', type=str, required=False, default='test')
    parser.add_argument('--min', type=int, required=False, default=2)
    parser.add_argument('--max', type=int, required=False, default=4)
    parser.add_argument('--stride', type=int, required=False, default=1)
    parser.add_argument('--euler', action="store_true")
    parser.add_argument('--daint', action="store_true")
    parser.add_argument('--skip_run', action="store_true")
    parser.add_argument('--quadratic', required=False, action="store_true")
    parser.add_argument('--plot_linear', required=False, action="store_true")
    args = parser.parse_args()

    assert not (args.euler and args.daint)

    folders = []

    if args.skip_run:
        # Just fetch all the folders inside run and
        # only do the visualization!
        folders = get_subfolders(RUNS_DIR)
    else:
        # Check if multiple implementations where provided
        impls = args.impl.split(',')
        for impl in impls:
            print(f"Executing implementation {impl}")
            for n in range(args.min, args.max+1, args.stride):
                if args.quadratic:
                    n = n*n
                run_result = run_mpi(impl, args.matrix, n, args.euler, args.daint)
                if run_result != "":
                    folders.append(run_result)

    (matrix, timings) = graph_multiple_runs(folders)

    # Plots to create
    do_plots(timings, matrix, args.plot_linear)

