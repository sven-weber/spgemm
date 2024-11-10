from datetime import datetime
from typing import Dict, List
from os.path import join
import pandas as pd
import subprocess
import pathlib
import argparse
import os
import matplotlib.pyplot as plt

CMD             = "./build/dphpc"
RUNS_DIR        = "runs"  # for plotting: "measurements/viscoplastic2/euler-5-40"
N_WARMUP        = 5
N_RUNS          = 10
N_SECTIONS      = 1
MEM_PER_CORE_GB = 3

# Does a run of the CMD with mpi using `nodes` nodes and returns
# the run folder.
def run_mpi(impl: str, matrix: str, nodes: int, euler: bool = False) -> str:
    print(f"Running with {nodes} nodes")
    date = datetime.now().strftime("%Y-%m-%d-%T")
    id = f"{date}-{nodes}"
    folder = join(RUNS_DIR, id)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


    if euler:
        cmd = ["sbatch", "--wait", f"--mem-per-cpu={MEM_PER_CORE_GB}G", "-n", str(nodes), "--wrap", f"mpirun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {N_SECTIONS}"]
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

FILE_NAME = "measurements"

DataFrames = Dict[int, pd.DataFrame]

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

# Because there may be multiple function names in the same file
def aggregate_vertically(dataframes: DataFrames):
    for node_id, df in dataframes.items():
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df = df.groupby("func").mean().reset_index()
        dataframes[node_id] = df

# To have the average, min and max for each function across the different files (each file is one node)
def aggregate_horizontally(dataframes: DataFrames) -> pd.DataFrame:
    all_func_names = pd.concat([df["func"] for df in dataframes.values()]).unique()
    aggregated_data = []
    for func_name in all_func_names:
        func_dfs = [df[df["func"] == func_name] for df in dataframes.values()]
        concatenated = pd.concat(func_dfs)
        avg_time = concatenated["duration"].mean()
        min_time = concatenated["duration"].min()
        max_time = concatenated["duration"].max()
        aggregated_data.append({
            "func": func_name,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time
        })
    aggregated_df = pd.DataFrame(aggregated_data)
    return aggregated_df

def graph_multiple_runs(folders: List[str]) -> Dict[str, pd.DataFrame]:
    timings_per_algo = {}
    for folder_path in folders:
        # Load algo and initialize
        algo = load_algorithm(folder_path)
        if not algo in timings_per_algo:
            timings_per_algo[algo] = pd.DataFrame()

        # Load DFs
        dataframes = load_timings(folder_path)
        aggregate_vertically(dataframes)

        # Aggregate horizontally timings related to same function
        aggregated_df = aggregate_horizontally(dataframes)
        aggregated_df['nodes'] = len(dataframes)
        timings_per_algo[algo] = pd.concat([timings_per_algo[algo], aggregated_df], axis=0)
    
    #Print the results:
    for algo in timings_per_algo:
        print(f"Aggregated data for {algo}")
        print(timings_per_algo[algo])
    
    return timings_per_algo

def plot_timings_increasingnodes(data: Dict[str, pd.DataFrame], linear: bool):
    _, ax = plt.subplots()
    for algo in data:
        timings = data[algo]
        func_data = timings[timings["func"] == "gemm"]
        timing_data = func_data["avg_time"]/(10**6)
        eb = [(func_data['avg_time'] - func_data['min_time'])/(10**6), (func_data['max_time'] - func_data['avg_time'])/(10**6)]
        ax.errorbar(func_data["nodes"], timing_data, yerr=eb, fmt='-o', label=algo)

        if (linear):
            # Calculate a linear progression based on the first timing point
            initial_time = timing_data.iloc[0]  # Timing for the first number of nodes
            linear_progression = initial_time * (1 / (func_data["nodes"] / func_data["nodes"].iloc[0]))
            #print(linear_progression)

            # Plot the linear progression line
            ax.plot(func_data["nodes"], linear_progression, linestyle='--', color='red', label="Linear Speedup")

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Time vs Nodes")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "timings_plot.png"))


def get_subfolders(folder_path):
    subfolders = []
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")
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
    parser.add_argument('--skip_run', action="store_true")
    parser.add_argument('--quadratic', required=False, action="store_true")
    parser.add_argument('--plot_linear', required=False, action="store_true")
    args = parser.parse_args()

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
                folders.append(run_mpi(impl, args.matrix, n, args.euler))

    timings = graph_multiple_runs(folders)
    plot_timings_increasingnodes(timings, args.plot_linear)
