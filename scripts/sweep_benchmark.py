from datetime import datetime
from typing import Dict, List
from os.path import join
import pandas as pd
import numpy as np
import subprocess
import pathlib
import argparse
import shutil
import os
import re
import math
import matplotlib.pyplot as plt

CMD                     = "./build/dphpc"
RUNS_DIR                = "runs"  # for plotting: "measurements/viscoplastic2/euler-5-40"
N_WARMUP                = 5
N_RUNS                  = 50
MAXIMUM_MEMORY          = 128
FILE_NAME               = "measurements"
SHUFFLING               = "none"
PARTITIONING            = "balanced"
DataFrames              = Dict[int, pd.DataFrame]
COLOR_MAP               = {
    "comb1d": "blue",
    "comb2d": "green",
    "comb3d": "sienna",
    "drop_parallel": "purple",
}
SBATCH_TIME_LIMIT_MIN   = 45

# Which algorithms should be run using OpenMP Threads
# in addition to MPI
OMP_ALGOS               = [
    "comb1d", "comb2d", "comb3d", "drop_parallel"
]

# Allows us to easily remove data from a graph
ALGOS_TO_SKIP_WHILE_PLOTTING = [

]

LAST_JOB = None

threads_per_mpi = 16
cpus_per_machine = 64
MPI_OPEN_MP_CONFIG = [
    {
        "nodes": (16*threads_per_mpi)/cpus_per_machine,
        "mpi": 16
    },
    {
        "nodes": (64*threads_per_mpi)/cpus_per_machine,
        "mpi": 64
    },
    {
        "nodes": (256*threads_per_mpi)/cpus_per_machine,
        "mpi": 256
    },
]

def should_skip_run(impl: str, matrix: str, nodes: int) -> bool:
    if "comb" in impl and matrix == "viscoplastic2" and nodes == 9:
        # Segfaults for unknown reason
        return True
    if "comb" in impl and nodes == 1:
        # Does not work on one node!
        return True
    
    return False

# Does a run of the CMD with mpi using `nodes` nodes and returns
# the run folder.
def run_mpi(impl: str, matrix: str, nodes: int, euler: bool = False, daint: bool = False, persist: bool = True, parallel_load: bool = True) -> str:
    if should_skip_run(impl, matrix, nodes):
        print(f"CAUTION: Skipping run with {nodes} nodes!!!")
        return ""
    
    print(f"Running with {nodes} cores")
    folder = create_runs_dir(nodes, nodes)
    # Environment variables for the task
    env = os.environ.copy()

    persist_str = "true" if persist else "false"
    parallel_load_str = "true" if parallel_load else "false"

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
            f"mpirun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {SHUFFLING} {PARTITIONING} {parallel_load_str} {persist_str} matrices"
        ]
    elif daint:
        total_number_cores = nodes * cpus_per_machine
        print(f"Running on {nodes} machines with a total of {total_number_cores} cores")
        
        algo_conf = []

        if impl not in OMP_ALGOS:
            # All our algorithms use one MPI process per core
            # Number of tasks = number of processors
            print("Using MPI placement with 1 process per core")
            algo_conf = [
                "-n", str(total_number_cores), # Number of tasks = number of cores
                "--cpus-per-task=1" # One CPU per MPI process
            ]
        else:
            # COMB uses MPI and OpenMP.
            # It will have one MPI task per node and as many threads as cores
            print(f"Using OpenMP placement with {cpus_per_machine} cores per task")
            algo_conf = [
                "-n", str(nodes), # Number of tasks = number of nodes!
                f"--cpus-per-task={cpus_per_machine}" # the whole machine for every task!
            ]

            # Set the OpenMP env variable to use all cpus per MPI task
            env.update({
                "OMP_NUM_THREADS": str(cpus_per_machine)
            })

        matrix_path = os.path.join(os.getenv("SCRATCH"), "matrices")

        # Sbatch command with the whole config
        cmd = [
            "sbatch",
            "--wait",
            f"--time=00:{SBATCH_TIME_LIMIT_MIN}:00",
            "--constraint=mc", # Constraint to XC40
            "--switches=1", # Make sure we are in the same electircal group
            "--mem=0", # Use all available memory on the node
            "-N", str(nodes), # Number of machines to use
            "-A", "g34" # The project we use
        ] + algo_conf + [
            "--wrap",
            f"srun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {SHUFFLING} {PARTITIONING} {parallel_load_str} {persist_str} {matrix_path}"
        ]
    else:
        cmd = ["mpirun", "-n", str(nodes), CMD, impl, matrix, folder, str(N_RUNS), str(N_WARMUP), SHUFFLING, PARTITIONING, parallel_load_str, persist_str, "matrices"]
    result = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        env=env
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

def create_runs_dir(nodes, mpi) -> str:
    date = datetime.now().strftime("%Y-%m-%d-%T")
    id = f"{date}-{nodes}-{mpi}"
    cwd = os.getcwd()
    folder = join(cwd, RUNS_DIR, id)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    return folder

# Special benchmarking target for the paper
# Runs the implementation with MPI & OpenMP
def run_mpi_with_open_mp_on_daint(impl: str, matrix: str, mpi_processes: int, n_machines: int,  persist: bool = True, parallel_load: bool = True) -> str:
    global LAST_JOB
    # Calculate the distribution
    total_number_cores = n_machines * cpus_per_machine
    if impl == "comb3d":
        mpi_processes = mpi_processes*8
    assert(total_number_cores % mpi_processes == 0)
    processes_per_mpi = int(total_number_cores / mpi_processes)
    print(f"Running with {mpi_processes} mpi processes with {processes_per_mpi} cores each, on {n_machines} machines using a total of {total_number_cores} cores.")
    
    folder = create_runs_dir(n_machines, mpi_processes)
    # Environment variables for the task
    env = os.environ.copy()

    persist_str = "true" if persist else "false"
    parallel_load_str = "true" if parallel_load else "false"

    # Set the OpenMP env variable to use all cpus per MPI task
    env.update({
        "OMP_NUM_THREADS": str(processes_per_mpi)
    })

    matrix_path = os.path.join(os.getenv("SCRATCH"), "matrices")

    n_switches = 1 if n_machines <= 256 else 2 
    
    log_out = os.path.join(folder, "slurm_out.txt")
    # Sbatch command with the whole config
    cmd = [
        "sbatch",
        #"--wait",
        *([f"--dependency=afterany:{LAST_JOB}"] if LAST_JOB is not None else []),
        f"--time=03:00:00", #Make sure we definetly have enough time!
        "--constraint=mc", # Constraint to XC40
        f"--output={log_out}", # output file
        f"--switches={n_switches}", # Make sure we are in the same electircal group
        "--mem=0", # Use all available memory on the node
        "-N", str(n_machines), # Number of machines
        "-n", str(mpi_processes), # Number of tasks = MPI processes
        f"--cpus-per-task={processes_per_mpi}",
        "-A", "g34", # The project we use
        "--wrap",
        f"srun {CMD} {impl} {matrix} {folder} {N_RUNS} {N_WARMUP} {SHUFFLING} {PARTITIONING} {parallel_load_str} {persist_str} {matrix_path}"
    ]

    result = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        env=env
    )

    output = result.stdout.strip()
    error = result.stderr.strip()
    if output != "":
        print(output)
    if error != "":
        print(error)

    jobid = re.search(r"Submitted batch job (\d+)", output)
    if jobid:
        LAST_JOB = jobid.group(1) 
    else:
        print("COULD NOT GET JOB ID!!!!!")
        exit(1)

    return_code = result.returncode
    assert return_code == 0, f"Slurm job execution with {n_machines} machines failed"
    return folder

# Returns a dict of dataframes
def load_timings(folder_path: str):
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

def graph_multiple_runs(folders: List[str], daint: bool = False) -> Dict[str, pd.DataFrame]:
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

        if daint and algo not in OMP_ALGOS:
            grouped_df['nodes'] = len(node_dict) / 36
        else:
            grouped_df['nodes'] = len(node_dict)
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
    scaling_factor, data: Dict[str, pd.DataFrame], linear: bool,
    daint: bool
):
    for algo in data:
        if algo in ALGOS_TO_SKIP_WHILE_PLOTTING:
            continue

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

        ax.errorbar(timing_data["nodes"], timing_data["avg"], yerr=eb, fmt='-o', label=algo, color=color)

        if (linear):
            # Calculate a linear progression based on the first timing point
            initial_time = timing_data.iloc[0]["avg"]  # Timing for the first number of nodes
            linear_progression = initial_time * (1 / (timing_data["nodes"] / timing_data["nodes"].iloc[0]))

            # Plot the linear progression line
            ax.plot(timing_data["nodes"], linear_progression, linestyle='--', color=color, label="Linear Speedup")

def plot_timings_increasingnodes(
    data: Dict[str, pd.DataFrame], matrix: str,
    linear: bool, daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "gemm", "duration", lambda _: 10**6, data, linear, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Time vs Nodes on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "timings_plot.png"))

def plot_bytes_increasingnodes(
    data: Dict[str, pd.DataFrame], matrix: str,
    daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "bytes", "bytes", lambda _: 1, data, False, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("bytes")
    ax.set_title(f"Maximum bytes send per multiplication on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "bytes_plot.png"))

def plot_waiting_times(
    data: Dict[str, pd.DataFrame], matrix: str,
    daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "wait_all", "duration", lambda _: 10**6, data, False, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Maximum wait time per Node and communication round on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "waiting_plot.png"))

def plot_mult_times(data: Dict[str, pd.DataFrame], matrix: str, daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "mult", "duration", lambda _: 10**6, data, False, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Maximum mult time per Node and communication round on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "mult_plot.png"))

def plot_deserialize(data: Dict[str, pd.DataFrame], matrix: str, daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "deserialize", "duration", lambda _: 10**3, data, False, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (μs)")
    ax.set_title(f"Maximum deserialization time per Node and communication round on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "deserialize_plot.png"))

def plot_filter(data: Dict[str, pd.DataFrame], matrix: str, daint: bool):
    _, ax = plt.subplots()
    plot_increasingnodes(ax, "filter", "duration", lambda _: 10**3, data, False, daint)

    # Set labels, title, and legend
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time (μs)")
    ax.set_title(f"Maximum filter time per Node and communication round on {matrix}")
    ax.legend()
    plt.savefig(join(RUNS_DIR, "filter_plot.png"))

def do_plots(data: Dict[str, pd.DataFrame], matrix: str, linear: bool, daint: bool):
    plot_timings_increasingnodes(timings, matrix, args.plot_linear, daint)
    plot_bytes_increasingnodes(timings, matrix, daint)
    plot_waiting_times(timings, matrix, daint)
    plot_mult_times(timings, matrix, daint)
    plot_deserialize(timings, matrix, daint)
    plot_filter(timings, matrix, daint)

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

def prepare_daint():
    global CMD
    # Copy the executable to SCRATCH storage
    # and update the CMD to run from there
    scratch_target = os.path.join(os.getenv("SCRATCH"), "dphpc")
    print(f"Copying binary to target{scratch_target}")
    shutil.copyfile(CMD, scratch_target)
    os.system(f"chmod +x {scratch_target}")
    CMD = scratch_target

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
    parser.add_argument('--no_persist', action="store_false")
    parser.add_argument('--no_parallel_load', action="store_false")
    parser.add_argument('--plot_linear', required=False, action="store_true")
    parser.add_argument('--mpi_run_paper', required=False, action="store_true")
    args = parser.parse_args()

    assert not (args.euler and args.daint)

    folders = []

    if args.skip_run:
        # Just fetch all the folders inside run and
        # only do the visualization!
        folders = get_subfolders(RUNS_DIR)
    else:
        if args.daint:
            prepare_daint()

        # Check if multiple implementations where provided
        impls = args.impl.split(',')
        for impl in impls:
            # Special case of the paper benchmarks
            if args.mpi_run_paper:
                assert(args.daint)
                matrices = args.matrix.split(',')
                for matrix in matrices:
                    print(f"Executing implementation {impl} on {matrix}")
                    for key in MPI_OPEN_MP_CONFIG:
                        run_mpi_with_open_mp_on_daint(impl, matrix, key["mpi"], key["nodes"], args.no_persist, args.no_parallel_load)
            else:
                for n in range(args.min, args.max+1, args.stride):
                    if args.quadratic:
                        n = n*n
                    run_result = run_mpi(impl, args.matrix, n, args.euler, args.daint, args.no_persist, args.no_parallel_load)
                    if run_result != "":
                        folders.append(run_result)

        if args.mpi_run_paper:
            # We dont wait and immediately exit once the jobs have beeen scheduled
            print("All jobs scheduled!")
            exit(0)

    (matrix, timings) = graph_multiple_runs(folders, args.daint)

    # Plots to create
    do_plots(timings, matrix, args.plot_linear, args.daint)

