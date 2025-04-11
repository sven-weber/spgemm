from pathlib import Path
from os import listdir
from typing import List, Tuple, Dict

import pandas as pd

Dataframes = Dict[int, pd.DataFrame]

def get_subfolders(folder_path: Path) -> List[Path]:
    subfolders = []
    for item in listdir(folder_path):
        item_path = folder_path / item
        if item_path.is_dir():
            subfolders.append(item_path)
    subfolders.sort()
    return subfolders

def load_algorithm(folder_path: Path) -> str:
    with open(folder_path / "algo", "r") as f:
        return f.read().strip()

def load_nodes_mpi(folder_path: Path) -> Tuple[int, int]:
    if '_' in folder_path.name:
        parts = folder_path.name.split('_')
        return int(parts[0]), int(parts[1])    
    else:
        parts = folder_path.name.split('-')
        return int(parts[-2]), int(parts[-1])

def load_matrix(folder_path: Path) -> str:
    matrix_path = folder_path / "matrix"
    return matrix_path.read_text().strip()

def load_node_id(measurement_path: Path) -> int:
    return int(measurement_path.name.split("_")[1].split(".")[0])

# Loads all the timings (measurement) for all nodes of a given run
def load_timings(folder_path: Path) -> Dataframes:
    dataframes = {}
    measurement_paths = [
        path 
        for path in folder_path.iterdir() 
        if path.name.endswith(".csv") and path.name.startswith("measurements")
    ]
    for file in measurement_paths:
        node_id = load_node_id(file)
        df = pd.read_csv(file)
        
        df['node_id'] = node_id
        first_gemm_idx = df[df['func'] == 'gemm'].index.min()
        df = df.drop(
            df[(df.index < first_gemm_idx) & (df['func'].isin(['mult', 'deserialize', 'wait_all']))].index
        )
        dataframes[node_id] = df
    return load_duration_as_numeric(dataframes)

# Because there may be multiple function names in the same file
def load_duration_as_numeric(dfs: Dataframes) -> Dataframes:
    dataframes = {}
    for node_id, df in dfs.items():
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        dataframes[node_id] = df
    return dataframes

# Groups the measurement of all nodes of one run together to
# get the maximum for each line in the CSV (the slowest node)
def group_runs(node_dict: Dataframes) -> pd.DataFrame:
    if len(node_dict) == 0:
        return pd.DataFrame()

    # Find all the functions that exists in the files
    functions: List[str] = []
    for _, df in node_dict.items():
        functions += df["func"].unique().tolist()
    functions = list(set(functions))

    # For each of the functions, do a grouping!
    result = []
    for function in functions:
        groups = {}
        for node_id, df in node_dict.items():
            df = df[node_dict[node_id]["func"] == function]
            # This ensures the functions are in the same order, no matter
            # which order they where in in the original file
            groups[node_id] = df.reset_index(drop=True)
        
        # Group by the same index!
        result.append(pd.concat(groups).groupby(level=1).max())
        
    # Return new dataframe with all the functions grouped!
    return pd.concat(result)

def load_multiple_timings(folders: List[Path], matrix: str) -> Tuple[str, pd.DataFrame]:
    timings_per_algo = {}
    for folder_path in folders:
        # Load algo and initialize
        if load_matrix(folder_path) != matrix:
            continue

        algo = load_algorithm(folder_path)
        if not algo in timings_per_algo:
            timings_per_algo[algo] = pd.DataFrame()

        # Load DFs
        node_dict = load_timings(folder_path)
        grouped_df = group_runs(node_dict)

        nodes, mpi = load_nodes_mpi(folder_path)
        grouped_df['nodes'] = nodes
        grouped_df['mpi'] = mpi
        timings_per_algo[algo] = pd.concat([timings_per_algo[algo], grouped_df], axis=0)
    
    if matrix is None:
        matrixes = [load_matrix(folder) for folder in folders]
        assert all([m1 == m2 for m1 in matrixes for m2 in matrixes])
    return (matrixes[0] if matrix is None else matrix, timings_per_algo)

def load_multiple_timings_scalability(folders: List[Path]) -> Tuple[str, pd.DataFrame]:
    timings_per_algo = {}
    for folder_path in folders:
        # Load algo and initialize
        key = folder_path.parent.name
        if not key in timings_per_algo:
            timings_per_algo[key] = pd.DataFrame()

        # Load DFs
        node_dict = load_timings(folder_path)
        grouped_df = group_runs(node_dict)

        nodes, mpi = load_nodes_mpi(folder_path)
        grouped_df['nodes'] = nodes
        grouped_df['mpi'] = mpi
        timings_per_algo[key] = pd.concat([timings_per_algo[key], grouped_df], axis=0)
    
    return timings_per_algo