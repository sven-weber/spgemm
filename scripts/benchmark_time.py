import pandas as pd
import argparse
import os


def load_timings(folder_path):
    dataframes = dict()
    for file in os.listdir(folder_path):
        if file.startswith("timing") and file.endswith(".csv"):
            node_id = file.split("_")[1].split(".")[0]
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dataframes[node_id] = df
    return dataframes

# Because there may be multiple function names in the same file
def aggregate_vertically(dataframes):
    for node_id, df in dataframes.items():
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df = df.groupby("func").mean().reset_index()
        dataframes[node_id] = df
    return dataframes

# To have the average, min and max for each function across the different files (each file is one node)
def aggregate_horizontally(dataframes):
    all_func_names = pd.concat([df["func"] for df in dataframes.values()]).unique()
    aggregated_data = list()
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


def main(folder_path):

    # Load timings
    dataframes = load_timings(folder_path)

    # Aggregate vertically timings related to same function
    dataframes = aggregate_vertically(dataframes)

    # Aggregate horizontally timings related to same function
    aggregated_df = aggregate_horizontally(dataframes)
    print(aggregated_df)


if __name__ == "__main__":
    folder_path = "runs/2024-10-20-10:23:07"
    main(folder_path)

   