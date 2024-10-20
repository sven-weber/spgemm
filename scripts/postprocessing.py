import os
import argparse
import pandas as pd
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, coo_matrix

def save_matrix(matrix, filename):
  """
  Saves a matrix to a Matrix Market (.mtx) file.
  """
  try:
    mmwrite(filename, matrix)
    print(f"Matrix saved to {filename}")
  except Exception as e:
    print(f"Error saving matrix to {filename}: {e}")

def load_matrix(filename):
  """
  Loads a matrix from a Matrix Market (.mtx) file.
  """
  try:
    matrix = mmread(filename)
    return csr_matrix(matrix)
  except Exception as e:
    print(f"Error loading {filename}: {e}")
    return None

def load_shuffle_mapping(filename):
  """
  Read the shuffle file and return a DataFrame
  """
  shuffle = pd.read_csv((filename), names=['shuffle'], header=None)
  shuffle = shuffle.reset_index()
  return shuffle

def load_partition_mapping(partitions_file):
  """
  Load the partition mapping file and return a DataFrame with the information
  """
  partition_mapping = pd.read_csv(partitions_file)
  partition_mapping.set_index('rank', inplace=True)
  return partition_mapping

def load_partition(partitions_directory, rank, shuffle_mapping_A, shuffle_mapping_B, partition_mapping):
  """
  Load a partition and map relative to absolute indexes
  """
  partition_file = os.path.join(partitions_directory, f'C_{rank}.mtx')
  partition = load_matrix(partition_file).tocoo(copy=False)
  
  df = pd.DataFrame({'row': partition.row, 'col': partition.col, 'data': partition.data})
  df['row'] += partition_mapping.loc[rank, 'start_row']
  df['row'] = df['row'].map(shuffle_mapping_A['shuffle'])
  df['col'] = df['col'].map(shuffle_mapping_B['shuffle'])
  return df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Merge shuffled matrix partitions into one unshuffled matrix.')
  parser.add_argument("--source", required=True,
                      help='Folder containing partitions, partitioning info and shuffling info.')
  parser.add_argument("--nodes", required=True,
                      help='Number of nodes used for computation.')
  args = parser.parse_args()

  smA = load_shuffle_mapping(os.path.join(args.source, f'A_shuffle'))
  smB = load_shuffle_mapping(os.path.join(args.source, f'B_shuffle'))
  pm = load_partition_mapping(os.path.join(args.source, f'partitions.csv'))

  dfs = []
  for i in range(args.nodes):
    df = load_partition(args.source, i, smA, smB, pm)
    dfs.append(df)
  c = pd.concat(dfs)

  n_rows = pm['end_row'].max()
  n_cols = pm['end_col'].max()
  coo = coo_matrix((c['data'], (c['row'], c['col'])), shape=(n_rows, n_cols))
  save_matrix(coo.tocsr(), os.path.join(args.source, f'C.mtx'))

