import os
import numpy as np
import pandas as pd

def load_partition_mapping(partitions_file):
    """
    Load the partition mapping file and return a DataFrame with the information sorted by start_row, 
    in order to properly reconstruct C
    """
    partition_mapping = pd.read_csv(partitions_file)
    partition_mapping = partition_mapping.sort_values(by='start_row').reset_index(drop=True)
    return partition_mapping

def read_mtx_file(filename):
    """
    Read a .mtx file and return a list of tuples (row, col, value).
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    matrix_data = [line.strip() for line in lines if not line.startswith('%')]

    elements = list()
    for element_line in matrix_data[1:]:
        row, col, value = map(int, element_line.split())
        elements.append((row, col, value))
    
    return elements

def calculate_matrix_dimensions(partition_mapping, partitions_directory):
    """
    Calculate the total number of rows, columns, and non-zero entries by summing up information
    from the headers of each partition file.
    """
    total_rows = 0
    num_cols = 0
    total_entries = 0
    cols = list()

    for _, row in partition_mapping.iterrows():
        rank = row['rank']
        partition_file = os.path.join(partitions_directory, f'C_{rank}.mtx')
        with open(partition_file, 'r') as file:
            lines = file.readlines()
            header = [line.strip() for line in lines if not line.startswith('%')][0]
            r, c, e = map(int, header.split())
            cols.append(c)
        total_rows += r
        num_cols = max(cols)
        total_entries += e
    return total_rows, num_cols, total_entries

def assemble_full_matrix(partition_mapping, partitions_directory):
    """
    Reconstruct the full matrix C from the individual partitions.
    """
    full_matrix = list()

    for _, row in partition_mapping.iterrows():
        rank = row['rank']
        start_row = row['start_row']
        
        partition_file = os.path.join(partitions_directory, f'C_{rank}.mtx')
        partition_elements = read_mtx_file(partition_file)
        
        for (local_row_index, col_index, value) in partition_elements:
            adjusted_row = start_row + local_row_index
            full_matrix.append((adjusted_row, col_index, value))
    
    full_matrix = sorted(full_matrix, key=lambda x: x[1])
    return full_matrix

def write_mtx_file(output_file, full_matrix, num_rows, num_cols, total_entries):
    """
    Write the full matrix in Matrix Market format to the output file.
    """
    with open(output_file, 'w') as file:
        file.write('%%MatrixMarket matrix coordinate integer general\n')
        file.write(f'{num_rows} {num_cols} {total_entries}\n')
        for row, col, value in full_matrix:
            file.write(f'{row} {col} {value}\n')

def main(partitions_file, partitions_directory, output_file):
    
    partition_mapping = load_partition_mapping(partitions_file)
    num_cols = None

    full_matrix = assemble_full_matrix(partition_mapping, partitions_directory)

    total_rows, num_cols, total_entries = calculate_matrix_dimensions(partition_mapping, partitions_directory)

    write_mtx_file(output_file, full_matrix, total_rows, num_cols, total_entries)

if __name__ == "__main__":
    
    partitions_file = 'matrices/test/partitions_order.csv'
    partitions_directory = 'matrices/test'
    output_file = 'matrices/test/C_expected.mtx'

    main(partitions_file, partitions_directory, output_file)
