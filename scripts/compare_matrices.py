import argparse
import numpy as np
import os
from scipy.io import mmread
from scipy.sparse import csr_matrix

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

def compare_matrices(C_expected, C_real, tolerance=1e-10):
    """
    Compares two sparse matrices (C_expected and C_real).
    Prints the positions (i, j) where the values differ beyond a given tolerance.
    """
    if C_expected.shape != C_real.shape:
        print(f"Matrix shape mismatch: C_expected {C_expected.shape}, C_real {C_real.shape}")
        return
    

    close_values = np.isclose(C_expected.data, C_real.data, atol=tolerance)
    not_close_values = ~close_values
    differences = np.sum(not_close_values)
    
    # expected_row_indices, expected_col_indices = C_expected.nonzero()
    # differences = 0
    
    # for i, j in zip(expected_row_indices, expected_col_indices):
    #     expected_value = C_expected[i, j]
    #     real_value = C_real[i, j]
    #     if not np.isclose(expected_value, real_value, atol=tolerance):
    #         differences += 1
    #         print(f"Mismatch at position ({i}, {j}): expected {expected_value}, got {real_value}")
    
    # real_row_indices, real_col_indices = C_real.nonzero()
    # for i, j in zip(real_row_indices, real_col_indices):
    #     if (i, j) not in zip(expected_row_indices, expected_col_indices):
    #         print(f"Extra entry in C_real at position ({i}, {j}): got {C_real[i, j]}")

    if differences == 0:
        print("The matrices are identical within the given tolerance.")
    else:
        print(f"Total differences found: {differences}")

def process_directory(directory):
    """
    Processes each subdirectory to compare C_expected and C_real matrices.
    """
    for root, _, files in os.walk(directory):
        C_expected_file = None
        C_real_file = None
        
        for file in files:
            if file == "C_expected.mtx":
                C_expected_file = os.path.join(root, file)
            elif file == "C.mtx":
                C_real_file = os.path.join(root, file)
        
        if C_expected_file and C_real_file:
            print(f"\nComparing {C_expected_file} and {C_real_file}...")
            C_expected = load_matrix(C_expected_file)
            C_real = load_matrix(C_real_file)
            
            if C_expected is None or C_real is None:
                print("Error loading matrices.")
                continue

            compare_matrices(C_expected, C_real)

def main():
    parser = argparse.ArgumentParser(description="Compare matrices from .mtx files.")
    parser.add_argument(
        "file1",
        nargs="?",
        help="Path to the first matrix file (C_expected.mtx).",
        default=None
    )
    parser.add_argument(
        "file2",
        nargs="?",
        help="Path to the second matrix file (C.mtx).",
        default=None
    )

    args = parser.parse_args()

    if args.file1 and args.file2:
        print(f"Comparing {args.file1} and {args.file2}...")
        C_expected = load_matrix(args.file1)
        C_real = load_matrix(args.file2)

        if C_expected is None or C_real is None:
            print("Error loading matrices.")
        else:
            compare_matrices(C_expected, C_real)
    else:
        process_directory("matrices")

if __name__ == "__main__":
    main()
