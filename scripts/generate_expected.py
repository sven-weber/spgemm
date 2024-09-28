import os
from scipy.io import mmread, mmwrite
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

def save_matrix(matrix, filename):
    """
    Saves a matrix to a Matrix Market (.mtx) file.
    """
    try:
        mmwrite(filename, matrix)
        print(f"Matrix saved to {filename}")
    except Exception as e:
        print(f"Error saving matrix to {filename}: {e}")

def process_directory(directory):
    """
    Processes each subdirectory to compute the expected matrix C = A * B.
    """
    for root, _, files in os.walk(directory):
        A_file = None
        B_file = None
        
        for file in files:
            if file == "A.mtx":
                A_file = os.path.join(root, file)
            elif file == "B.mtx":
                B_file = os.path.join(root, file)
        
        if A_file and B_file:
            print(f"Processing {A_file} and {B_file}...")
            A = load_matrix(A_file)
            B = load_matrix(B_file)
            
            if A is None or B is None:
                print("Error loading matrices.")
                continue

            C_expected = A.dot(B)
            save_matrix(C_expected, os.path.join(root, "C_expected.mtx"))

def main():
    process_directory("matrices")

if __name__ == "__main__":
    main()
