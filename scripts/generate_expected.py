import os
import sys
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
import shutil

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
          process_matrices(root)

def process_matrices(folder):
  A_file = os.path.join(folder, "A.mtx")
  B_file = os.path.join(folder, "B.mtx")
  
  print(f"Processing {A_file} and {B_file}...")
  A = load_matrix(A_file)
  B = load_matrix(B_file)
  
  if A is None or B is None:
    print("Error loading matrices.")
    sys.exit(1)

  C_expected = A.dot(B)
  C_expected_file = os.path.join(folder, "C_expected.mtx")
  save_matrix(C_expected, C_expected_file)
  # Copy as sparsity matrix (for now)
  # TODO: Compute smth we actual need
  shutil.copyfile(C_expected_file,  os.path.join(folder, "C_sparsity.mtx"))

def main():
  if len(sys.argv) <= 1:
    # Process the whole dir
    process_directory("matrices")
  else:
    # Process a specific dir that is provided!
    process_matrices(sys.argv[1])

if __name__ == "__main__":
  main()
