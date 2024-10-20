import re
import sys
import subprocess
import os
from datetime import datetime

def fetch_matrix_target():
  with open("Makefile", 'r') as file:
    content = file.read()

    # Use regex to find the MATRIX_TARGET parameter
    match = re.search(r'^\s*MATRIX_TARGET\s*=\s*(.+)', content, re.MULTILINE)
    if match:
      return match.group(1).strip()  # Return the value without leading/trailing whitespace
    else:
      print("Target not found in MAKE!")
      sys.exit(1)

def fetch_num_machines():
  with open("Makefile", 'r') as file:
    content = file.read()

    # Use regex to find the MATRIX_TARGET parameter
    match = re.search(r'^\s*TEST_MACHINES\s*=\s*(.+)', content, re.MULTILINE)
    if match:
      return match.group(1).strip()  # Return the value without leading/trailing whitespace
    else:
      print("Target not found in MAKE!")
      sys.exit(1)
          
def find_latest_folder():
    latest_folder = None
    latest_time = None
    directory = "runs"

    for folder in os.listdir(directory):
      folder_path = os.path.join(directory, folder)
      if os.path.isdir(folder_path):
        try:
          # Convert folder name to datetime object
          folder_time = datetime.strptime(folder, "%Y-%m-%d-%H:%M:%S")
          if latest_time is None or folder_time > latest_time:
            latest_time = folder_time
            latest_folder = folder_path
        except ValueError:
          # Skip folders that don't match the expected date format
          continue

    return latest_folder

def run_postprocessing(folder, num_machines):
  result = subprocess.run(
      ["python", "scripts/postprocessing.py", "--source", folder, "--nodes", num_machines],
      capture_output=True,
      text=True
  )

  # Print the script output
  output = result.stdout.strip()
  error = result.stderr.strip()
  print(output)
  if error != "":
    print(error)
  
  return_code = result.returncode
  assert return_code == 0, "Running postprocessing failed!"

def run_comparison(source_matrix, target_matrix):
  result = subprocess.run(
      ["python", "scripts/compare_matrices.py", source_matrix, target_matrix],
      capture_output=True,
      text=True
  )

  # Print the script output
  output = result.stdout.strip()
  error = result.stderr.strip()
  print(output)
  if error != "":
    print(error)

  return_code = result.returncode
  assert return_code == 0, "Run comparison failed!"

def main():
  run_target = find_latest_folder()
  matrix_target = fetch_matrix_target()
  num_machines = fetch_num_machines()
  print("Running postprocessing")
  run_postprocessing(run_target, num_machines)
  print("Running comparison")
  run_comparison(os.path.join("matrices", matrix_target, "C_expected.mtx"), os.path.join(run_target, "C.mtx"))

if __name__ == "__main__":
  main()
