import requests
import os
import tarfile
import tempfile
import sys
import shutil
import subprocess
import argparse
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix

formats_that_support_conversion = [
  "%%MatrixMarket matrix coordinate real general",
  "%%MatrixMarket matrix coordinate real symmetric",
  "%%MatrixMarket matrix coordinate integer symmetric",
  "%%MatrixMarket matrix coordinate integer general"
]

PYTHON_BIN = "python3"

matrices = {
  "cont-300" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/cont-300.tar.gz",
    "extract_files": ["cont-300.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("cont-300.mtx"),
    "daint_only": False,
    "compute_expected": True
  },
  "cell1" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Lucifora/cell1.tar.gz",
    "extract_files": ["cell1.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("cell1.mtx"),
    "daint_only": False,
    "compute_expected": True
  },
  "jan99jac060sc" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Hollinger/jan99jac060sc.tar.gz",
    "extract_files": ["jan99jac060sc.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("jan99jac060sc.mtx"),
    "daint_only": False,
    "compute_expected": True
  },
  "viscoplastic2" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Quaglino/viscoplastic2.tar.gz",
    "extract_files": ["viscoplastic2.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("viscoplastic2.mtx"),
    "daint_only": False,
    "compute_expected": True
  },
  # 5 million non-zeros - 160 MB uncompressed
  "largebasis": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/QLi/largebasis.tar.gz",
    "extract_files": ["largebasis.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("largebasis.mtx"),
    "daint_only": False,
    "compute_expected": True
  },
  # 50 million non-zeros - ~700 MB uncompressed
  "af_shell10": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz",
    "extract_files": ["af_shell10.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("af_shell10.mtx"),
    "daint_only": True,
    "compute_expected": True
  },
  # 440 million non-zeros
  "nlpkkt200": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt200.tar.gz",
    "extract_files": ["nlpkkt200.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("nlpkkt200.mtx"),
    "daint_only": True,
    "compute_expected": False
  },
  "nlpkkt240": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt240.tar.gz",
    "extract_files": ["nlpkkt240.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("nlpkkt240.mtx"),
    "daint_only": True,
    # TOO big for daint, we computed this outside on a bigger machine..
    "compute_expected": False
  },
  "vas_stokes_1M": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_1M.tar.gz",
    "extract_files": ["vas_stokes_1M.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("vas_stokes_1M.mtx"),
    "daint_only": True,
    "compute_expected": True
  },
  "dielFilterV2real": {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV2real.tar.gz",
    "extract_files": ["dielFilterV2real.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("dielFilterV2real.mtx"),
    "daint_only": True,
    "compute_expected": True
  },
}

TARGET_PATH = ""

def copy_one_matrix_to_A_and_B(source_name):
  global TARGET_PATH
  assert(TARGET_PATH != "")
  try:
    # Rename or move the file
    source = os.path.join(TARGET_PATH, source_name)
    print(f"Renaming matrix at {source}")
    fix_file(source)
    target_A = os.path.join(TARGET_PATH, "A.mtx")
    target_B = os.path.join(TARGET_PATH, "B.mtx")
    os.rename(source, target_A)
    shutil.copyfile(target_A, target_B)
  except Exception as e:
    print(f"Renaming {source_name} failed: {e}")
    raise e

def fix_file(target):
  # Check and correct the file header
  target_format = "%%MatrixMarket matrix coordinate real general"
  try:
    with open(target, 'r+') as file:
      first_line = file.readline()
      assert first_line.strip() in formats_that_support_conversion, f"Downloaded matrix has unsupported format {first_line}"
      # Replace first line
      file.seek(0)
      file.write(target_format)
      file.write(' ' * (len(first_line) - len(target_format) - 1))
      file.write("\n")
  except Exception as e:
    print(f"Error loading {target}: {e}")

  # Read & Write the file to ensure
  # it does not have zero values... (some do for some STUPID reason)
  matrix = mmread(target)
  # CSC to have sorting by column
  new_matrix = csc_matrix(matrix)
  new_matrix.eliminate_zeros()
  mmwrite(target, new_matrix)

def exec_subprocess(CMD, euler, daint):
  # Environment variables for the task
  env = os.environ.copy()

  if euler:
    print("Submitting job to euler via SLURM")
    # Submit this as a slurm job!
    cmd = ["sbatch", "--wait", "-n", "2", "--wrap", " ".join(CMD)]
  elif daint:
    print("Submitting job to daint via SLURM")
    cmd = [
      "sbatch",
      "--wait",
      # Running on broadwell cluster with
      # 2 sockets - 18 cores each per machine 
      "--constraint=mc", 
      "-n", "36", # 1 whole machine with two sockets
      "--mem=0", # All memory on the node!
      "-N", "1", # 1 node
      "-A", "g34", # The project we use
      "--wrap",
      " ".join(CMD)
    ]
    # Set the OpenMP env to enable parallelization
    env.update({
      "OMP_NUM_THREADS": "36"
    })
  else:
    cmd = CMD

  # Set output to unbuffered for python scripts
  # So we get output on failures.
  env.update({
    "PYTHONUNBUFFERED": "YES"
  })

  result = subprocess.run(
    cmd,
    cwd=os.getcwd(),
    capture_output=True,
    text=True,
    env=env
  )
  # Print the script output
  output = result.stdout.strip()
  print(output)
  
  return_code = result.returncode
  print(f"Execution finished with exit code {return_code}")
  assert return_code == 0, "Failed to execute subprocess"

def extract_tar_gz(name, target_folder, temp_file_path, target_files, euler, daint):
  target_files = [os.path.join(name, file) for file in target_files]
  exec_subprocess(
    [
      "tar",
      "--strip-components=1",
      "-xzf",
      str(temp_file_path),
      "-C",
      str(target_folder)
    ] + target_files,
    euler,
    daint
  )

def download_and_extract_tar_gz(name, info_dict, target_folder, euler, daint):
  url = info_dict["target-url"]
  target_files = info_dict["extract_files"]

  # Ensure the target folder exists
  if not os.path.exists(target_folder):
    os.makedirs(target_folder)

  # Create a temporary directory to store the downloaded file
  with tempfile.TemporaryDirectory(dir=target_folder) as temp_dir:
    # Extract the file name from the URL
    file_name = url.split("/")[-1]
    temp_file_path = os.path.join(temp_dir, file_name)
    
    try:
      # Download the file
      response = requests.get(url, stream=True)
      response.raise_for_status()
      
      # Write the downloaded file to the temporary directory
      with open(temp_file_path, 'wb') as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
          temp_file.write(chunk)
      
      print(f"File downloaded successfully to temporary location: {temp_file_path}")
      
      print(f"Extracting target files into folder")
      # Extract files
      extract_tar_gz(name, target_folder, temp_file_path, target_files, euler, daint)
      
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except tarfile.TarError as e:
        print(f"Error extracting tar.gz file: {e}")
   
def compute_expected(folder, euler, daint):
  exec_subprocess([PYTHON_BIN, "scripts/generate_expected.py", folder], euler, daint)

def add_to_gitignore(target, gitignore_path=".gitignore"):
  # Read the existing .gitignore file if it exists
  with open(gitignore_path, "r") as file:
    lines = file.readlines()
  
  # Check if the target already exists in the file
  if any(line.strip() == target for line in lines):
    print(f"'{target}' already exists in {gitignore_path}.")
    return
    
  # Append the new target and write the file back
  with open(gitignore_path, "a") as file:
    file.write(f"{target}\n")
  
  print(f"'{target}' has been added to {gitignore_path}.")

def get_target_folder(name, euler, daint) -> str:
  if daint:
    # On daint, we store the matrices in scratch storage!
    scratch = os.getenv("SCRATCH")
    assert(scratch != "")
    return f"{scratch}/matrices/{name}"
  else:
    return f"matrices/{name}"


def main(euler: bool, daint: bool):
  global TARGET_PATH
  print("Fetching missing matrices")
  for name, dict in matrices.items():
    TARGET_PATH = get_target_folder(name, euler, daint)
    assert(TARGET_PATH != "")
    print(f"Downloading to target path: {TARGET_PATH}")
    if os.path.exists(TARGET_PATH) and name:
      print(f"Skipped {name} since it already exists.")
    elif dict["daint_only"] == True and daint == False:
      print(f"Skipped {name} since it should only be computed on daint")
    else:
      print(f"-------- {name} --------")
      download_and_extract_tar_gz(name, dict, TARGET_PATH, euler, daint)
      # Execute post extract func
      if dict["post_extract_func"] is not None:
        dict["post_extract_func"]()
      # Compute expected value
      if dict["compute_expected"] == True:
        print("Computing expected output")
        compute_expected(TARGET_PATH, euler, daint)
      else:
        print("Skipped expected value computation")
      print("Processing finished.")
      if not daint:
        # Daint uses scratch storage that is not part of the repo!
        add_to_gitignore(TARGET_PATH)
      print(f"-------- {name} --------")
  print("Fetching done.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='fetch_matrices',
    description='Fetch matrices to compute')
  parser.add_argument('--euler', action="store_true")
  parser.add_argument('--daint', action="store_true")
  args = parser.parse_args()

  assert not (args.euler and args.daint)
  main(args.euler, args.daint)


