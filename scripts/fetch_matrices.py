import requests
import os
import tarfile
import tempfile
import sys
import shutil
import subprocess
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix

formats_that_support_conversion = [
  "%%MatrixMarket matrix coordinate real general",
  "%%MatrixMarket matrix coordinate real symmetric",
  "%%MatrixMarket matrix coordinate integer symmetric",
  "%%MatrixMarket matrix coordinate integer general"
]

matrices = {
  "cont-300" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/cont-300.tar.gz",
    "extract_files": ["cont-300.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("cont-300", "cont-300.mtx")
  },
  "cell1" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Lucifora/cell1.tar.gz",
    "extract_files": ["cell1.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("cell1", "cell1.mtx")
  },
  "jan99jac060sc" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Hollinger/jan99jac060sc.tar.gz",
    "extract_files": ["jan99jac060sc.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("jan99jac060sc", "jan99jac060sc.mtx")
  },
  "viscoplastic2" : {
    "target-url": "https://suitesparse-collection-website.herokuapp.com/MM/Quaglino/viscoplastic2.tar.gz",
    "extract_files": ["viscoplastic2.mtx"],
    "post_extract_func": lambda: copy_one_matrix_to_A_and_B("viscoplastic2", "viscoplastic2.mtx")
  },
}

def copy_one_matrix_to_A_and_B(folder, source_name):
  try:
    # Rename or move the file
    source = os.path.join("matrices", folder, source_name)
    fix_file(source)
    target_A = os.path.join("matrices", folder, "A.mtx")
    target_B = os.path.join("matrices", folder, "B.mtx")
    os.rename(source, target_A)
    shutil.copyfile(target_A, target_B)
  except Exception as e:
    print(f"Renaming {source_name} failed: {e}")

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

def download_and_extract_tar_gz(info_dict, target_folder):
  url = info_dict["target-url"]
  target_files = info_dict["extract_files"]

  # Create a temporary directory to store the downloaded file
  with tempfile.TemporaryDirectory() as temp_dir:
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
      
      # Ensure the target folder exists
      if not os.path.exists(target_folder):
        os.makedirs(target_folder)
      
      # Extract the .tar.gz file to the target folder
      with tarfile.open(temp_file_path, 'r:gz') as tar:
        # Extract only the files we need
        members = [member for member in tar.getmembers() if member.name.split("/")[-1] in target_files]

        if members:
            # Extract only the specific file(s) to the target folder
            for member in members:
              # Remove the directory structure from the member
              member.name = os.path.basename(member.name)
              tar.extract(member, path=target_folder)
        else:
            print("Failed to find any of the files in the archive:")
            print(target_files)
            sys.exit(1)
      
      print(f"File extracted successfully to {target_folder}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except tarfile.TarError as e:
        print(f"Error extracting tar.gz file: {e}")
   
def compute_expected(folder):
  result = subprocess.run(
    ["python", "scripts/generate_expected.py", folder],
    capture_output=True,
    text=True
  )

  # Print the script output
  output = result.stdout.strip()
  print(output)
  
  return_code = result.returncode
  assert return_code == 0, "Failed to generate expected values"

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

if __name__ == "__main__":
  print("Fetching missing matrices")
  for name, dict in matrices.items():
    target_path = f"matrices/{name}"
    if not os.path.exists(target_path):
      print(f"-------- {name} --------")
      download_and_extract_tar_gz(dict, target_path)
      # Execute post extract func
      if dict["post_extract_func"] is not None:
        dict["post_extract_func"]()
      # Compute expected matrix
      print("Computing expected output")
      compute_expected(target_path)
      print("Processing finished.")
      add_to_gitignore(target_path)
      print(f"-------- {name} --------")
    else:
      print(f"Skipped {name} since it already exists.")
  print("Fetching done.")
