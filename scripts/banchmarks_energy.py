import os
import re

def extract_energy_from_slurm_file(file_path):
    """
    Extracts the energy value in kJ from a slurm.out file.
    Returns the energy as a float if found, otherwise None.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Match any floating-point number followed by 'kJ' (e.g., "2529.932 kJ")
                match = re.search(r'(\d+\.\d+)\s+kJ', line)
                if match:
                    return float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def sum_energy_in_slurm_files(base_dir):
    """
    Recursively searches for 'slurm.out' files and sums up the energy values.
    Also counts the total number of files processed.
    """
    total_energy = 0.0
    file_count = 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "slurm_out.txt":
                file_count += 1
                file_path = os.path.join(root, file)
                energy = extract_energy_from_slurm_file(file_path)
                if energy is not None:
                    total_energy += energy

    return total_energy, file_count

if __name__ == "__main__":
    base_directory = os.getcwd()  # Start from the current working directory
    total_energy_sum, total_files = sum_energy_in_slurm_files(base_directory)

    # Convert kJ to watt-hours (Wh)
    total_energy_wh = total_energy_sum * 0.000277778

    print("------------------------------------------------")
    print(f"Number of Files Processed: {total_files}")
    print(f"Total Energy Sum: {total_energy_sum:.3f} kJ")
    print(f"Total Energy Sum: {total_energy_wh:.3f} KWh")
    print("------------------------------------------------")
