import re
import sys
import os

def calculate_average(file_path):
    try:
        parent_folder = os.path.basename(os.path.dirname(file_path))
        print(f"Title: {parent_folder}")

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Regex to match the desired pattern and extract the floating-point value
        pattern = r"Bitmap drop percentage:\s+(\d+\.\d+)"

        values = []

        for line in lines:
            match = re.search(pattern, line)
            if match:
                values.append(float(match.group(1)))

        if values:
            average = sum(values) / len(values)
            print(f"Average Bitmap drop percentage: {average:.3f}")
            print(f"Number of lines considered: {len(values)}")
        else:
            print("No matching lines found in the file.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    calculate_average(file_path)
