import os
import matplotlib.pyplot as plt


def plot_shuffling_iterations(file_path):
    # Extract folder and file name details
    folder, file_name = os.path.split(file_path)
    file_stem, _ = os.path.splitext(file_name)
    plot_path = os.path.join(folder, f"{file_stem}.png")

    # Read the data
    iterations = []
    values = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # Ignore empty lines
                    try:
                        parts = line.split(":")
                        iteration = int(parts[0].split()[-1])
                        value = float(parts[1])
                        iterations.append(iteration)
                        values.append(value)
                    except (ValueError, IndexError):
                        # ignore invalid lines
                        continue
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Plot the data
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, values, marker="o", linestyle="-", color="b")
        plt.title(f"Shuffling Iteration Values - {file_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved as {plot_path}")
    except Exception as e:
        print(f"Error creating the plot: {e}")


# If running as a script
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_data_file>")
    else:
        plot_shuffling_iterations(sys.argv[1])
