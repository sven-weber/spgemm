def write_mtx_file(output_file, full_matrix, num_rows, num_cols, tot_entries):
    """
    Write the full matrix in Matrix Market format to the output file.
    """
    with open(output_file, 'w') as file:
        file.write('%%MatrixMarket matrix coordinate integer general\n')
        file.write(f'{num_rows} {num_cols} {tot_entries}\n')
        for row, col, value in full_matrix:
            file.write(f'{row} {col} {value}\n')

def read_mtx_file(filename):
    """
    Read a .mtx file and return a list of tuples (row, col, value).
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    matrix_data = [line.strip() for line in lines if not line.startswith('%')]
    nrows, ncols, nentires = matrix_data[0].split()

    elements = list()
    for element_line in matrix_data[1:]:
        row, col, value = map(int, element_line.split())
        elements.append((row, col, value))
    
    return elements, (nrows, ncols, nentires)

def load_shuffle(filename):
    """
    Read the shuffle file and return a list with the inverse mapping.
    """
    with open(filename, 'r') as file:
        shuffle = [int(line.strip()) for line in file.readlines()]
    
    inverse_shuffle = [0] * len(shuffle)
    for original_index, shuffled_index in enumerate(shuffle):
        inverse_shuffle[shuffled_index] = original_index
    
    return inverse_shuffle

def unshuffle_matrix(C, inverse_shuffleA, inverse_shuffleB):
    """
    Apply the inverse row and column shuffling to the matrix C.
    """
    unshuffled_C = list()
    for row, col, value in C:
        original_row = inverse_shuffleA[row - 1] + 1
        original_col = inverse_shuffleB[col - 1] + 1
        unshuffled_C.append((original_row, original_col, value))

    return unshuffled_C

def main(input_file, output_file):

    inverse_shuffleA = load_shuffle('matrices/test/shuffle_A.txt')
    inverse_shuffleB = load_shuffle('matrices/test/shuffle_B.txt')
    C_entries, header = read_mtx_file(input_file)

    total_rows, total_cols, total_entries = header

    unshuffled_C = unshuffle_matrix(C_entries, inverse_shuffleA, inverse_shuffleB)

    sorted_unshuffle_C = sorted(unshuffled_C, key=lambda x: x[1])

    write_mtx_file(output_file, sorted_unshuffle_C, total_rows, total_cols, total_entries)

if __name__ == "__main__":

    input_file = 'matrices/test/C_expected.mtx'
    output_file = 'matrices/test/C_unshuffled.mtx'
    main(input_file, output_file)
