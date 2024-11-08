import numpy as np
import random
from scipy.optimize import linear_sum_assignment

# Key coordinates from the provided layout
keyboard_layout = {
    'h': [492, 1403, 590, 1576],
    'e': [196, 1230, 294, 1403],
    'l': [787, 1403, 886, 1576],
    'o': [689, 1230, 787, 1403],
    'inputbox': [0, 130, 1080, 224],
    'backspace': [886, 1576, 1080, 1749]
}


def calculate_center(coords):
    x1, y1, x2, y2 = coords
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def add_noise(coord, noise_level=2):
    # int noise
    x, y = coord
    x += random.randint(-noise_level, noise_level)
    y += random.randint(-noise_level, noise_level)
    return int(x), int(y)


def generate_gaze_sequence(sequence, layout, noise_level=2):
    gaze_sequence = []
    for char in sequence:
        center = calculate_center(layout[char])
        noisy_center = add_noise(center, noise_level)
        # normalize the duration to seconds
        duration = random.randint(100, 300) / 1000  # Random duration between 100ms and 300ms
        gaze_sequence.append((*noisy_center, duration))

    return gaze_sequence


def multi_match(X, Y, dimensions=2, return_components=False):
    if dimensions not in [2, 3]:
        raise ValueError('Invalid number of dimensions')

    # Split the scanpaths into individual components
    X = np.array(X)
    Y = np.array(Y)
    Xpos = X[:, :dimensions]
    Ypos = Y[:, :dimensions]
    Xdur = X[:, dimensions:dimensions + 1]
    Ydur = Y[:, dimensions:dimensions + 1]

    # Component-wise similarity
    pos_sim = _compute_position_similarity(Xpos, Ypos)
    dur_sim = _compute_duration_similarity(Xdur, Ydur)

    if return_components:
        return (pos_sim + dur_sim) / 2, pos_sim, dur_sim
    return (pos_sim + dur_sim) / 2


def _compute_position_similarity(X, Y):
    # normalize the x, y coordinates to [0, 1]
    X = X / np.max(X)
    Y = Y / np.max(Y)
    distance_matrix = np.linalg.norm(X[:, None] - Y[None, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    similarity = np.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


def _compute_duration_similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    # normalize the x, y coordinates to [0, 1]
    distance_matrix = np.abs(X[:, None] - Y[None, :])
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    similarity = np.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


# Define the sequence
sequence = ['h', 'e', 'l', 'l', 'l', 'inputbox', 'backspace', 'o']

# Generate two sequences for comparison
seq1 = generate_gaze_sequence(sequence, keyboard_layout, noise_level=20)
seq2 = generate_gaze_sequence(sequence, keyboard_layout, noise_level=20)

# Print the generated sequences
print("Sequence 1 (x, y, duration):")
for point in seq1:
    print(point)

print("\nSequence 2 (x, y, duration):")
for point in seq2:
    print(point)


# Prepare the sequences for multi_match function
def prepare_sequence_for_multimatch(gaze_sequence):
    return np.array([[x, y, duration] for (x, y, duration) in gaze_sequence])


seq1_prepared = prepare_sequence_for_multimatch(seq1)
seq2_prepared = prepare_sequence_for_multimatch(seq2)

# Calculate the similarity score
overall_similarity, pos_sim, dur_sim = multi_match(seq1_prepared, seq2_prepared, return_components=True)
print(f"\nPosition similarity: {pos_sim}")
print(f"Duration similarity: {dur_sim}")
print(f"\nOverall similarity score between the two sequences: {overall_similarity}")
