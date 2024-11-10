import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw
from scipy.spatial.distance import directed_hausdorff, euclidean
from scipy.spatial.distance import cdist

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
    trailtime = 0
    gaze_sequence = []
    for char in sequence:
        center = calculate_center(layout[char])
        noisy_center = add_noise(center, noise_level)
        # Normalize the duration to seconds
        duration = random.randint(100, 300)  # Random duration between 100ms and 300ms
        gaze_sequence.append((*noisy_center, duration))

    # Normalize the trailtime to (0, 1)
    max_trailtime = trailtime
    # normalized_gaze_sequence = [(x, y, tt / max_trailtime) for (x, y, tt) in gaze_sequence]
    normalized_gaze_sequence = gaze_sequence
    return normalized_gaze_sequence


def multi_match(X, Y, dimensions=2, return_components=False, include_duration=True):
    if dimensions not in [2, 3]:
        raise ValueError('Invalid number of dimensions')

    # Split the scanpaths into individual components
    X = np.array(X)
    Y = np.array(Y)
    Xpos = X[:, :dimensions]
    Ypos = Y[:, :dimensions]
    if include_duration:
        Xdur = X[:, dimensions:dimensions + 1]
        Ydur = Y[:, dimensions:dimensions + 1]

    # Component-wise similarity
    pos_sim = _compute_position_similarity(Xpos, Ypos)
    if include_duration:
        dur_sim = _compute_trailtime_similarity(Xdur, Ydur)
    else:
        dur_sim = 0

    shape_sim = _compute_shape_similarity(Xpos, Ypos)
    direction_sim = _compute_direction_similarity(Xpos, Ypos)
    length_sim = _compute_length_similarity(Xpos, Ypos)

    # total_sim = (pos_sim + dur_sim) / 2
    total_sim = (pos_sim + dur_sim + shape_sim + direction_sim + length_sim) / 5
    if return_components:
        return total_sim, pos_sim, dur_sim, shape_sim, direction_sim, length_sim
    return total_sim


def _compute_position_similarity(X, Y):
    # normalize the x, y coordinates to [0, 1]
    X = X / np.max(X)
    Y = Y / np.max(Y)
    distance_matrix = np.linalg.norm(X[:, None] - Y[None, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    similarity = np.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


def _compute_trailtime_similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    # normalize the x, y coordinates to [0, 1]
    max_trailtime = max(X.max(), Y.max())
    X = X / max_trailtime
    Y = Y / max_trailtime
    distance_matrix = np.abs(X[:, None] - Y[None, :])
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    similarity = np.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


def _compute_shape_similarity(X, Y):
    X_vectors = np.diff(X, axis=0)
    Y_vectors = np.diff(Y, axis=0)
    X_angles = np.arctan2(X_vectors[:, 1], X_vectors[:, 0])
    Y_angles = np.arctan2(Y_vectors[:, 1], Y_vectors[:, 0])
    angle_diff = np.abs(X_angles[:, None] - Y_angles[None, :])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    row_ind, col_ind = linear_sum_assignment(angle_diff)
    similarity = np.exp(-angle_diff[row_ind, col_ind].mean())
    return similarity


def _compute_direction_similarity(X, Y):
    X_direction = X[-1] - X[0]
    Y_direction = Y[-1] - Y[0]
    X_norm = X_direction / np.linalg.norm(X_direction)
    Y_norm = Y_direction / np.linalg.norm(Y_direction)
    dot_product = np.dot(X_norm, Y_norm)
    similarity = (dot_product + 1) / 2  # scale to [0, 1]
    return similarity


def _compute_length_similarity(X, Y):
    X_lengths = np.linalg.norm(np.diff(X, axis=0), axis=-1)
    Y_lengths = np.linalg.norm(np.diff(Y, axis=0), axis=-1)
    X_total = X_lengths.sum()
    Y_total = Y_lengths.sum()
    length_diff = np.abs(X_total - Y_total)
    max_length = max(X_total, Y_total)
    similarity = np.exp(-length_diff / max_length)
    return similarity


# Prepare the sequences for multi_match function
def prepare_sequence_for_multimatch(gaze_sequence):
    return np.array([[x, y, trailtime] for (x, y, trailtime) in gaze_sequence])


# def dtw_scanpaths(X, Y, dimensions=2, include_duration=True):
#     if dimensions not in [2, 3]:
#         raise ValueError('Invalid number of dimensions')
#
#     # Split the scanpaths into individual components
#     X = np.array(X)
#     Y = np.array(Y)
#     Xpos = X[:, :dimensions]
#     Ypos = Y[:, :dimensions]
#
#     Xpos = Xpos / np.max(Xpos)
#     Ypos = Ypos / np.max(Ypos)
#
#     if include_duration:
#         Xdur = X[:, dimensions:dimensions + 1].flatten()
#         Ydur = Y[:, dimensions:dimensions + 1].flatten()
#         max_trailtime = max(Xdur.max(), Ydur.max())
#         Xdur = Xdur / max_trailtime
#         Ydur = Ydur / max_trailtime
#
#     # Compute DTW for spatial components
#     spatial_dtw_result = dtw(Xpos, Ypos)
#     spatial_dtw_distance = spatial_dtw_result.distance
#
#     if include_duration:
#         # Compute DTW for temporal components
#         temporal_dtw_result = dtw(Xdur, Ydur)
#         temporal_dtw_distance = temporal_dtw_result.distance
#     else:
#         temporal_dtw_distance = 0
#
#     # Combine spatial and temporal DTW distances
#     total_dtw_distance = (spatial_dtw_distance + temporal_dtw_distance) / 2
#
#     return total_dtw_distance

def normalize_scanpath(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    Xpos = X[:, :2]
    Ypos = Y[:, :2]

    Xpos = Xpos / np.max(Xpos)
    Ypos = Ypos / np.max(Ypos)
    Xdur = X[:, 2:3].flatten()
    Ydur = Y[:, 2:3].flatten()
    # max_trailtime = max(Xdur.max(), Ydur.max())
    max_trailtime = Ydur.max()
    Xdur = Xdur / max_trailtime
    Ydur = Ydur / max_trailtime

    X = np.concatenate((Xpos, Xdur[:, None]), axis=1)
    Y = np.concatenate((Ypos, Ydur[:, None]), axis=1)
    return X, Y


def dtw_scanpaths(X, Y):
    dimensions = X.shape[1]
    if dimensions not in [2, 3]:
        raise ValueError('Invalid number of dimensions')
    X = np.array(X)
    Y = np.array(Y)
    Xpos = X[:, :2]
    Ypos = Y[:, :2]

    Xpos = Xpos / np.max(Xpos)
    Ypos = Ypos / np.max(Ypos)
    if dimensions == 2:
        X = Xpos
        Y = Ypos
    else:
        X, Y = normalize_scanpath(X, Y)
    dist, _ = fastdtw(X, Y, dist=euclidean)
    return dist


def sted_scanpaths(X, Y, dimensions=2, time_delay=1, scaling_factor=1.0):
    """
    Scaled Time Delay Embedding Distance (STED) between two scanpaths.
    """
    if dimensions not in [2, 3]:
        raise ValueError('Invalid number of dimensions')

    def convert_to_accumulated_time(data):
        """
        Convert the time intervals to accumulated time.

        Args:
            data: A list or numpy array of shape (n, 3), where each row is [x, y, t],
                  and t represents the time interval between positions.

        Returns:
            A numpy array of shape (n, 3), where t is the accumulated time.
        """
        data = np.array(data)
        accumulated_time = np.cumsum(data[:, 2])
        data[:, 2] = accumulated_time
        return data

    X = convert_to_accumulated_time(X)
    Y = convert_to_accumulated_time(Y)
    X, Y = normalize_scanpath(X, Y)
    # Convert the scanpaths into numpy arrays

    # Split the scanpaths into spatial and accumulated time components
    Xpos = X[:, :dimensions]
    Ypos = Y[:, :dimensions]
    Xtime = X[:, dimensions:dimensions + 1].flatten()
    Ytime = Y[:, dimensions:dimensions + 1].flatten()

    # Create the time-delayed embedding for both scanpaths, using accumulated time
    X_embedded = _time_delay_embedding(Xpos, Xtime, time_delay, scaling_factor)
    Y_embedded = _time_delay_embedding(Ypos, Ytime, time_delay, scaling_factor)

    # Compute the distance matrix between the embedded scanpaths
    distance_matrix = cdist(X_embedded, Y_embedded, metric='euclidean')

    # Find the minimum path distance using dynamic programming
    sted_distance = _compute_min_path_distance(distance_matrix)

    return sted_distance


def _time_delay_embedding(positions, times, time_delay, scaling_factor):
    n = len(positions)
    embedding = []

    for i in range(n):
        delayed_position = positions[max(i - time_delay, 0)]  # Handle boundary condition
        accumulated_time = scaling_factor * times[i]
        embedded_vector = np.concatenate([positions[i], delayed_position, [accumulated_time]])
        embedding.append(embedded_vector)

    return np.array(embedding)


def _compute_min_path_distance(distance_matrix):
    n, m = distance_matrix.shape
    dp = np.zeros((n, m))

    dp[0, 0] = distance_matrix[0, 0]

    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + distance_matrix[i, 0]

    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + distance_matrix[0, j]

    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = distance_matrix[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[-1, -1]


if __name__ == "__main__":
    # Define the sequence
    sequence = ['h', 'e', 'l', 'l', 'l', 'inputbox', 'backspace', 'o']

    # Generate two sequences for comparison
    seq1 = generate_gaze_sequence(sequence, keyboard_layout, noise_level=20)
    seq2 = generate_gaze_sequence(sequence, keyboard_layout, noise_level=20)

    # Print the generated sequences
    print("Sequence 1 (x, y, trailtime):")
    for point in seq1:
        print(point)

    print("\nSequence 2 (x, y, trailtime):")
    for point in seq2:
        print(point)

    seq1_prepared = prepare_sequence_for_multimatch(seq1)
    seq2_prepared = prepare_sequence_for_multimatch(seq2)

    # Calculate the similarity score
    overall_similarity, pos_sim, dur_sim = multi_match(seq1_prepared, seq2_prepared, return_components=True)
    total_dtw_distance = dtw_scanpaths(seq1_prepared, seq2_prepared)
    sted_distance = sted_scanpaths(seq1_prepared, seq2_prepared)
    print(f"\nPosition similarity: {pos_sim}")
    print(f"Trailtime similarity: {dur_sim}")
    print(f"Overall similarity: {overall_similarity}")
    print(f"Total DTW Distance: {total_dtw_distance}")
    print(f"STED Distance: {sted_distance}")
