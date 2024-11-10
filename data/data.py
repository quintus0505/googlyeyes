import sys
import os

# Detect the current directory and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import pandas as pd
import os.path as osp
import os
from config import DEFAULT_DATA_DIR, GAZE_INFERENCE_DIR
from config import how_we_type_filtered_trails_one_finger, how_we_type_key_coordinate
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
from torchmetrics.text import CharErrorRate
from src.nets import AmortizedInferenceMLP
import torch

# Define file names and columns
typinglog_data_name = 'one_hand_typinglog_data.csv'
gaze_data_name = 'one_hand_gaze_data.csv'

typinglog_columns = ['index', 'trailtime', 'key', 'x', 'y', 'duration']
gaze_columns = ['index', 'trailtime', 'x', 'y', 'duration']

# Define directories
HOW_WE_TYPE_DATA_DIR = osp.join(osp.dirname(DEFAULT_DATA_DIR), 'data', 'how_we_type')
HOW_WE_TYPE_GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_gaze')
HOW_WE_TYPE_TYPING_LOG_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_typing_log')

GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
TYPING_LOG_DIR = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')

original_log_columns = ['systime', 'subject_id', 'block', 'sentence_id', 'trailtime', 'event', 'layout', 'INPUT',
                        'touchx', 'touchy']
original_gaze_columns = ['subject_id', 'block', 'sentence_id', 'trailtime', 'x', 'y']

# Desired ranges
x_min, x_max = 501.5, 1942.5
y_min, y_max = 100, 2645

x_min -= 501.5
x_max -= 501.5


def reshaping_typing_log_to_1080_1920(df, x_col, y_col):
    df[x_col] = df[x_col] * 1080 / 1441
    df[y_col] = df[y_col] * 1920 / 2760
    return df


# Scaling function
def scale_to_range(df, column, new_min, ratio):
    old_min = df[column].min()
    df[column] = ((df[column] - old_min) * ratio) + new_min
    return df


def reshaping_to_1080_1920(df, x_col, y_col):
    df[x_col] = df[x_col] * 1080 / 1260
    df[y_col] = df[y_col] * 1920 / 2574

    df[x_col] = df[x_col].apply(lambda x: x if x <= 1080 else 1080)
    df[y_col] = df[y_col].apply(lambda x: x if x <= 1920 else 1920)
    df[x_col] = df[x_col].apply(lambda x: x if x >= 0 else 0)
    df[y_col] = df[y_col].apply(lambda x: x if x >= 0 else 0)
    df[x_col] = df[x_col] * 0.9
    df[y_col] = df[y_col] * 0.9
    return df


def detect_fixations_ivt(df, x_col, y_col, trailtime_col, velocity_threshold=2.25):
    """
    Detects fixations using the I-VT (velocity threshold) algorithm.

    Parameters:
    df : DataFrame
        The input DataFrame containing gaze data.
    x_col : str
        The name of the column containing the x coordinates.
    y_col : str
        The name of the column containing the y coordinates.
    trailtime_col : str
        The name of the column containing the timestamp or trail time.
    velocity_threshold : float
        The velocity threshold (in pixels per ms) to determine fixations.

    Returns:
    fixation_df : DataFrame
        A DataFrame containing the start time, mean x and y coordinates,
        and duration of each detected fixation.
    """

    # Calculate velocities between consecutive points
    velocities = np.sqrt(np.diff(df[x_col]) ** 2 + np.diff(df[y_col]) ** 2) / np.diff(df[trailtime_col])
    velocities = np.append(velocities, 0)  # Append zero for the last point

    # Determine fixation points based on the velocity threshold
    fixation_mask = velocities < velocity_threshold
    fixations = []
    fixation_start = None

    for i, is_fixation in enumerate(fixation_mask):
        if is_fixation and fixation_start is None:
            fixation_start = i
        elif not is_fixation and fixation_start is not None:
            fixations.append((fixation_start, i))
            fixation_start = None

    # Initialize a list to hold the fixation data
    fixation_data = []

    for start, end in fixations:
        end += 1  # Ensure 'end' includes the last point in the fixation
        duration = df[trailtime_col].iloc[end] - df[trailtime_col].iloc[start]
        fixation_data.append({
            trailtime_col: df[trailtime_col].iloc[start],
            x_col: df[x_col].iloc[start:end].mean(),
            y_col: df[y_col].iloc[start:end].mean(),
            'duration': duration
        })

    # Create a DataFrame from the fixation data
    fixation_df = pd.DataFrame(fixation_data)

    return fixation_df


def calculate_single_trail_finger_metrics(group_input, sentence_df, sentence_id):
    group = group_input.copy()
    # if sentence colunm exists, use the target sentence from the group
    cer = CharErrorRate()
    if 'sentence' in group.columns:
        target_sentence = group['sentence'].iloc[0]
    else:
        target_sentence = sentence_df[sentence_df['sentence_n'] == sentence_id]['sentence'].iloc[0]

    # if the 'duration' column exists, rename it to "trailtime"
    if 'duration' in group.columns:
        # add up the duration columns using .loc to avoid the SettingWithCopyWarning
        group.loc[:, 'trailtime'] = group['duration'].cumsum()

    # rename all the "<" in the 'key' column to "B"
    group.loc[:, 'key'] = group['key'].apply(lambda x: 'B' if x == '<' else x)
    if 'duration' in group.columns:
        iki = group['duration'].mean()
    else:
        iki = group['trailtime'].diff().mean()
    num_backspaces = group[group['key'] == 'B'].shape[0]
    if 'INPUT' in group.columns:
        committed_sentence = group['INPUT'].iloc[-1]
    else:
        committed_sentence = ''
        for index_row, row in group.iterrows():
            tapped = str(row['key'])
            if tapped == 'B':
                if len(committed_sentence) > 0:
                    committed_sentence = committed_sentence[:-1]
                else:
                    committed_sentence = ""
            elif tapped != '>':
                committed_sentence += tapped
    wpm = (len(committed_sentence) - 1) / (group['trailtime'].max() / 1000) * 60 * (1 / 5)
    char_error_rate = cer(committed_sentence, target_sentence).item()
    return iki, num_backspaces, wpm, char_error_rate


def calculate_finger_metrics(typing_df, log_index=[]):
    # calculate the finger metrics for the human data
    # get char_error_rate, IKI, WPM, num_backspaces
    sentence_path = osp.join(DEFAULT_DATA_DIR, 'Sentences.csv')
    sentence_df = pd.read_csv(sentence_path)
    metric_dict = {}
    for index, group in typing_df.groupby('index'):
        metric_index = index.split('_')[0]
        sentence_id = int(index.split('_')[1])
        iki, num_backspaces, wpm, char_error_rate = calculate_single_trail_finger_metrics(group, sentence_df,
                                                                                          sentence_id)
        if metric_index not in metric_dict:
            metric_dict[metric_index] = {
                'IKI': [],
                'WPM': [],
                'num_backspaces': [],
                'char_error_rate': []
            }
        metric_dict[metric_index]['IKI'].append(iki)
        metric_dict[metric_index]['WPM'].append(wpm)
        metric_dict[metric_index]['num_backspaces'].append(num_backspaces)
        metric_dict[metric_index]['char_error_rate'].append(char_error_rate)
    avg_metric_dict = {}
    for metric_index, metrics in metric_dict.items():
        avg_metric_dict[metric_index] = {
            'IKI': np.mean(metrics['IKI']),
            'WPM': np.mean(metrics['WPM']),
            'num_backspaces': np.mean(metrics['num_backspaces']),
            'char_error_rate': np.mean(metrics['char_error_rate'])
        }
    return avg_metric_dict


def calculate_single_trail_gaze_metrics(group_input):
    fixation_counts = group_input.shape[0]
    fixation_duration_group = group_input.groupby('index').apply(lambda x: x.iloc[1:])
    fixation_duration = fixation_duration_group['duration'].mean()
    group = group_input.copy()
    group.loc[:, 'on_keyboard'] = group['y'] > 1230
    group.loc[:, 'on_text_entry'] = group['y'] < 1230

    # Create a 'state' column to identify where the gaze is
    group['state'] = np.where(group['on_keyboard'], 'on_keyboard',
                              np.where(group['on_text_entry'], 'on_text_entry', 'other'))

    # Extract fixations that are either on the keyboard or text entry area
    state_changes = group[group['state'].isin(['on_keyboard', 'on_text_entry'])].reset_index(drop=True)

    # Shift the 'state' column to compare with the previous state
    state_changes['prev_state'] = state_changes['state'].shift(1)

    # Identify transitions from 'on_keyboard' to 'on_text_entry'
    state_changes['shift'] = (state_changes['prev_state'] == 'on_keyboard') & (
            state_changes['state'] == 'on_text_entry')
    gaze_shifts = state_changes['shift'].sum()

    group.loc[:, 'keyboard_fixation_time'] = group.apply(lambda row: row['duration'] if row['on_keyboard'] else 0,
                                                         axis=1)
    group.loc[:, 'text_entry_fixation_time'] = group.apply(
        lambda row: row['duration'] if row['on_text_entry'] else 0, axis=1)
    keyboard_fixation_time = group['keyboard_fixation_time'].sum()
    text_entry_fixation_time = group['text_entry_fixation_time'].sum()
    total_fixation_time = group['duration'].sum()

    time_ratio_on_keyboard = keyboard_fixation_time / total_fixation_time
    time_ratio_on_text_entry = text_entry_fixation_time / total_fixation_time

    return fixation_counts, fixation_duration, gaze_shifts, time_ratio_on_keyboard, time_ratio_on_text_entry


def calculate_gaze_metrics(fixation_df, log_index=[]):
    # calculate the gaze metrics for the human data
    fixation_counts_list = []
    fixation_duration_list = []
    gaze_shifts_list = []
    time_ratio_on_keyboard_list = []
    time_ratio_on_text_entry_list = []
    for index, group in fixation_df.groupby('index'):
        fixation_counts, fixation_duration, gaze_shifts, time_ratio_on_keyboard, time_ratio_on_text_entry = calculate_single_trail_gaze_metrics(
            group)
        fixation_counts_list.append(fixation_counts)
        fixation_duration_list.append(fixation_duration)
        gaze_shifts_list.append(gaze_shifts)
        time_ratio_on_keyboard_list.append(time_ratio_on_keyboard)
        time_ratio_on_text_entry_list.append(time_ratio_on_text_entry)

    mean_fixations = np.mean(fixation_counts_list)
    std_fixations = np.std(fixation_counts_list)
    mean_fixation_duration = np.mean(fixation_duration_list)
    std_fixation_duration = np.std(fixation_duration_list)
    mean_gaze_shifts = np.mean(gaze_shifts_list)
    std_gaze_shifts = np.std(gaze_shifts_list)
    mean_time_ratio_on_keyboard = np.mean(time_ratio_on_keyboard_list)
    std_time_ratio_on_keyboard = np.std(time_ratio_on_keyboard_list)
    mean_time_ratio_on_text_entry = np.mean(time_ratio_on_text_entry_list)
    std_time_ratio_on_text_entry = np.std(time_ratio_on_text_entry_list)

    return {
        'mean_fixations': mean_fixations,
        'std_fixations': std_fixations,
        'mean_fixation_duration': mean_fixation_duration,
        'std_fixation_duration': std_fixation_duration,
        'mean_gaze_shifts': mean_gaze_shifts,
        'std_gaze_shifts': std_gaze_shifts,
        'mean_time_ratio_on_keyboard': mean_time_ratio_on_keyboard,
        'std_time_ratio_on_keyboard': std_time_ratio_on_keyboard,
        'mean_time_ratio_on_text_entry': mean_time_ratio_on_text_entry,
        'std_time_ratio_on_text_entry': std_time_ratio_on_text_entry
    }


def add_duration_column(df, trailtime_col):
    df['duration'] = df[trailtime_col].diff().fillna(df[trailtime_col])
    return df


def load_human_data(calculate_params=False):
    typinglog_path = osp.join(DEFAULT_DATA_DIR, typinglog_data_name)
    gaze_path = osp.join(DEFAULT_DATA_DIR, gaze_data_name)
    if not osp.exists(typinglog_path) and not osp.exists(gaze_path):
        print('Data not found, generating...')
        # Currently only have data for one hand typing, use the same method for two hand typing generation
        typing_data_df = pd.DataFrame(columns=typinglog_columns)
        gaze_data_df = pd.DataFrame(columns=gaze_columns)

        for file in os.listdir(GAZE_DATA_DIR):
            if file.endswith('2.csv'):
                continue
            print("Processing file: ", file)
            file_path = osp.join(GAZE_DATA_DIR, file)
            gaze_df = pd.read_csv(file_path, names=original_gaze_columns)
            gaze_df = gaze_df.iloc[1:]
            gaze_df['x'] = gaze_df['x'].astype(float)
            gaze_df['y'] = gaze_df['y'].astype(float)
            gaze_df['sentence_id'] = gaze_df['sentence_id'].astype(int)
            gaze_df['trailtime'] = gaze_df['trailtime'].astype(float).astype(int)
            typing_file = file.replace("gaze", "typinglog")
            typing_path = osp.join(TYPING_LOG_DIR, typing_file)
            if not osp.exists(typing_path):
                continue

            typinglog_df = pd.read_csv(typing_path, names=original_log_columns)
            typinglog_df = typinglog_df.iloc[1:]
            typinglog_df['touchx'] = typinglog_df['touchx'].astype(float)
            typinglog_df['touchy'] = typinglog_df['touchy'].astype(float)
            typinglog_df['trailtime'] = typinglog_df['trailtime'].astype(float).astype(int)
            typinglog_df['sentence_id'] = typinglog_df['sentence_id'].astype(int)

            typinglog_df.loc[:, 'touchy'] += 1840 - typinglog_df['touchy'].min()
            typinglog_df = reshaping_typing_log_to_1080_1920(typinglog_df, 'touchx', 'touchy')

            sentence_groups = gaze_df.groupby('sentence_id')

            for sentence_id, group in sentence_groups:
                subject_id = group['subject_id'].iloc[0]
                if int(subject_id) == 129 and int(sentence_id) == 1:
                    test = 1
                if str(sentence_id) in how_we_type_filtered_trails_one_finger[
                    str(subject_id)]:
                    continue
                save_index = subject_id + '_' + str(sentence_id)
                typinglog_group = typinglog_df[typinglog_df['sentence_id'] == sentence_id].copy()
                if typinglog_group['trailtime'].max() - group['trailtime'].max() > 2000:
                    continue
                group = group[group['trailtime'] >= 0]
                group = group[group['trailtime'] <= typinglog_group['trailtime'].max() + 100]
                group = scale_to_range(group, 'x', x_min + 30, 1.22)
                group = scale_to_range(group, 'y', y_min, 1.28)

                group = reshaping_to_1080_1920(group, 'x', 'y')

                # Detect fixations
                fixation_df = detect_fixations_ivt(group, 'x', 'y', 'trailtime')
                fixation_df['index'] = save_index
                fixation_df = add_duration_column(fixation_df, 'trailtime')
                # Transforming the columns before appending
                typinglog_temp_df = typinglog_group[['trailtime', 'event', 'touchx', 'touchy', 'INPUT']].copy()
                typinglog_temp_df.rename(columns={'event': 'key', 'touchx': 'x', 'touchy': 'y'}, inplace=True)
                typinglog_temp_df['index'] = save_index

                # Add duration column to typing log
                typinglog_temp_df = add_duration_column(typinglog_temp_df, 'trailtime')

                # Selecting only the columns we need
                typinglog_temp_df = typinglog_temp_df[['index', 'trailtime', 'key', 'x', 'y', 'duration', 'INPUT']]

                # updating the typing data and gaze data, not using append
                typing_data_df = pd.concat([typing_data_df, typinglog_temp_df], ignore_index=True)
                gaze_data_df = pd.concat([gaze_data_df, fixation_df], ignore_index=True)

        typing_data_df.to_csv(typinglog_path, index=False)
        gaze_data_df.to_csv(gaze_path, index=False)

        print('Number of unique indexes in typing_data_df: ', typing_data_df['index'].nunique())

    else:
        typing_data_df = pd.read_csv(typinglog_path)
        gaze_data_df = pd.read_csv(gaze_path)
        print('Data loaded successfully')
        print('Number of unique indexes in typing_data_df: ', typing_data_df['index'].nunique())
    params_df = None
    if calculate_params:
        print("Calculating finger metrics...")
        input_size = 4  # Dimensionality of the first 4 averaged stats
        hidden_size = 128  # Can be adjusted
        output_size = 3  # Use only the first 3 dimensions of params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_save_path = os.path.join(GAZE_INFERENCE_DIR, "src", "../src/best_outputs", "amortized_inference.pth")
        model = AmortizedInferenceMLP(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        finger_metrics = calculate_finger_metrics(typing_data_df)
        simulated_params_columns = ['index', 'params']
        params_data = []
        with torch.no_grad():
            for key, item in finger_metrics.items():
                human_metric = np.array([item['char_error_rate'], item['IKI'], item['WPM'], item['num_backspaces']])
                human_metric = torch.tensor(human_metric, dtype=torch.float32).unsqueeze(0).to(device)
                param = model(human_metric)
                # change the param from touch tensor to numpy array
                param = param.squeeze().cpu().numpy().astype(np.float64)
                params_data.append({'index': key, 'params': param})

        params_df = pd.DataFrame(params_data, columns=simulated_params_columns)
    return typing_data_df, gaze_data_df, params_df


def load_simulated_data(fpath_header='train'):
    params_list = []
    for file in os.listdir(DEFAULT_DATA_DIR):
        if file.startswith(fpath_header):
            params_list.append(file)
    params_list.sort()
    fpath = osp.join(DEFAULT_DATA_DIR, params_list[0])

    with open(fpath, "rb") as f:
        dataset = pickle.load(f)
        print("shape of params_arr: {}".format(dataset['params'].shape))
        print("shape of stats_arr: {}".format(dataset['stats'].shape))
    simulated_params_columns = ['index', 'params']

    # Initialize lists to collect data
    typinglog_data = []
    gaze_data = []
    params_data = []
    group_trail_num = dataset['stats'].shape[1]
    group_num = dataset['params'].shape[0]
    # Loop through the dataset and collect the data in lists
    for param_index in tqdm(range(group_num)):

        for grounp_index in range(group_trail_num):
            finger_log = dataset['finger_log'][param_index][grounp_index]
            gaze_log = dataset['vision_log'][param_index][grounp_index]
            # if the finger log or gaze log is empty, skip
            if len(finger_log) == 0 or len(gaze_log) == 0:
                continue

            # if any of the gaze_log or finger_log's duration < 0 or > 10000, skip
            if any([x['duration'] < 0 or x['duration'] > 10000 for x in finger_log]) or any(
                    [x['duration'] < 0 or x['duration'] > 10000 for x in gaze_log]):
                continue
            # Convert lists with dicts to DataFrames
            finger_log_df = pd.DataFrame(finger_log)
            gaze_log_df = pd.DataFrame(gaze_log)

            # Add index information
            finger_log_df['index'] = f"{param_index + 200}_{grounp_index}"
            gaze_log_df['index'] = f"{param_index + 200}_{grounp_index}"

            # add trailtime column, when adding, set the first duration to 0 of finger_log_df only
            finger_log_df.iloc[0, finger_log_df.columns.get_loc('duration')] = 0
            finger_log_df['trailtime'] = finger_log_df['duration'].cumsum()
            gaze_log_df['trailtime'] = gaze_log_df['duration'].cumsum()

            # Append the DataFrames to the lists
            typinglog_data.append(finger_log_df)
            gaze_data.append(gaze_log_df)
        # if no data is collected for param_index + 200, skip
        if len([x for x in typinglog_data if x['index'].iloc[0].split('_')[0] == str(param_index + 200)]) == 0:
            continue
        params = dataset['params'][param_index].astype(np.float64)
        params_data.append({'index': f"{param_index + 200}", 'params': params})

    # Concatenate all collected DataFrames at once
    typinglog_df = pd.concat(typinglog_data, ignore_index=True)
    gaze_df = pd.concat(gaze_data, ignore_index=True)
    params_df = pd.DataFrame(params_data, columns=simulated_params_columns)

    print("unique trails in typing and gaze data: ", typinglog_df['index'].nunique(), gaze_df['index'].nunique())

    return typinglog_df, gaze_df, params_df


def rebuild_input(typing_data):
    input_str = ''
    # start from the second row
    for index, row in typing_data.iterrows():
        if row['key'] == 'B':
            input_str += '<'
        else:
            input_str += row['key']
    return input_str


def calculated_simulated_data_metrics(fpath_header='train'):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        False, include_duration=True, max_pred_len=32, data_use='simulated', fpath_header=fpath_header)
    testing_gaze_data = gaze_data[gaze_data['index'].isin(indices_test)].copy()
    training_gaze_data = gaze_data[gaze_data['index'].isin(indices_train)].copy()

    testing_finger_data = typing_data[typing_data['index'].isin(indices_test)].copy()
    training_finger_data = typing_data[typing_data['index'].isin(indices_train)].copy()
    print("Computing testing gaze data metrics")
    metrics = calculate_gaze_metrics(testing_gaze_data)
    print(
        f'Average fixation time: {metrics["mean_fixation_duration"]} ms (std: {metrics["std_fixation_duration"]} ms)')
    print(
        f'Average number of fixations per trail: {metrics["mean_fixations"]} (std: {metrics["std_fixations"]})')
    print(
        f'Average number of gaze shifts per trail: {metrics["mean_gaze_shifts"]} (std: {metrics["std_gaze_shifts"]})')
    print(
        f'Time ratio for gaze on keyboard: {metrics["mean_time_ratio_on_keyboard"]} (std: {metrics["std_time_ratio_on_keyboard"]})')

    metrics = calculate_finger_metrics(testing_finger_data)
    iki_list = []
    wpm_list = []
    num_backspaces_list = []
    error_rate_list = []
    for key, item in metrics.items():
        iki_list.append(item['IKI'])
        wpm_list.append(item['WPM'])
        num_backspaces_list.append(item['num_backspaces'])
        error_rate_list.append(item['char_error_rate'])

    print("Computing testing finger data metrics")
    print(f'Average IKI: {np.mean(iki_list)} ms (std: {np.std(iki_list)} ms)')
    print(f'Average WPM: {np.mean(wpm_list)} (std: {np.std(wpm_list)})')
    print(f'Average number of backspaces: {np.mean(num_backspaces_list)} (std: {np.std(num_backspaces_list)})')
    print(f'Average character error rate: {np.mean(error_rate_list)} (std: {np.std(error_rate_list)})')

    print("Computing training gaze data metrics")
    metrics = calculate_gaze_metrics(training_gaze_data)
    print(
        f'Average fixation time: {metrics["mean_fixation_duration"]} ms (std: {metrics["std_fixation_duration"]} ms)')
    print(
        f'Average number of fixations per trail: {metrics["mean_fixations"]} (std: {metrics["std_fixations"]})')
    print(
        f'Average number of gaze shifts per trail: {metrics["mean_gaze_shifts"]} (std: {metrics["std_gaze_shifts"]})')
    print(
        f'Time ratio for gaze on keyboard: {metrics["mean_time_ratio_on_keyboard"]} (std: {metrics["std_time_ratio_on_keyboard"]})')

    metrics = calculate_finger_metrics(training_finger_data)
    iki_list = []
    wpm_list = []
    num_backspaces_list = []
    error_rate_list = []
    for key, item in metrics.items():
        iki_list.append(item['IKI'])
        wpm_list.append(item['WPM'])
        num_backspaces_list.append(item['num_backspaces'])
        error_rate_list.append(item['char_error_rate'])

    print("Computing training finger data metrics")
    print(f'Average IKI: {np.mean(iki_list)} ms (std: {np.std(iki_list)} ms)')
    print(f'Average WPM: {np.mean(wpm_list)} (std: {np.std(wpm_list)})')
    print(f'Average number of backspaces: {np.mean(num_backspaces_list)} (std: {np.std(num_backspaces_list)})')
    print(f'Average character error rate: {np.mean(error_rate_list)} (std: {np.std(error_rate_list)})')


def load_and_preprocess_data(include_key=True, split_indx=128, include_duration=True, max_len=32, max_pred_len=32,
                             data_use='human', fpath_header='train', calculate_params=False):
    if data_use == 'human':
        typing_data, gaze_data, params_data = load_human_data(calculate_params=calculate_params)
    elif data_use == 'simulated':
        typing_data, gaze_data, params_data = load_simulated_data(fpath_header=fpath_header)
    else:
        human_typing_data, human_gaze_data, human_params_data = load_human_data(calculate_params=calculate_params)
        simulated_typing_data, simulated_gaze_data, simulated_params_data = load_simulated_data(
            fpath_header=fpath_header)
        print("unique trails in simulated typing and gaze data after filtering with {} max_len: {}".format(max_len,
                                                                                                           simulated_typing_data[
                                                                                                               'index'].nunique()))
        # use human data's own columns
        typing_data = pd.concat([human_typing_data,
                                 simulated_typing_data[['index', 'key', 'x', 'y', 'duration', 'trailtime']]],
                                ignore_index=True)
        # keep 'index' 'x' 'y' 'duration' in the gaze data
        gaze_data = pd.concat([human_gaze_data,
                               simulated_gaze_data[['index', 'x', 'y', 'duration', 'trailtime']]], ignore_index=True)
        params_data = pd.concat([human_params_data, simulated_params_data], ignore_index=True)

    # Select necessary columns
    input_columns = ['x', 'y']
    output_columns = ['x', 'y']
    if include_duration:
        input_columns.append('duration')
        output_columns.append('duration')
    if include_key:
        input_columns.append('key')
        # if x, y exist, remove them
        if 'x' in input_columns:
            input_columns.remove('x')
        if 'y' in input_columns:
            input_columns.remove('y')

    # Integer encode the 'key' column if included
    if include_key:
        key_vocab_size = len(how_we_type_key_coordinate) - 2
        typing_data['key'] = typing_data['key'].astype('category').cat.codes
        typing_data['key'] = typing_data['key'] / (key_vocab_size - 1)  # Normalize to (0, 1)

    if calculate_params:
        # add a new column 'params' to the typing_data, use the index before the first '_' as the key to find the params
        # in the params_data
        input_columns.append('params')
        typing_data['params'] = typing_data['index'].apply(
            lambda x: params_data[params_data['index'] == x.split('_')[0]]['params'].values[0])

    # Group by index
    grouped_typing = typing_data.groupby('index')
    grouped_gaze = gaze_data.groupby('index')

    filtered_typing_indices = []
    filtered_gaze_indices = []
    for idx, group in grouped_typing:
        if len(group) <= max_len:
            filtered_typing_indices.append(idx)
    for idx, group in grouped_gaze:
        if len(group) <= max_len:
            filtered_gaze_indices.append(idx)

    filtered_indices = list(set(filtered_typing_indices).intersection(set(filtered_gaze_indices)))

    typing_data = typing_data[typing_data['index'].isin(filtered_indices)]
    gaze_data = gaze_data[gaze_data['index'].isin(filtered_indices)]

    print("unique trails in typing and gaze data after filtering with {} max_len: {}".format(max_len, typing_data[
        'index'].nunique()))

    # Group by index
    grouped_typing = typing_data.groupby('index')
    grouped_gaze = gaze_data.groupby('index')

    # Ensure sequences are the same length by padding
    max_len_typing = max(grouped_typing.size())
    max_len_gaze = max(grouped_gaze.size())
    max_len = max(max_len_typing, max_len_gaze)

    mean_gaze_len = np.mean(grouped_gaze.size())
    mean_typing_len = np.mean(grouped_typing.size())
    print(f'Mean gaze sequence length: {mean_gaze_len}')
    print(f'Mean typing sequence length: {mean_typing_len}')

    def pad_sequence(seq, max_len):
        seq = seq.values
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        mask = np.ones((len(seq),), dtype=bool)
        mask = np.concatenate([mask, np.zeros((max_len - len(seq),), dtype=bool)])
        return np.vstack((seq, padding)), mask

    X_list = []
    y_list = []
    masks_x_list = []
    masks_y_list = []
    indices_list = []
    indices_list_test = []
    for idx, group in grouped_typing:
        X_padded, mask_x = pad_sequence(group[input_columns], max_len)
        X_list.append(X_padded)
        masks_x_list.append(mask_x)
        indices_list.append(idx)

    for idx, group in grouped_gaze:
        y_padded, mask_y = pad_sequence(group[output_columns], max_len)
        y_list.append(y_padded)
        masks_y_list.append(mask_y)
        indices_list_test.append(idx)

    # print if the two indices are the same
    print("The two indices are the same: ", indices_list == indices_list_test)

    X = np.array(X_list)
    y = np.array(y_list)
    masks_x = np.array(masks_x_list)
    masks_y = np.array(masks_y_list)
    indices = np.array(indices_list)

    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_shape = X.shape
    y_shape = y.shape

    # Scale only the numeric columns, excluding the key if included
    if include_key:
        if include_duration:
            X_num = X[:, :, 0].reshape(X.shape[0], X.shape[1], 1)  # Only 'key'
            X_cat = X[:, :, 1].reshape(X.shape[0], X.shape[1], 1)  # Only 'duration'
        else:
            X_cat = X[:, :, 1].reshape(X.shape[0], X.shape[1], 1)  # Only 'key'
            X_num = np.array([]).reshape(X.shape[0], X.shape[1],
                                         0)  # No numeric columns if only 'key' is included and 'duration' is not
    else:
        X_num = X[:, :, :3]

    if X_num.size > 0:  # Check if X_num is not empty
        X_num_scaled = scaler_X.fit_transform(X_num.reshape(-1, X_num.shape[-1])).reshape(X_num.shape)
    else:
        X_num_scaled = X_num  # No scaling needed if X_num is empty

    if include_key:
        if X_num_scaled.size > 0:
            X_scaled = np.concatenate((X_cat, X_num_scaled), axis=2)
        else:
            X_scaled = X_cat
    else:
        X_scaled = X_num_scaled

    if calculate_params:
        params_cat = X[:, :, -1].reshape(X.shape[0], X.shape[1], 1)
        # Initialize an empty array to hold the expanded shape
        params_expanded = np.zeros((params_cat.shape[0], params_cat.shape[1], 3))

        # Iterate over each element and assign the (3,) array to the appropriate place in the new array
        for i in range(params_cat.shape[0]):
            for j in range(params_cat.shape[1]):
                params_expanded[i, j, :] = params_cat[
                    i, j, 0]  # The [0] is used to access the (3,) array inside each element

        X_scaled = np.concatenate((X_scaled, params_expanded), axis=2)

    y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y_shape)

    # Split the data based on the condition
    # 100 < index <
    if data_use == 'human':
        train_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) <= split_indx]
        test_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) > split_indx]
    elif data_use == 'simulated':
        # get the first 500 indexes as testing data (<500)
        train_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) > 205]
        test_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) <= 205]
    else:
        # all simulated data (index >= 200) will be training data, testing data will be with index between split_index and 200
        # training data: index < split_index and index >= 200
        train_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) <= split_indx or int(
            idx.split('_')[0]) >= 200]
        # testing data: index >= split_index and index < 200
        test_indices = [i for i, idx in enumerate(indices) if int(idx.split('_')[0]) > split_indx and int(
            idx.split('_')[0]) < 200]

    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y_scaled[train_indices], y_scaled[test_indices]
    masks_x_train, masks_x_test = masks_x[train_indices], masks_x[test_indices]
    masks_y_train, masks_y_test = masks_y[train_indices], masks_y[test_indices]
    indices_train, indices_test = indices[train_indices], indices[test_indices]

    # Limit the prediction length to max_pred_len
    def limit_length(arr, length):
        return arr[:, :length, :]

    X_train = limit_length(X_train, max_pred_len)
    X_test = limit_length(X_test, max_pred_len)
    y_train = limit_length(y_train, max_pred_len)
    y_test = limit_length(y_test, max_pred_len)
    masks_x_train = masks_x_train[:, :max_pred_len]
    masks_x_test = masks_x_test[:, :max_pred_len]
    masks_y_train = masks_y_train[:, :max_pred_len]
    masks_y_test = masks_y_test[:, :max_pred_len]

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data


if __name__ == '__main__':
    typing_data, gaze_data, params_data = load_human_data(calculate_params=True)
