from config import HOW_WE_TYPE_TYPING_LOG_DATA_DIR, HOW_WE_TYPE_GAZE_DATA_DIR, \
    HOW_WE_TYPE_FINGER_DATA_DIR, GAZE_INFERENCE_DIR, how_we_type_key_coordinate
import pandas as pd
# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from How_we_type_code.correlation_study import process_all_distance_and_similarity

original_gaze_columns = ['subject_id', 'block', 'sentence_id', 'trialtime', 'x', 'y']
original_gaze_columns_plus_position = original_gaze_columns + ['position']

original_finger_columns = ['optitime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'x1', 'y1', 'z1', 'x2', 'y2',
                           'z2']
original_finger_columns_plus_position = original_finger_columns + ['position1, position2']
original_log_columns = ['systime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'DATA', 'layout', 'INPUT',
                        'touchx', 'touchy']


# Function to filter out top and bottom 2.5% of values
def filter_percentiles(df, column, lower_percentile=2.5, upper_percentile=97.5):
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Desired ranges
x_min, x_max = 501.5, 1942.5
y_min, y_max = 100, 2645  # 130 * 2760 / 1920 = 187

x_min -= 501.5
x_max -= 501.5


# Scaling function
def scale_to_range(df, column, new_min, new_max):
    old_min = df[column].min()
    old_max = df[column].max()
    df[column] = ((df[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return df


# x_min, y_min = 0, 0

def get_key_position(x, y):
    for key, value in how_we_type_key_coordinate.items():
        if value[0] <= x <= value[2] and value[1] <= int(y) <= value[3]:
            return key
    return '-'


def reshaping_to_1080_1920(df, x_col, y_col, finger=False):
    # get the 99% max value of x and y
    if not finger:
        df_max_x = df[x_col].max()
        df_max_y = df[y_col].max()
        # Using .loc to avoid SettingWithCopyWarning
        df.loc[:, x_col] = df[x_col] * 1080 / df_max_x
        df.loc[:, y_col] = df[y_col] * 1920 / df_max_y
    else:
        df_max_x = df[x_col].quantile(0.95)
        df_max_y = df[y_col].quantile(0.95)
        # removing those values that are not in the range of the keyboard
        # df = df[df[x_col] <= df[x_col].quantile(0.87)]
        # df = df[df[y_col] <= df[y_col].quantile(0.85)]
        df[x_col] = df[x_col] * 1080 / df_max_x
        df[y_col] = df[y_col] * 1920 / df_max_y

        df[x_col] = df[x_col].apply(lambda x: x if x <= 1080 else 1080)
        df[y_col] = df[y_col].apply(lambda x: x if x <= 1920 else 1920)
        df[x_col] = df[x_col].apply(lambda x: x if x >= 0 else 0)
        df[y_col] = df[y_col].apply(lambda x: x if x >= 0 else 0)
    return df


def reshaping_typing_log_to_1080_1920(df, x_col, y_col):
    df[x_col] = df[x_col] * 1080 / 1441
    df[y_col] = df[y_col] * 1920 / 2760
    return df


class GazeTypingAlignmentNN(nn.Module):
    def __init__(self):
        super(GazeTypingAlignmentNN, self).__init__()
        self.fc = nn.Linear(2, 2)  # 2 inputs: x, y and 2 outputs: x', y'

    def forward(self, x):
        return self.fc(x)


def linear_adjust_nn(gaze_group, typinglog_group, epochs=1000, learning_rate=0.001, threshold=100):
    # Filter gaze points with y > 600
    gaze_group = gaze_group[gaze_group['y'] > 600]

    # Normalize the data to the range [0, 1]
    gaze_group['x_norm'] = gaze_group['x'] / 1080.0
    gaze_group['y_norm'] = gaze_group['y'] / 1920.0
    typinglog_group['touchx_norm'] = typinglog_group['touchx'] / 1080.0
    typinglog_group['touchy_norm'] = typinglog_group['touchy'] / 1920.0

    gaze_coords = torch.tensor(gaze_group[['x_norm', 'y_norm']].values, dtype=torch.float32)
    typing_coords = torch.tensor(typinglog_group[['touchx_norm', 'touchy_norm']].values, dtype=torch.float32)

    model = GazeTypingAlignmentNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        transformed_gaze = model(gaze_coords)

        unique_points = set()
        for tg in transformed_gaze:
            distances = torch.sqrt((typing_coords[:, 0] - tg[0]) ** 2 + (typing_coords[:, 1] - tg[1]) ** 2)
            closest_idx = torch.argmin(distances)
            if distances[closest_idx] < threshold:
                unique_points.add(closest_idx.item())

        # Define the loss as the negative number of unique points covered
        loss = -len(unique_points)
        loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss_tensor.item()}, Unique Points: {len(unique_points)}')

    # Transform the gaze points using the trained model
    model.eval()
    with torch.no_grad():
        transformed_gaze = model(gaze_coords).numpy()

    # Recover the original values
    gaze_group['x_transformed'] = transformed_gaze[:, 0] * 1080.0
    gaze_group['y_transformed'] = transformed_gaze[:, 1] * 1920.0
    gaze_group['x'] = gaze_group['x_transformed']
    gaze_group['y'] = gaze_group['y_transformed']
    return gaze_group


def visualize_data():
    GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
    TYPY_DATA_DIR = osp.join(HOW_WE_TYPE_FINGER_DATA_DIR, 'Finger_Motion_Capture')
    TYPING_LOG_DIR = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')

    # reshape the keys to x (0, 1080) y (0, 1920)
    for key, coord in how_we_type_key_coordinate.items():
        how_we_type_key_coordinate[key] = [coord[0] - 501.5, coord[1], coord[2] - 501.5, coord[3]]
        # reshape the keys to x (0, 1080) y (0, 1920)
    for key, coord in how_we_type_key_coordinate.items():
        how_we_type_key_coordinate[key] = [coord[0] * 1080 / 1441, coord[1] * 1920 / 2760, coord[2] * 1080 / 1441,
                                           coord[3] * 1920 / 2760]

    for file in os.listdir(GAZE_DATA_DIR):
        print("Processing file: ", file)
        # if not file.endswith("129_1.csv"):
        #     continue
        # get the number like this 101_1 as save_dir_name in gaze_101_1.csv
        save_dir_name = file.split('.')[0].split('_')[1] + '_' + file.split('.')[0].split('_')[2]
        if not os.path.exists(f'../figs/how_we_type/{save_dir_name}'):
            os.makedirs(f'../figs/how_we_type/{save_dir_name}')
        file_path = osp.join(GAZE_DATA_DIR, file)
        gaze_df = pd.read_csv(file_path, names=original_gaze_columns)
        gaze_df = gaze_df.iloc[1:]
        gaze_df['x'] = gaze_df['x'].astype(float)
        gaze_df['y'] = gaze_df['y'].astype(float)
        gaze_df['sentence_id'] = gaze_df['sentence_id'].astype(int)
        gaze_df['trialtime'] = gaze_df['trialtime'].astype(float).astype(int)
        # Load the corresponding finger data
        finger_file = file.replace("gaze", "finger")
        finger_path = osp.join(TYPY_DATA_DIR, finger_file)
        if not osp.exists(finger_path):
            continue

        typing_file = file.replace("gaze", "typinglog")
        typing_path = osp.join(TYPING_LOG_DIR, typing_file)

        typinglog_df = pd.read_csv(typing_path, names=original_log_columns)
        typinglog_df = typinglog_df.iloc[1:]
        typinglog_df['touchx'] = typinglog_df['touchx'].astype(float)
        typinglog_df['touchy'] = typinglog_df['touchy'].astype(float)
        typinglog_df['trialtime'] = typinglog_df['trialtime'].astype(float).astype(int)
        typinglog_df['sentence_id'] = typinglog_df['sentence_id'].astype(int)

        # typinglog_df.loc[:, 'touchx'] += 501.5 - typinglog_df['touchx'].min()
        typinglog_df.loc[:, 'touchy'] += 1840 - typinglog_df['touchy'].min()
        typinglog_df = reshaping_typing_log_to_1080_1920(typinglog_df, 'touchx', 'touchy')

        # finger_df = pd.read_csv(finger_path, names=original_finger_columns)
        # finger_df = finger_df.iloc[1:]
        # finger_df[['x1', 'y1', 'x2', 'y2']] = finger_df[['x1', 'y1', 'x2', 'y2']].astype(float)
        # finger_df['sentence_id'] = finger_df['sentence_id'].astype(int)
        #
        # finger_df.loc[:, 'x1'] -= 501.5
        # finger_df.loc[:, 'x2'] = finger_df['x2'].apply(lambda x: x - 501.5 if x != 0 else 0)
        #
        # finger_df = reshaping_to_1080_1920(finger_df, 'x1', 'y1', finger=True)
        # if finger_df['x2'].max() != 0:
        #     finger_df = reshaping_to_1080_1920(finger_df, 'x2', 'y2', finger=True)

        sentence_groups = gaze_df.groupby('sentence_id')
        for sentence_id, group in sentence_groups:
            typinglog_group = typinglog_df[typinglog_df['sentence_id'] == sentence_id].copy()

            # remove those rows in group that have trailtime < 0
            group = group[group['trialtime'] >= 0]
            group = group[group['trialtime'] <= typinglog_group['trialtime'].max() + 100]

            plt.figure(figsize=(9, 16))
            # group = filter_percentiles(group, 'x', lower_percentile=5, upper_percentile=95)
            # group = filter_percentiles(group, 'y', lower_percentile=5, upper_percentile=95)
            #
            group = scale_to_range(group, 'x', x_min + 30, x_max - 30)
            group = scale_to_range(group, 'y', y_min, y_max)

            group = reshaping_to_1080_1920(group, 'x', 'y')

            # group = linear_adjust_nn(group, typinglog_group)
            # Draw the keyboard layout
            for key, coord in how_we_type_key_coordinate.items():
                plt.gca().add_patch(plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1],
                                                  fill=None, edgecolor='blue', linewidth=1))
                plt.text((coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2, key,
                         horizontalalignment='center', verticalalignment='center', color='blue')
            # Plot gaze trail
            plt.plot(group['x'], group['y'], color='red', marker='o', linestyle='-', markersize=2, linewidth=0.5)

            # Reshape and plot typing trail
            # finger_group = finger_df[finger_df['sentence_id'] == sentence_id].copy()
            # if not finger_group.empty:
            #     plt.plot(finger_group['x1'], finger_group['y1'], color='green', marker='o', linestyle='-',
            #              markersize=2, linewidth=0.5, label='Finger 1')
            #     plt.plot(finger_group['x2'], finger_group['y2'], color='purple', marker='o', linestyle='-',
            #              markersize=2, linewidth=0.5, label='Finger 2')

            # plot the touch points as dots
            plt.scatter(typinglog_group['touchx'], typinglog_group['touchy'], color='yellow', marker='o', s=40,
                        label='Touch Points')
            plt.title(f'Gaze and Typing Trail for Sentence {sentence_id}')
            # plt.xlim(501.5, 1942.5)
            # plt.ylim(0, 2760)
            plt.xlim(0, 1080)
            plt.ylim(0, 1920)
            plt.gca().invert_yaxis()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            # plt.savefig(f'../figs/how_we_type/gaze_typing_trail_sentence_{sentence_id}.png')
            plt.savefig(f'../figs/how_we_type/{save_dir_name}/gaze_typing_trail_sentence_{sentence_id}.png')
            plt.close()


if __name__ == '__main__':
    visualize_data()
    # process_all_distance_and_similarity()
