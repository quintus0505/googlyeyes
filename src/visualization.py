import sys
import os
# Detect the current directory and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import os.path as osp
import numpy as np
import time
import pygame
import torch
from config import GAZE_INFERENCE_DIR
from src.nets import TransformerModel
from data.data import load_and_preprocess_data, rebuild_input, \
    calculate_single_trail_gaze_metrics, calculate_single_trail_finger_metrics, DEFAULT_DATA_DIR
from src.nets import GooglyeyesModel, TypingGazeInferenceDataset, TypingGazeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import pandas as pd
import cv2

keyboard_image = osp.join(GAZE_INFERENCE_DIR, 'figs', 'chi_keyboard.png')
img_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'baseline_validation')
video_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'videos')
saved_model_dir = osp.join(GAZE_INFERENCE_DIR, 'src', 'best_outputs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def visualize_gaze_and_typing(index, typing_data, gaze_data, predicted_gaze_data, model_type='lstm', data_use='human',
                              amortized_inference=False, visualize_text=False, parameter=None,
                              visualize_user_params=False):
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1080, 1920), pygame.HIDDEN)
    pygame.display.set_caption(f'Index {index}: {model_type} {data_use} Visualization')

    # Load the keyboard image
    keyboard_img = pygame.image.load(keyboard_image)
    keyboard_img = pygame.transform.scale(keyboard_img, (1080, 1920))

    def plot_points(data, color_map, alpha=255):
        for i, point in enumerate(data):
            x, y, dur = point
            color = color_map(i / len(data))
            size = max(int(3 + dur * 0.04), 0)  # Scale point size by duration
            # Create a surface with per-pixel alpha
            point_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(point_surface, (*color, alpha), (size, size), size)
            screen.blit(point_surface, (int(x) - size, int(y) - size))
            # get the margin of the point black with thick line
            pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), size + 1, 3)

    def plot_lines(data, color_map, alpha=128):
        for i in range(1, len(data)):
            color = color_map(i / len(data))
            color_with_alpha = (*color, alpha)
            start_pos = (int(data[i - 1][0]), int(data[i - 1][1]))
            end_pos = (int(data[i][0]), int(data[i][1]))
            line_surface = pygame.Surface((1080, 1920), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color_with_alpha, start_pos, end_pos, 4)
            screen.blit(line_surface, (0, 0))

    def plot_heatmap(data, screen, keyboard_img):
        heatmap = np.zeros((1920, 1080), dtype=np.float32)

        # Accumulate heat at gaze points, scaled by duration
        for x, y, dur in data:
            x = max(0, min(int(x), 1079))
            y = max(0, min(int(y), 1919))
            heatmap[y, x] += dur  # Adding the duration to the heatmap

        # Apply a larger Gaussian blur to make heat areas bigger
        heatmap = cv2.GaussianBlur(heatmap, (651, 651), 0)  # Increased kernel size

        # Normalize the heatmap to have values between 0 and 255
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)

        # Apply color mapping (convert grayscale to color)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert the heatmap to an RGBA image for blending
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)

        # Set the alpha channel based on the intensity, making low values transparent
        alpha = 0.9  # Adjust transparency here
        heatmap_color[:, :, 3] = (heatmap * alpha).astype(np.uint8)  # Alpha based on heatmap intensity

        # Make the background (low-intensity areas) fully transparent
        transparency_threshold = 10  # Threshold to control transparency
        heatmap_color[:, :, 3][heatmap < transparency_threshold] = 0

        # Convert the array to a Pygame surface
        heatmap_surface = pygame.image.frombuffer(heatmap_color.tobytes(), heatmap_color.shape[1::-1], "RGBA")

        # First blit the keyboard image
        screen.blit(keyboard_img, (0, 0))

        # Then overlay the heatmap with transparency
        screen.blit(heatmap_surface, (0, 0))

    base_index = str(index).split('_')[0]
    index_dir = osp.join(img_output_dir, base_index)
    sentence_path = osp.join(DEFAULT_DATA_DIR, 'Sentences.csv')
    sentence_df = pd.read_csv(sentence_path)
    # TODO: currently only for the human data
    sentence_id = int(index.split('_')[1])
    if not osp.exists(index_dir):
        os.makedirs(index_dir)
    input_str = rebuild_input(typing_data)
    font_input = pygame.freetype.SysFont('Arial', 48)
    font_metrics = pygame.freetype.SysFont('Arial', 36)
    # Plot and save typing log + ground truth gaze
    screen.blit(keyboard_img, (0, 0))
    # plot_points(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
    # plot_lines(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))

    # modify the simulation data
    if int(index.split("_")[0]) >= 200:
        gaze_data['x'] = gaze_data.apply(lambda row: row['x'] - 200 if row['y'] < 400 else row['x'], axis=1)

    plot_points(gaze_data[['x', 'y', 'duration']].values, lambda x: color_map_ground_truth_gaze(x))
    plot_lines(gaze_data[['x', 'y', 'duration']].values, lambda x: color_map_ground_truth_gaze(x))
    fixation_counts, fixation_duration, gaze_shifts, time_ratio_on_keyboard, time_ratio_on_text_entry = calculate_single_trail_gaze_metrics(
        gaze_data)
    if visualize_text:
        font_metrics.render_to(screen, (10, 300), f'Fixation Counts: {fixation_counts}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 350), f'Fixation Duration: {fixation_duration:.2f}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 400), f'Gaze Shifts: {gaze_shifts}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 450), f'Time Ratio on Keyboard: {time_ratio_on_keyboard:.2f}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 500), f'Time Ratio on Text Entry: {time_ratio_on_text_entry:.2f}',
                               (0, 0, 0))
        if visualize_user_params and parameter is not None:
            font_metrics.render_to(screen, (10, 550), f'User Parameters: {parameter}', (0, 0, 0))
    font_input.render_to(screen, (10, 150), input_str, (0, 0, 0))
    pygame.image.save(screen, osp.join(index_dir, f'index_{index}_ground_truth.png'))

    # Plot and save typing log + predicted gaze
    screen.blit(keyboard_img, (0, 0))
    # plot_points(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
    # plot_lines(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
    font_input.render_to(screen, (10, 150), input_str, (0, 0, 0))
    plot_points(predicted_gaze_data, lambda x: color_map_ground_truth_gaze(x))
    plot_lines(predicted_gaze_data, lambda x: color_map_ground_truth_gaze(x))
    # make predicted_gaze_data_df from predicted_gaze_data, with the same index as gaze_data
    predicted_gaze_data_df = pd.DataFrame(predicted_gaze_data, columns=['x', 'y', 'duration'])
    predicted_gaze_data_df['index'] = index
    predict_fixation_counts, predict_fixation_duration, predict_gaze_shifts, \
    predict_time_ratio_on_keyboard, predict_time_ratio_on_text_entry = calculate_single_trail_gaze_metrics(
        predicted_gaze_data_df)
    if visualize_text:
        font_metrics.render_to(screen, (10, 300), f'Fixation Counts: {predict_fixation_counts}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 350), f'Fixation Duration: {predict_fixation_duration:.2f}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 400), f'Gaze Shifts: {predict_gaze_shifts}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 450), f'Time Ratio on Keyboard: {predict_time_ratio_on_keyboard:.2f}',
                               (0, 0, 0))
        font_metrics.render_to(screen, (10, 500), f'Time Ratio on Text Entry: {predict_time_ratio_on_text_entry:.2f}',
                               (0, 0, 0))
        if visualize_user_params and parameter is not None:
            font_metrics.render_to(screen, (10, 550), f'User Parameters: {parameter}', (0, 0, 0))

    pygame.image.save(screen, osp.join(index_dir, f'index_{index}_predicted.png'))

    screen.blit(keyboard_img, (0, 0))
    # add the input string to the image on the top left, a litter bit below the top, with big black font

    font_input.render_to(screen, (10, 150), input_str, (0, 0, 0))
    plot_points(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
    plot_lines(typing_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
    if int(base_index) < 200 and visualize_text:
        iki, num_backspaces, wpm, char_error_rate = calculate_single_trail_finger_metrics(typing_data, sentence_df,
                                                                                          sentence_id)
        font_metrics.render_to(screen, (10, 300), f'IKI: {iki:.2f}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 350), f'Number of Backspaces: {num_backspaces}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 400), f'WPM: {wpm:.2f}', (0, 0, 0))
        font_metrics.render_to(screen, (10, 450), f'Character Error Rate: {char_error_rate:.2f}', (0, 0, 0))
    font_input.render_to(screen, (10, 150), input_str, (0, 0, 0))
    pygame.image.save(screen, osp.join(index_dir, f'index_{index}_typing_log.png'))

    # plot heatmap for predicted gaze
    screen.blit(keyboard_img, (0, 0))
    plot_heatmap(predicted_gaze_data, screen, keyboard_img)
    pygame.image.save(screen, osp.join(index_dir, f'index_{index}_predicted_heatmap.png'))

    # plot heatmap for ground truth gaze
    screen.blit(keyboard_img, (0, 0))
    plot_heatmap(gaze_data[['x', 'y', 'duration']].values, screen, keyboard_img)
    pygame.image.save(screen, osp.join(index_dir, f'index_{index}_ground_truth_heatmap.png'))

    pygame.quit()


def generate_color_legend(output_path, color_map, image_size=(40, 1000)):
    pygame.init()
    screen = pygame.Surface(image_size)
    width, height = image_size

    # Create the gradient
    for i in range(height):
        color = color_map(i / height)
        pygame.draw.line(screen, color, (0, i), (width, i))

    # Rotate the image
    screen = pygame.transform.rotate(screen, 90)

    # Add time direction arrow and label
    arrow_length = height // 4
    arrow_thickness = 5
    font = pygame.freetype.SysFont('Arial', 24)

    # pygame.draw.line(screen, (255, 255, 255), (height // 2 - arrow_length, width - 20),
    #                  (height // 2 + arrow_length, width - 20), arrow_thickness)
    # pygame.draw.polygon(screen, (255, 255, 255),
    #                     [(height // 2 + arrow_length, width - 20), (height // 2 + arrow_length - 10, width - 10),
    #                      (height // 2 + arrow_length - 10, width - 30)])

    # Draw the label with white color
    # font.render_to(screen, (height // 2 + arrow_length + 10, width - 30), "Time", (255, 255, 255))

    # Save the image
    pygame.image.save(screen, output_path)
    pygame.quit()


def color_map_typing_log(x):
    return (255, int((1 - x) * 255), 0)


def color_map_ground_truth_gaze(x):
    return (0, int((1 - x) * 255), 255)


def color_map_predicted_gaze(x):
    return (0, 255, int((1 - x) * 255))


def visualization(model_type='transformer', max_pred_len=32, loss_type='combined', data_use='human',
                  fpath_header='train', amortized_inference=False, visualize_text=True, visualize_user_params=False):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        include_key=False, include_duration=True, max_pred_len=max_pred_len, data_use=data_use,
        fpath_header=fpath_header,
        calculate_params=amortized_inference)
    include_duration = True
    if not amortized_inference:
        test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataset = TypingGazeDataset(X_train, y_train, masks_x_train, masks_y_train, indices_train)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        input_dim = X_train.shape[2]
        output_dim = 3 if include_duration else 2

        model = TransformerModel(input_dim, output_dim).to(device)
        model_filename = f'{model_type}_{loss_type}_{data_use}_model_without_key_with_duration.pth'
    else:
        user_params_train = X_train[:, :, 3:6]  # Shape (batch_size, seq_len, 3)
        features_train = X_train[:, :, :3]  # Shape (batch_size, seq_len, 3)

        # Similarly, for the test set
        user_params_test = X_test[:, :, 3:6]  # Shape (batch_size, seq_len, 3)
        features_test = X_test[:, :, :3]  # Shape (batch_size, seq_len, 3)

        input_dim = features_train.shape[2]  # Update input_dim to include the encoded keys if included
        user_params_dim = user_params_train.shape[2]
        output_dim = 3 if include_duration else 2

        if model_type == "transformer":
            model = GooglyeyesModel(input_dim=input_dim, output_dim=output_dim,
                                    user_param_dim=user_params_dim, dropout=0.1).to(device)
        else:
            raise ValueError("Only transformer model is supported for amortized inference")

        model_filename = f'amortized_inference_{model_type}_{data_use}_model.pth'

        test_dataset = TypingGazeInferenceDataset(features_test, y_test, masks_x_test, masks_y_test, indices_test,
                                                  user_params_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataset = TypingGazeInferenceDataset(features_train, y_train, masks_x_train, masks_y_train, indices_train,
                                                   user_params_train)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Use a larger batch size
    last_modified_time = time.ctime(os.path.getmtime(osp.join(saved_model_dir, model_filename)))
    print("Loading model from {}".format(osp.join(saved_model_dir, model_filename)))
    print(f"Model was last modified on: {last_modified_time}")
    model.load_state_dict(torch.load(osp.join(saved_model_dir, model_filename), map_location=device))
    model.eval()

    def create_padding_mask(seq):
        seq = seq == 0
        return seq

    first_gaze_duration = 25
    with torch.no_grad():
        if not amortized_inference:
            for inputs, targets, masks_x, masks_y, index in tqdm(test_loader):
                # if int(index[0].split('_')[0]) <= 200:    # only for simulated data visualization
                #     continue
                # get the max trailtime in typing_data
                current_typing_data = typing_data[typing_data['index'] == index[0]]
                max_trailtime = current_typing_data['trailtime'].max()
                inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                    device), masks_y.to(
                    device)
                inputs = inputs.squeeze(0)  # Remove batch dimension
                targets = targets.squeeze(0)  # Remove batch dimension
                masks_x = masks_x.squeeze(0)  # Remove batch dimension
                masks_y = masks_y.squeeze(0)  # Remove batch dimension

                src_mask = create_padding_mask(masks_x.unsqueeze(0)) if model_type == "transformer" else None
                gaze_mean, gaze_log_std, padding_outputs = model(inputs.unsqueeze(0),
                                                                 src_mask) if model_type == "transformer" else model(
                    inputs.unsqueeze(0))
                gaze_std = torch.exp(gaze_log_std)
                epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                gaze_outputs = gaze_mean + gaze_std * epsilon
                gaze_outputs = gaze_outputs.squeeze(0)
                padding_outputs = padding_outputs.squeeze(0)

                # Convert to numpy arrays
                gaze_outputs_np = gaze_outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                masks_y_np = masks_y.cpu().numpy()

                # Reshape to 2D before inverse transforming
                gaze_outputs_reshaped = gaze_outputs_np.reshape(-1, output_dim)
                targets_reshaped = targets_np.reshape(-1, output_dim)

                # Inverse transform
                gaze_outputs_inv = scaler_y.inverse_transform(gaze_outputs_reshaped)
                targets_inv = scaler_y.inverse_transform(targets_reshaped)

                # Limit the prediction and target lengths to max_pred_len
                valid_gaze_outputs = gaze_outputs_inv[:max_pred_len]
                valid_targets = targets_inv[:max_pred_len]
                valid_masks_y_np = masks_y_np[:max_pred_len]

                # Use predicted padding instead of original mask
                predicted_masks_y_np = (padding_outputs[:max_pred_len] > 0.5).cpu().numpy()

                # Remove padding using the predicted mask
                valid_gaze_outputs = valid_gaze_outputs[predicted_masks_y_np == 1]
                valid_targets = valid_targets[valid_masks_y_np == 1]

                # remove those gaze with sum of duration greater than max_trailtime
                if valid_gaze_outputs[:, 2].sum() > max_trailtime:
                    valid_gaze_outputs = valid_gaze_outputs[valid_gaze_outputs[:, 2].cumsum() <= max_trailtime]

                typing_log = typing_data[typing_data['index'] == index[0]]
                gaze_log = gaze_data[gaze_data['index'] == index[0]]

                # For visualization, the first duration of typing log is set to 300
                typing_log.loc[typing_log.index[0], 'duration'] = 300
                visualize_gaze_and_typing(index[0], typing_log, gaze_log, valid_gaze_outputs, model_type=model_type,
                                          data_use=data_use, amortized_inference=False, visualize_text=visualize_text,
                                          parameter=None)

        else:
            for inputs, targets, masks_x, masks_y, index, user_params in tqdm(test_loader):
                # get the max trailtime in typing_data
                current_typing_data = typing_data[typing_data['index'] == index[0]]
                max_trailtime = current_typing_data['trailtime'].max()
                inputs, targets, masks_x, masks_y, user_params = inputs.to(device), targets.to(device), masks_x.to(
                    device), masks_y.to(device), user_params.to(device)
                inputs = inputs.squeeze(0)  # Remove batch dimension
                targets = targets.squeeze(0)  # Remove batch dimension
                masks_x = masks_x.squeeze(0)  # Remove batch dimension
                masks_y = masks_y.squeeze(0)  # Remove batch dimension
                user_params = user_params.squeeze(0)

                src_mask = create_padding_mask(masks_x.unsqueeze(0)) if model_type == "transformer" else None
                gaze_mean, gaze_log_std, padding_outputs = model(inputs.unsqueeze(0), user_params.unsqueeze(0),
                                                                 src_mask) if model_type == "transformer" else model(
                    inputs.unsqueeze(0))
                gaze_std = torch.exp(gaze_log_std)
                epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                gaze_outputs = gaze_mean + gaze_std * epsilon
                gaze_outputs = gaze_outputs.squeeze(0)
                padding_outputs = padding_outputs.squeeze(0)

                # Convert to numpy arrays
                gaze_outputs_np = gaze_outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                masks_y_np = masks_y.cpu().numpy()

                # Reshape to 2D before inverse transforming
                gaze_outputs_reshaped = gaze_outputs_np.reshape(-1, output_dim)
                targets_reshaped = targets_np.reshape(-1, output_dim)

                # Inverse transform
                gaze_outputs_inv = scaler_y.inverse_transform(gaze_outputs_reshaped)
                targets_inv = scaler_y.inverse_transform(targets_reshaped)

                # Limit the prediction and target lengths to max_pred_len
                valid_gaze_outputs = gaze_outputs_inv[:max_pred_len]
                valid_targets = targets_inv[:max_pred_len]
                valid_masks_y_np = masks_y_np[:max_pred_len]

                # Use predicted padding instead of original mask
                predicted_masks_y_np = (padding_outputs[:max_pred_len] > 0.5).cpu().numpy()

                # Remove padding using the predicted mask
                valid_gaze_outputs = valid_gaze_outputs[predicted_masks_y_np == 1]
                valid_targets = valid_targets[valid_masks_y_np == 1]

                # remove those gaze with sum of duration greater than max_trailtime
                if valid_gaze_outputs[:, 2].sum() > max_trailtime:
                    valid_gaze_outputs = valid_gaze_outputs[valid_gaze_outputs[:, 2].cumsum() <= max_trailtime]

                typing_log = typing_data[typing_data['index'] == index[0]]
                gaze_log = gaze_data[gaze_data['index'] == index[0]]

                # For visualization, the first duration of typing log is set to 300
                typing_log.loc[typing_log.index[0], 'duration'] = 300

                user_params_np = user_params.cpu().numpy()[0]

                visualize_gaze_and_typing(index[0], typing_log, gaze_log, valid_gaze_outputs, model_type=model_type,
                                          data_use=data_use, amortized_inference=True, parameter=user_params_np,
                                          visualize_text=visualize_text, visualize_user_params=visualize_user_params)


def video_generation(model_type='transformer', max_pred_len=32, loss_type='combined', data_use='human',
                     fpath_header='train', amortized_inference=False, gaze_data_source='predicted'):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        include_key=False, include_duration=True, max_pred_len=max_pred_len, data_use=data_use,
        fpath_header=fpath_header,
        calculate_params=amortized_inference)
    include_duration = True
    if not amortized_inference:
        test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataset = TypingGazeDataset(X_train, y_train, masks_x_train, masks_y_train, indices_train)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        input_dim = X_train.shape[2]
        output_dim = 3 if include_duration else 2

        model = TransformerModel(input_dim, output_dim).to(device)

        model_filename = f'{model_type}_{loss_type}_{data_use}_model_without_key_with_duration.pth'
    else:
        user_params_train = X_train[:, :, 3:6]  # Shape (batch_size, seq_len, 3)
        features_train = X_train[:, :, :3]  # Shape (batch_size, seq_len, 3)

        # Similarly, for the test set
        user_params_test = X_test[:, :, 3:6]  # Shape (batch_size, seq_len, 3)
        features_test = X_test[:, :, :3]  # Shape (batch_size, seq_len, 3)

        input_dim = features_train.shape[2]  # Update input_dim to include the encoded keys if included
        user_params_dim = user_params_train.shape[2]
        output_dim = 3 if include_duration else 2

        if model_type == "transformer":
            model = GooglyeyesModel(input_dim=input_dim, output_dim=output_dim,
                                    user_param_dim=user_params_dim, dropout=0.1).to(device)
        else:
            raise ValueError("Only transformer model is supported for amortized inference")
        model_filename = f'amortized_inference_{model_type}_{data_use}_model.pth'

        test_dataset = TypingGazeInferenceDataset(features_test, y_test, masks_x_test, masks_y_test, indices_test,
                                                  user_params_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataset = TypingGazeInferenceDataset(features_train, y_train, masks_x_train, masks_y_train, indices_train,
                                                   user_params_train)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Use a larger batch size
    last_modified_time = time.ctime(os.path.getmtime(osp.join(saved_model_dir, model_filename)))
    print("Loading model from {}".format(osp.join(saved_model_dir, model_filename)))
    print(f"Model was last modified on: {last_modified_time}")
    model.load_state_dict(torch.load(osp.join(saved_model_dir, model_filename), map_location=device))
    model.eval()

    def create_padding_mask(seq):
        seq = seq == 0
        return seq

    with torch.no_grad():
        if not amortized_inference:
            for inputs, targets, masks_x, masks_y, index in tqdm(test_loader):
                # get the max trailtime in typing_data
                current_typing_data = typing_data[typing_data['index'] == index[0]]
                max_trailtime = current_typing_data['trailtime'].max()
                inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                    device), masks_y.to(
                    device)
                inputs = inputs.squeeze(0)  # Remove batch dimension
                targets = targets.squeeze(0)  # Remove batch dimension
                masks_x = masks_x.squeeze(0)  # Remove batch dimension
                masks_y = masks_y.squeeze(0)  # Remove batch dimension

                src_mask = create_padding_mask(masks_x.unsqueeze(0)) if model_type == "transformer" else None
                gaze_mean, gaze_log_std, padding_outputs = model(inputs.unsqueeze(0),
                                                                 src_mask) if model_type == "transformer" else model(
                    inputs.unsqueeze(0))
                gaze_std = torch.exp(gaze_log_std)
                epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                gaze_outputs = gaze_mean + gaze_std * epsilon
                gaze_outputs = gaze_outputs.squeeze(0)
                padding_outputs = padding_outputs.squeeze(0)

                # Convert to numpy arrays
                gaze_outputs_np = gaze_outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                masks_y_np = masks_y.cpu().numpy()

                # Reshape to 2D before inverse transforming
                gaze_outputs_reshaped = gaze_outputs_np.reshape(-1, output_dim)
                targets_reshaped = targets_np.reshape(-1, output_dim)

                # Inverse transform
                gaze_outputs_inv = scaler_y.inverse_transform(gaze_outputs_reshaped)
                targets_inv = scaler_y.inverse_transform(targets_reshaped)

                # Limit the prediction and target lengths to max_pred_len
                valid_gaze_outputs = gaze_outputs_inv[:max_pred_len]
                valid_targets = targets_inv[:max_pred_len]
                valid_masks_y_np = masks_y_np[:max_pred_len]

                # Use predicted padding instead of original mask
                predicted_masks_y_np = (padding_outputs[:max_pred_len] > 0.5).cpu().numpy()

                # Remove padding using the predicted mask
                valid_gaze_outputs = valid_gaze_outputs[predicted_masks_y_np == 1]
                valid_targets = valid_targets[valid_masks_y_np == 1]

                # remove those gaze with sum of duration greater than max_trailtime
                if valid_gaze_outputs[:, 2].sum() > max_trailtime:
                    valid_gaze_outputs = valid_gaze_outputs[valid_gaze_outputs[:, 2].cumsum() <= max_trailtime]

                typing_log = typing_data[typing_data['index'] == index[0]]
                gaze_log = gaze_data[gaze_data['index'] == index[0]]

                # For visualization, the first duration of typing log is set to 300
                typing_log.loc[typing_log.index[0], 'duration'] = 300

                video_producing(index[0], typing_log, valid_gaze_outputs, model_type=model_type,
                                data_use=data_use, amortized_inference=True, gaze_data_source=gaze_data_source)
        else:
            for inputs, targets, masks_x, masks_y, index, user_params in tqdm(test_loader):
                # get the max trailtime in typing_data
                current_typing_data = typing_data[typing_data['index'] == index[0]]
                max_trailtime = current_typing_data['trailtime'].max()
                inputs, targets, masks_x, masks_y, user_params = inputs.to(device), targets.to(device), masks_x.to(
                    device), masks_y.to(device), user_params.to(device)
                inputs = inputs.squeeze(0)  # Remove batch dimension
                targets = targets.squeeze(0)  # Remove batch dimension
                masks_x = masks_x.squeeze(0)  # Remove batch dimension
                masks_y = masks_y.squeeze(0)  # Remove batch dimension
                user_params = user_params.squeeze(0)

                src_mask = create_padding_mask(masks_x.unsqueeze(0)) if model_type == "transformer" else None
                gaze_mean, gaze_log_std, padding_outputs = model(inputs.unsqueeze(0), user_params.unsqueeze(0),
                                                                 src_mask) if model_type == "transformer" else model(
                    inputs.unsqueeze(0))
                gaze_std = torch.exp(gaze_log_std)
                epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                gaze_outputs = gaze_mean + gaze_std * epsilon
                gaze_outputs = gaze_outputs.squeeze(0)
                padding_outputs = padding_outputs.squeeze(0)

                # Convert to numpy arrays
                gaze_outputs_np = gaze_outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                masks_y_np = masks_y.cpu().numpy()

                # Reshape to 2D before inverse transforming
                gaze_outputs_reshaped = gaze_outputs_np.reshape(-1, output_dim)
                targets_reshaped = targets_np.reshape(-1, output_dim)

                # Inverse transform
                gaze_outputs_inv = scaler_y.inverse_transform(gaze_outputs_reshaped)
                targets_inv = scaler_y.inverse_transform(targets_reshaped)

                # Limit the prediction and target lengths to max_pred_len
                valid_gaze_outputs = gaze_outputs_inv[:max_pred_len]
                valid_targets = targets_inv[:max_pred_len]
                valid_masks_y_np = masks_y_np[:max_pred_len]

                # Use predicted padding instead of original mask
                predicted_masks_y_np = (padding_outputs[:max_pred_len] > 0.5).cpu().numpy()

                # Remove padding using the predicted mask
                valid_gaze_outputs = valid_gaze_outputs[predicted_masks_y_np == 1]
                valid_targets = valid_targets[valid_masks_y_np == 1]

                # remove those gaze with sum of duration greater than max_trailtime
                if valid_gaze_outputs[:, 2].sum() > max_trailtime:
                    valid_gaze_outputs = valid_gaze_outputs[valid_gaze_outputs[:, 2].cumsum() <= max_trailtime]

                typing_log = typing_data[typing_data['index'] == index[0]]
                gaze_log = gaze_data[gaze_data['index'] == index[0]]

                # For visualization, the first duration of typing log is set to 300
                typing_log.loc[typing_log.index[0], 'duration'] = 300

                video_producing(index[0], typing_log, valid_gaze_outputs, model_type=model_type,
                                data_use=data_use, amortized_inference=True, gaze_data_source=gaze_data_source)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Typing Log and Gaze Movements")
    parser.add_argument("--model_type", choices=["transformer"], default="transformer",
                        help="Type of model to use for prediction")
    parser.add_argument("--max_pred_len", type=int, default=32, help="Maximum number of gaze data points to predict")
    parser.add_argument("--loss_type", type=str, choices=['combined'], default='combined',
                        help="Loss function to use for training")
    parser.add_argument("--data_use", type=str, choices=['both', 'human'], default='human',
                        help="Use human data, simulated data, or both")
    parser.add_argument("--fpath_header", type=str, default='final_distribute', help='File path header for data use')
    parser.add_argument("--amortized-inference", action="store_true", help="Use amortized inference", default=False)
    parser.add_argument("--visualize_text", action="store_true", help="Visualize text", default=False)
    parser.add_argument("--visualize_user_params", action="store_true", help="Visualize user params", default=True)
    parser.add_argument("--gaze_data_source", type=str, choices=['ground_truth', 'predicted'], default='predicted',
                        help="Source of gaze data to visualize")
    parser.add_argument("--method", type=str, default='video', help='visualization method', choices=['video', 'image'])
    args = parser.parse_args()
    print("Visualizing with data use: ", args.data_use)
    print("Using amortized inference:", args.amortized_inference)
    if args.amortized_inference:
        sub_dir = "amortized_inference_" + args.model_type + "_" + args.data_use
    else:
        sub_dir = args.model_type + "_" + args.data_use

    global img_output_dir
    img_output_dir = osp.join(img_output_dir, sub_dir)

    global video_output_dir
    video_output_dir = osp.join(video_output_dir, sub_dir)

    if args.method == 'image':
        if osp.exists(img_output_dir):
            shutil.rmtree(img_output_dir)
        os.makedirs(img_output_dir)
    elif args.method == 'video':
        if osp.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir)

    if args.method == 'image':
        visualization(args.model_type, args.max_pred_len, args.loss_type, args.data_use,
                      args.fpath_header, args.amortized_inference, args.visualize_text, args.visualize_user_params)
    else:
        video_generation(args.model_type, args.max_pred_len, args.loss_type, args.data_use,
                         args.fpath_header, args.amortized_inference, args.gaze_data_source)


def plot_and_save_trails(typing_data, gaze_data, output_dir, max_len=32, min_len=16):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def filter_by_length(data, min_length, max_length):
        return data.groupby('index').filter(lambda x: min_length < len(x) <= max_length)

    def trim_to_length(data, length):
        return data.groupby('index').apply(lambda x: x.head(length)).reset_index(drop=True)

    def plot_trails_for_index(index, typing_data, gaze_data, save_path):
        pygame.init()
        screen = pygame.display.set_mode((1080, 1920), pygame.HIDDEN)
        pygame.display.set_caption(f"Trails for Index {index}")

        keyboard_img = pygame.image.load(keyboard_image)  # Update this path to your keyboard image
        keyboard_img = pygame.transform.scale(keyboard_img, (1080, 1920))

        def plot_points(data, color_map, alpha=255):
            for i, point in enumerate(data):
                x, y, dur = point
                color = color_map(i / len(data))
                size = int(3 + dur * 0.04)
                point_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(point_surface, (*color, alpha), (size, size), size)
                screen.blit(point_surface, (int(x) - size, int(y) - size))
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), size + 1, 3)

        def plot_lines(data, color_map, alpha=128):
            for i in range(2, len(data)):
                color = color_map(i / len(data))
                color_with_alpha = (*color, alpha)
                start_pos = (int(data[i - 1][0]), int(data[i - 1][1]))
                end_pos = (int(data[i][0]), int(data[i][1]))
                line_surface = pygame.Surface((1080, 1920), pygame.SRCALPHA)
                pygame.draw.line(line_surface, color_with_alpha, start_pos, end_pos, 4)
                screen.blit(line_surface, (0, 0))

        # Filter data for the specific index
        typing_index_data = typing_data[typing_data['index'] == index]
        gaze_index_data = gaze_data[gaze_data['index'] == index]

        screen.blit(keyboard_img, (0, 0))
        if not gaze_index_data.empty:
            plot_lines(gaze_index_data[['x', 'y', 'duration']].values, lambda x: color_map_ground_truth_gaze(x))
            plot_points(gaze_index_data[['x', 'y', 'duration']].values, lambda x: color_map_ground_truth_gaze(x))
        if not typing_index_data.empty:
            plot_lines(typing_index_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))
            plot_points(typing_index_data[['x', 'y', 'duration']].values, lambda x: color_map_typing_log(x))

        pygame.image.save(screen, save_path)
        pygame.quit()

    # Filter and trim data
    typing_long = typing_data.groupby('index').filter(lambda x: len(x) > max_len)
    gaze_long = gaze_data.groupby('index').filter(lambda x: len(x) > max_len)
    typing_mid = filter_by_length(typing_data, min_len, max_len)
    gaze_mid = filter_by_length(gaze_data, min_len, max_len)

    # Group 1: both gaze and typing trail length > 32
    indices_both_long = set(typing_long['index']).intersection(set(gaze_long['index']))

    # Group 2: only gaze trail length > 32, while typing trail length between 16 and 32
    indices_gaze_long_typing_mid = set(gaze_long['index']).intersection(set(typing_mid['index']))

    # Group 3: only typing trail length > 32, while gaze trail length between 16 and 32
    indices_typing_long_gaze_mid = set(typing_long['index']).intersection(set(gaze_mid['index']))

    def process_and_plot(indices, typing_data, gaze_data, group_name):
        for index in indices:
            save_path = os.path.join(output_dir, f'{index}_{group_name}_before_cut.png')
            plot_trails_for_index(index, typing_data, gaze_data, save_path)
            trimmed_typing_data = trim_to_length(typing_data[typing_data['index'] == index], max_len)
            trimmed_gaze_data = trim_to_length(gaze_data[gaze_data['index'] == index], max_len)
            save_path = os.path.join(output_dir, f'{index}_{group_name}_after_cut.png')
            plot_trails_for_index(index, trimmed_typing_data, trimmed_gaze_data, save_path)

    process_and_plot(indices_both_long, typing_data, gaze_data, 'both_long')
    process_and_plot(indices_gaze_long_typing_mid, typing_data, gaze_data, 'gaze_long_typing_mid')
    process_and_plot(indices_typing_long_gaze_mid, typing_data, gaze_data, 'typing_long_gaze_mid')


def video_producing(index, typing_data, gaze_log, screen_size=(1080, 1920), fps=30,
                    model_type='transformer', data_use='human', amortized_inference=True,
                    gaze_data_source='ground_truth'):
    pygame.init()
    first_gaze_duration = 25
    # Set up Pygame screen
    screen = pygame.Surface(screen_size)

    # Load the keyboard image
    keyboard_img = pygame.image.load(keyboard_image)
    keyboard_img = pygame.transform.scale(keyboard_img, screen_size)

    # Get the screen size
    width, height = screen_size

    base_index = str(index).split('_')[0]
    index_dir = osp.join(video_output_dir, gaze_data_source, base_index)
    sentence_path = osp.join(DEFAULT_DATA_DIR, 'Sentences.csv')
    sentence_df = pd.read_csv(sentence_path)
    sentence_id = int(index.split('_')[1])
    if not osp.exists(index_dir):
        os.makedirs(index_dir)

    output_file = osp.join(index_dir, f'index_{index}_{gaze_data_source}.mp4')

    input_str = rebuild_input(typing_data)

    # Extract x, y, and duration from typing and gaze data
    typing_data = typing_data[['x', 'y', 'duration']].values
    if gaze_data_source == 'ground_truth':
        gaze_data = gaze_log[['x', 'y', 'duration']].values
    else:
        gaze_data = gaze_log
    num_typing_points = len(typing_data)
    num_gaze_points = len(gaze_data)

    # Calculate cumulative time stamps for each point in typing and gaze data
    time_stamps_typing = np.cumsum([0] + [d for _, _, d in typing_data[1:]])
    time_stamps_gaze = np.cumsum([0] + [d for _, _, d in gaze_data[1:]])

    # Calculate the total duration in milliseconds (take max of typing and gaze durations)
    total_duration = max(time_stamps_typing[-1], time_stamps_gaze[-1]) + first_gaze_duration

    # Calculate number of frames for the entire video
    total_frames = int((total_duration / 1000) * fps)
    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    def plot_points(data, color_map, alpha=255):
        for i, point in enumerate(data):
            x, y, dur = point
            color = color_map(i / len(data))  # Use the color map
            size = int(3 + dur * 0.04)  # Scale point size by duration
            # Create a surface with per-pixel alpha
            point_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(point_surface, (*color, alpha), (size, size), size)
            screen.blit(point_surface, (int(x) - size, int(y) - size))
            # Get the margin of the point black with a thick line
            pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), size + 1, 3)

    def plot_lines(data, color_map, alpha=128):
        for i in range(1, len(data)):
            color = color_map(i / len(data))  # Use the color map
            color_with_alpha = (*color, alpha)
            start_pos = (int(data[i - 1][0]), int(data[i - 1][1]))
            end_pos = (int(data[i][0]), int(data[i][1]))
            line_surface = pygame.Surface((1080, 1920), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color_with_alpha, start_pos, end_pos, 4)
            screen.blit(line_surface, (0, 0))

    def get_typed_str(input_str, typed_str, typed_str_index):
        typed_char = input_str[typed_str_index]
        if len(typed_str) == 0 and typed_char == '<':
            pass
        elif typed_char == '<':
            typed_str = typed_str[:-1]
        else:
            typed_str += typed_char

        return typed_str

    # Main loop for creating frames
    current_frame = -15
    typed_str_index = 1

    typed_str = get_typed_str(input_str, '', 0)

    font_input = pygame.freetype.SysFont('Arial', 48)
    typing_point_index = 0
    while current_frame < total_frames:
        if current_frame < 0:
            screen.blit(keyboard_img, (0, 0))
            frame_data = pygame.surfarray.array3d(screen)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            # Write frame to video
            video.write(cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
            current_frame += 1
            continue
        # Clear the screen and set the keyboard as the background
        screen.blit(keyboard_img, (0, 0))

        # Calculate the current time in milliseconds
        current_time = (current_frame / fps) * 1000

        # Ensure the first point in typing log is drawn immediately at t=0
        if current_frame > 0:
            plot_points(typing_data[:1], lambda x: color_map_typing_log(x))
            plot_points(gaze_data[:1], lambda x: color_map_ground_truth_gaze(x))  # For gaze

        for i in range(1, num_typing_points):
            # Draw lines for typing log
            if time_stamps_typing[i - 1] <= current_time and time_stamps_typing[i] <= current_time:
                typing_point_index = max(typing_point_index, i)
                plot_lines(typing_data[:i + 1], lambda x: color_map_typing_log(x))

            # Draw dynamic line for typing
            if time_stamps_typing[i - 1] <= current_time < time_stamps_typing[i]:
                progress_ratio = (current_time - time_stamps_typing[i - 1]) / (
                        time_stamps_typing[i] - time_stamps_typing[i - 1])
                current_x = int(typing_data[i - 1][0] + progress_ratio * (typing_data[i][0] - typing_data[i - 1][0]))
                current_y = int(typing_data[i - 1][1] + progress_ratio * (typing_data[i][1] - typing_data[i - 1][1]))
                pygame.draw.line(screen, (255, 0, 0), (int(typing_data[i - 1][0]), int(typing_data[i - 1][1])),
                                 (current_x, current_y), 2)

        for i in range(1, num_gaze_points):
            if time_stamps_gaze[i] <= current_time:
                # Draw the line between the previous and current point after the current point is reached
                plot_lines(gaze_data[:i + 1], lambda x: color_map_ground_truth_gaze(x))

        # Then, draw the points on top of the lines (both typing and gaze)
        for i in range(num_typing_points):
            if time_stamps_typing[i] <= current_time:
                plot_points(typing_data[:i + 1], lambda x: color_map_typing_log(x))

        for i in range(num_gaze_points):
            if time_stamps_gaze[i] <= current_time:
                plot_points(gaze_data[:i + 1], lambda x: color_map_ground_truth_gaze(x))

        # Convert Pygame surface to numpy array for OpenCV
        if typing_point_index >= typed_str_index:
            typed_str = get_typed_str(input_str, typed_str, typed_str_index)
            typed_str_index += 1
        font_input.render_to(screen, (10, 150), typed_str, (0, 0, 0))

        frame_data = pygame.surfarray.array3d(screen)
        frame_data = np.transpose(frame_data, (1, 0, 2))

        # Write frame to video
        video.write(cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))

        # Increment the frame counter
        current_frame += 1

    # Finalize the video
    video.release()
    pygame.quit()
    print(f'Video saved as {output_file}')


if __name__ == "__main__":
    main()
