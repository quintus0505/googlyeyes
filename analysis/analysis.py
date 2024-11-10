import sys
import os

# Detect the current directory and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import time
import os.path as osp
import numpy as np
import pygame
import torch
from config import GAZE_INFERENCE_DIR
from src.nets import TransformerModel, GooglyeyesModel, TypingGazeInferenceDataset
from data.data import load_and_preprocess_data, calculate_gaze_metrics
from config import how_we_type_key_coordinate_resized
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from sklearn.metrics.pairwise import cosine_similarity
from correlation_study import plot_distances, FIG_DIR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.nets import TypingGazeDataset

keyboard_image_path = osp.join(GAZE_INFERENCE_DIR, 'figs', 'chi_keyboard.png')
video_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'videos')
saved_model_dir = osp.join(GAZE_INFERENCE_DIR, 'src', 'best_outputs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tail_offset = -1000
head_offset = 1000


def analysis(model_type='transformer', max_pred_len=32, loss_type='combined', data_use='human',
             fpath_header='train', amortized_inference=False, user_index=None, analysis_data_choice='test'):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        include_key=False, include_duration=True, max_pred_len=max_pred_len, max_len=max_pred_len, data_use=data_use,
        fpath_header=fpath_header,
        calculate_params=amortized_inference)
    include_duration = True
    if not amortized_inference:
        img_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'analysis', model_type + '_' + data_use)

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

        img_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'analysis',
                                  'amortized_inference_' + model_type + '_' + data_use)
    last_modified_time = time.ctime(os.path.getmtime(osp.join(saved_model_dir, model_filename)))
    print("Loading model from {}".format(osp.join(saved_model_dir, model_filename)))
    print(f"Model was last modified on: {last_modified_time}")
    model.load_state_dict(torch.load(osp.join(saved_model_dir, model_filename), map_location=device))
    model.eval()
    if analysis_data_choice == 'both':
        img_output_dir = osp.join(GAZE_INFERENCE_DIR, 'figs', 'analysis', 'ground_truth')
    if not osp.exists(img_output_dir):
        os.makedirs(img_output_dir)

    def create_padding_mask(seq):
        seq = seq == 0
        return seq

    all_distances_predict = {}
    all_distances_ground_truth = {}
    all_similarities = {}

    predict_gaze_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration'])
    target_gaze_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration'])
    typing_log_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration', 'key', 'trailtime'])
    ground_truth_all_keys_on_keyboard = {}
    prediction_all_keys_on_keyboard = {}
    ground_truth_all_distances_on_keyboard = {}
    prediction_all_distances_on_keyboard = {}
    ground_truth_ikis_on_keyboard = {}
    prediction_ikis_on_keyboard = {}

    ground_truth_all_keys_on_text_entry = {}
    prediction_all_keys_on_text_entry = {}
    ground_truth_all_distances_on_text_entry = {}
    prediction_all_distances_on_text_entry = {}
    ground_truth_ikis_on_text_entry = {}
    prediction_ikis_on_text_entry = {}
    ikis_distance = {}
    iki_mean_list = []
    if analysis_data_choice == 'both':
        loaders = [train_loader, test_loader]
    elif analysis_data_choice == 'train':
        loaders = [train_loader]
    else:
        loaders = [test_loader]
    first_gaze_duration = 25
    for loader in loaders:
        with torch.no_grad():
            if not amortized_inference:
                for inputs, targets, masks_x, masks_y, index in tqdm(loader):
                    if user_index and index[0].split('_')[0] != user_index:
                        continue
                    inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                        device), masks_y.to(
                        device)
                    inputs = inputs.squeeze(0)
                    targets = targets.squeeze(0)
                    masks_x = masks_x.squeeze(0)
                    masks_y = masks_y.squeeze(0)
                    current_typing_data = typing_data[typing_data['index'] == index[0]]
                    max_trailtime = current_typing_data['trailtime'].max()

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
                    ikis = get_iki_vs_distance(typing_log)
                    for iki, distance in ikis.items():
                        if iki not in ikis_distance:
                            ikis_distance[iki] = []
                        ikis_distance[iki].extend(distance)

                    distances, similarities, gaze_on_keyboard_ratio = distance_and_cosine_similarity_analysis(index[0],
                                                                                                              typing_log,
                                                                                                              valid_targets,
                                                                                                              model_type=model_type,
                                                                                                              data_use=data_use,
                                                                                                              amortized_inference=True,
                                                                                                              gaze_data_source='ground_truth')
                    for offset, dists in distances.items():
                        if offset not in all_distances_ground_truth:
                            all_distances_ground_truth[offset] = []
                        all_distances_ground_truth[offset].extend(dists)
                    # new_img_output_dir = osp.join(img_output_dir, str(index[0]))
                    # plot_distances(distances, gaze_on_keyboard_ratio=0,
                    #                gaze_data_source='ground_truth', save_dir=new_img_output_dir)
                    if analysis_data_choice != 'both':
                        gaze = valid_gaze_outputs
                    else:
                        gaze = valid_targets
                    distances, similarities, gaze_on_keyboard_ratio = distance_and_cosine_similarity_analysis(index[0],
                                                                                                              typing_log,
                                                                                                              gaze,
                                                                                                              model_type=model_type,
                                                                                                              data_use=data_use,
                                                                                                              amortized_inference=True,
                                                                                                              gaze_data_source='predicted')
                    new_img_output_dir = osp.join(img_output_dir, str(index[0]))
                    plot_distances(distances, gaze_on_keyboard_ratio=0,
                                   gaze_data_source='predicted', save_dir=new_img_output_dir)
                    for offset, dists in distances.items():
                        if offset not in all_distances_predict:
                            all_distances_predict[offset] = []
                        all_distances_predict[offset].extend(dists)

                    for offset, sims in similarities.items():
                        if offset not in all_similarities:
                            all_similarities[offset] = []
                        all_similarities[offset].extend(sims)

                    # add the predicted gaze to the dataframe
                    temp_list = []
                    target_gaze_list = []
                    typing_list = []
                    for i, (x, y, duration) in enumerate(valid_gaze_outputs):
                        temp_list.append([index[0], x, y, duration])
                    for i, (x, y, duration) in enumerate(valid_targets):
                        target_gaze_list.append([index[0], x, y, duration])
                    for i, (x, y, duration, key, trailtime) in enumerate(
                            typing_log[['x', 'y', 'duration', 'key', 'trailtime']].values):
                        typing_list.append([index[0], x, y, duration, key, trailtime])
                    current_predict_gaze_df = pd.DataFrame(temp_list, columns=['index', 'x', 'y', 'duration'])
                    predict_gaze_df = pd.concat([predict_gaze_df,
                                                 current_predict_gaze_df],
                                                ignore_index=True)
                    current_target_gaze_df = pd.DataFrame(target_gaze_list,
                                                          columns=['index', 'x', 'y', 'duration'])
                    target_gaze_df = pd.concat([target_gaze_df,
                                                current_target_gaze_df],
                                               ignore_index=True)
                    current_typing_log_df = pd.DataFrame(typing_list,
                                                         columns=['index', 'x', 'y', 'duration', 'key', 'trailtime'])
                    typing_log_df = pd.concat([typing_log_df,
                                               current_typing_log_df],
                                              ignore_index=True)
                    # computing key vs proofreading
                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_predict_gaze_df, typing_df=current_typing_log_df,
                                                       position='keyboard')
                        for key, proofreading in keys.items():
                            if key not in prediction_all_keys_on_keyboard:
                                prediction_all_keys_on_keyboard[key] = []
                            prediction_all_keys_on_keyboard[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with predicted gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_target_gaze_df, typing_df=current_typing_log_df,
                                                       position='keyboard')
                        for key, proofreading in keys.items():
                            if key not in ground_truth_all_keys_on_keyboard:
                                ground_truth_all_keys_on_keyboard[key] = []
                            ground_truth_all_keys_on_keyboard[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with ground truth gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_predict_gaze_df, typing_df=current_typing_log_df,
                                                       position='text_entry')
                        for key, proofreading in keys.items():
                            if key not in prediction_all_keys_on_text_entry:
                                prediction_all_keys_on_text_entry[key] = []
                            prediction_all_keys_on_text_entry[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with predicted gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_target_gaze_df, typing_df=current_typing_log_df,
                                                       position='text_entry')
                        for key, proofreading in keys.items():
                            if key not in ground_truth_all_keys_on_text_entry:
                                ground_truth_all_keys_on_text_entry[key] = []
                            ground_truth_all_keys_on_text_entry[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with ground truth gaze')

                    # computing iki vs proofreading
                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_predict_gaze_df,
                                                                 typing_df=current_typing_log_df, position='keyboard')
                        for iki, proofreading in ikis.items():
                            if iki not in prediction_ikis_on_keyboard:
                                prediction_ikis_on_keyboard[iki] = []
                            prediction_ikis_on_keyboard[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with predicted gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_target_gaze_df,
                                                                 typing_df=current_typing_log_df, position='keyboard')
                        for iki, proofreading in ikis.items():
                            if iki not in ground_truth_ikis_on_keyboard:
                                ground_truth_ikis_on_keyboard[iki] = []
                            ground_truth_ikis_on_keyboard[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with ground truth gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_predict_gaze_df,
                                                                 typing_df=current_typing_log_df, position='text_entry')
                        for iki, proofreading in ikis.items():
                            if iki not in prediction_ikis_on_text_entry:
                                prediction_ikis_on_text_entry[iki] = []
                            prediction_ikis_on_text_entry[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with predicted gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_target_gaze_df,
                                                                 typing_df=current_typing_log_df, position='text_entry')
                        for iki, proofreading in ikis.items():
                            if iki not in ground_truth_ikis_on_text_entry:
                                ground_truth_ikis_on_text_entry[iki] = []
                            ground_truth_ikis_on_text_entry[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with ground truth gaze')

                    iki_mean_list.append(iki_mean)

                    # computing distance vs proofreading
                    try:
                        distances = get_distance_vs_proofreading(current_predict_gaze_df, current_typing_log_df,
                                                                 position='keyboard')
                        for distance, proofreading in distances.items():
                            if distance not in prediction_all_distances_on_keyboard:
                                prediction_all_distances_on_keyboard[distance] = []
                            prediction_all_distances_on_keyboard[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with predicted gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_target_gaze_df, current_typing_log_df,
                                                                 position='keyboard')
                        for distance, proofreading in distances.items():
                            if distance not in ground_truth_all_distances_on_keyboard:
                                ground_truth_all_distances_on_keyboard[distance] = []
                            ground_truth_all_distances_on_keyboard[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with ground truth gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_predict_gaze_df, current_typing_log_df,
                                                                 position='text_entry')
                        for distance, proofreading in distances.items():
                            if distance not in prediction_all_distances_on_text_entry:
                                prediction_all_distances_on_text_entry[distance] = []
                            prediction_all_distances_on_text_entry[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with predicted gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_target_gaze_df, current_typing_log_df,
                                                                 position='text_entry')
                        for distance, proofreading in distances.items():
                            if distance not in ground_truth_all_distances_on_text_entry:
                                ground_truth_all_distances_on_text_entry[distance] = []
                            ground_truth_all_distances_on_text_entry[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with ground truth gaze')

            else:
                for inputs, targets, masks_x, masks_y, index, user_params in tqdm(loader):
                    if user_index and index[0].split('_')[0] != user_index:
                        continue
                    inputs, targets, masks_x, masks_y, user_params = inputs.to(device), targets.to(device), masks_x.to(
                        device), masks_y.to(device), user_params.to(device)
                    current_typing_data = typing_data[typing_data['index'] == index[0]]
                    max_trailtime = current_typing_data['trailtime'].max()
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

                    distances, similarities, gaze_on_keyboard_ratio = distance_and_cosine_similarity_analysis(index[0],
                                                                                                              typing_log,
                                                                                                              valid_targets,
                                                                                                              model_type=model_type,
                                                                                                              data_use=data_use,
                                                                                                              amortized_inference=True,
                                                                                                              gaze_data_source='ground_truth')
                    for offset, dists in distances.items():
                        if offset not in all_distances_ground_truth:
                            all_distances_ground_truth[offset] = []
                        all_distances_ground_truth[offset].extend(dists)

                    if analysis_data_choice != 'both':
                        gaze = valid_gaze_outputs
                    else:
                        gaze = valid_targets
                    distances, similarities, gaze_on_keyboard_ratio = distance_and_cosine_similarity_analysis(index[0],
                                                                                                              typing_log,
                                                                                                              gaze,
                                                                                                              model_type=model_type,
                                                                                                              data_use=data_use,
                                                                                                              amortized_inference=True,
                                                                                                              gaze_data_source='predicted')
                    for offset, dists in distances.items():
                        if offset not in all_distances_predict:
                            all_distances_predict[offset] = []
                        all_distances_predict[offset].extend(dists)

                    for offset, sims in similarities.items():
                        if offset not in all_similarities:
                            all_similarities[offset] = []
                        all_similarities[offset].extend(sims)

                    # add the predicted gaze to the dataframe
                    # add the predicted gaze to the dataframe
                    temp_list = []
                    target_gaze_list = []
                    typing_list = []
                    for i, (x, y, duration) in enumerate(valid_gaze_outputs):
                        temp_list.append([index[0], x, y, duration])
                    for i, (x, y, duration) in enumerate(valid_targets):
                        target_gaze_list.append([index[0], x, y, duration])
                    for i, (x, y, duration, key, trailtime) in enumerate(
                            typing_log[['x', 'y', 'duration', 'key', 'trailtime']].values):
                        typing_list.append([index[0], x, y, duration, key, trailtime])
                    current_predict_gaze_df = pd.DataFrame(temp_list, columns=['index', 'x', 'y', 'duration'])
                    predict_gaze_df = pd.concat([predict_gaze_df,
                                                 current_predict_gaze_df],
                                                ignore_index=True)
                    current_target_gaze_df = pd.DataFrame(target_gaze_list,
                                                          columns=['index', 'x', 'y', 'duration'])
                    target_gaze_df = pd.concat([target_gaze_df,
                                                current_target_gaze_df],
                                               ignore_index=True)
                    current_typing_log_df = pd.DataFrame(typing_list,
                                                         columns=['index', 'x', 'y', 'duration', 'key', 'trailtime'])
                    typing_log_df = pd.concat([typing_log_df,
                                               current_typing_log_df],
                                              ignore_index=True)
                    # computing key vs proofreading
                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_predict_gaze_df, typing_df=current_typing_log_df,
                                                       position='keyboard')
                        for key, proofreading in keys.items():
                            if key not in prediction_all_keys_on_keyboard:
                                prediction_all_keys_on_keyboard[key] = []
                            prediction_all_keys_on_keyboard[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with predicted gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_target_gaze_df, typing_df=current_typing_log_df,
                                                       position='keyboard')
                        for key, proofreading in keys.items():
                            if key not in ground_truth_all_keys_on_keyboard:
                                ground_truth_all_keys_on_keyboard[key] = []
                            ground_truth_all_keys_on_keyboard[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with ground truth gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_predict_gaze_df, typing_df=current_typing_log_df,
                                                       position='text_entry')
                        for key, proofreading in keys.items():
                            if key not in prediction_all_keys_on_text_entry:
                                prediction_all_keys_on_text_entry[key] = []
                            prediction_all_keys_on_text_entry[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with predicted gaze')

                    try:
                        keys = get_key_vs_proofreading(gaze_df=current_target_gaze_df, typing_df=current_typing_log_df,
                                                       position='text_entry')
                        for key, proofreading in keys.items():
                            if key not in ground_truth_all_keys_on_text_entry:
                                ground_truth_all_keys_on_text_entry[key] = []
                            ground_truth_all_keys_on_text_entry[key].extend(proofreading)
                    except:
                        print('error in computing key vs proofreading with ground truth gaze')

                    # computing iki vs proofreading
                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_predict_gaze_df,
                                                                 typing_df=current_typing_log_df, position='keyboard')
                        for iki, proofreading in ikis.items():
                            if iki not in prediction_ikis_on_keyboard:
                                prediction_ikis_on_keyboard[iki] = []
                            prediction_ikis_on_keyboard[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with predicted gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_target_gaze_df,
                                                                 typing_df=current_typing_log_df, position='keyboard')
                        for iki, proofreading in ikis.items():
                            if iki not in ground_truth_ikis_on_keyboard:
                                ground_truth_ikis_on_keyboard[iki] = []
                            ground_truth_ikis_on_keyboard[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with ground truth gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_predict_gaze_df,
                                                                 typing_df=current_typing_log_df, position='text_entry')
                        for iki, proofreading in ikis.items():
                            if iki not in prediction_ikis_on_text_entry:
                                prediction_ikis_on_text_entry[iki] = []
                            prediction_ikis_on_text_entry[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with predicted gaze')

                    try:
                        ikis, iki_mean = get_iki_vs_proofreading(gaze_df=current_target_gaze_df,
                                                                 typing_df=current_typing_log_df, position='text_entry')
                        for iki, proofreading in ikis.items():
                            if iki not in ground_truth_ikis_on_text_entry:
                                ground_truth_ikis_on_text_entry[iki] = []
                            ground_truth_ikis_on_text_entry[iki].extend(proofreading)
                    except:
                        print('error in computing iki vs proofreading with ground truth gaze')

                    iki_mean_list.append(iki_mean)

                    # computing distance vs proofreading
                    try:
                        distances = get_distance_vs_proofreading(current_predict_gaze_df, current_typing_log_df,
                                                                 position='keyboard')
                        for distance, proofreading in distances.items():
                            if distance not in prediction_all_distances_on_keyboard:
                                prediction_all_distances_on_keyboard[distance] = []
                            prediction_all_distances_on_keyboard[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with predicted gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_target_gaze_df, current_typing_log_df,
                                                                 position='keyboard')
                        for distance, proofreading in distances.items():
                            if distance not in ground_truth_all_distances_on_keyboard:
                                ground_truth_all_distances_on_keyboard[distance] = []
                            ground_truth_all_distances_on_keyboard[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with ground truth gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_predict_gaze_df, current_typing_log_df,
                                                                 position='text_entry')
                        for distance, proofreading in distances.items():
                            if distance not in prediction_all_distances_on_text_entry:
                                prediction_all_distances_on_text_entry[distance] = []
                            prediction_all_distances_on_text_entry[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with predicted gaze')

                    try:
                        distances = get_distance_vs_proofreading(current_target_gaze_df, current_typing_log_df,
                                                                 position='text_entry')
                        for distance, proofreading in distances.items():
                            if distance not in ground_truth_all_distances_on_text_entry:
                                ground_truth_all_distances_on_text_entry[distance] = []
                            ground_truth_all_distances_on_text_entry[distance].extend(proofreading)
                    except:
                        print('error in computing distance vs proofreading with ground truth gaze')
    print("Processing Data Done, Generating Plots")

    metrics_predict = calculate_gaze_metrics(predict_gaze_df, log_index=[])
    print("Predicted Metrics:")
    print('fixations: {}({})'.format(metrics_predict['mean_fixations'], metrics_predict['std_fixations']))
    print('fixation duration: {}({})'.format(metrics_predict['mean_fixation_duration'],
                                             metrics_predict['std_fixation_duration']))
    print('gaze shifts: {}({})'.format(metrics_predict['mean_gaze_shifts'], metrics_predict['std_gaze_shifts']))
    print('time ratio on keyboard: {}({})'.format(metrics_predict['mean_time_ratio_on_keyboard'],
                                                  metrics_predict['std_time_ratio_on_keyboard']))

    metrics_ground_truth = calculate_gaze_metrics(target_gaze_df, log_index=[])
    print("Ground Truth Metrics:")
    print('fixations: {}({})'.format(metrics_ground_truth['mean_fixations'], metrics_ground_truth['std_fixations']))
    print('fixation duration: {}({})'.format(metrics_ground_truth['mean_fixation_duration'],
                                             metrics_ground_truth['std_fixation_duration']))
    print(
        'gaze shifts: {}({})'.format(metrics_ground_truth['mean_gaze_shifts'], metrics_ground_truth['std_gaze_shifts']))
    print('time ratio on keyboard: {}({})'.format(metrics_ground_truth['mean_time_ratio_on_keyboard'],
                                                  metrics_ground_truth['std_time_ratio_on_keyboard']))

    final_avg_distances_predict = {offset: np.nanmean(all_distances_predict[offset]) for offset in
                                   all_distances_predict}
    final_avg_distances_ground_truth = {offset: np.nanmean(all_distances_ground_truth[offset]) for offset in
                                        all_distances_ground_truth}

    plot_key_counts(ground_truth_all_keys_on_keyboard, save_dir=img_output_dir, position='keyboard', )
    plot_key_counts(ground_truth_all_keys_on_text_entry, save_dir=img_output_dir, position='text_entry')

    for key in list(prediction_all_keys_on_keyboard.keys()):
        if key not in ground_truth_all_keys_on_keyboard:
            prediction_all_keys_on_keyboard.pop(key)
    for key in list(ground_truth_all_keys_on_keyboard.keys()):
        if key not in prediction_all_keys_on_keyboard:
            ground_truth_all_keys_on_keyboard.pop(key)
    # if the key count < 5, remove the key
    for key in list(prediction_all_keys_on_keyboard.keys()):
        if len(prediction_all_keys_on_keyboard[key]) < 5:
            prediction_all_keys_on_keyboard.pop(key)
    for key in list(ground_truth_all_keys_on_keyboard.keys()):
        if len(ground_truth_all_keys_on_keyboard[key]) < 5:
            ground_truth_all_keys_on_keyboard.pop(key)

    for key in list(prediction_all_keys_on_text_entry.keys()):
        if key not in ground_truth_all_keys_on_text_entry:
            prediction_all_keys_on_text_entry.pop(key)
    for key in list(ground_truth_all_keys_on_text_entry.keys()):
        if key not in prediction_all_keys_on_text_entry:
            ground_truth_all_keys_on_text_entry.pop(key)
    # if the key count < 5, remove the key
    for key in list(prediction_all_keys_on_text_entry.keys()):
        if len(prediction_all_keys_on_text_entry[key]) < 5:
            prediction_all_keys_on_text_entry.pop(key)
    for key in list(ground_truth_all_keys_on_text_entry.keys()):
        if len(ground_truth_all_keys_on_text_entry[key]) < 5:
            ground_truth_all_keys_on_text_entry.pop(key)

    if analysis_data_choice != 'test':
        metrics = metrics_ground_truth
    else:
        metrics = metrics_predict
    plot_distances(final_avg_distances_predict, gaze_on_keyboard_ratio=metrics["mean_time_ratio_on_keyboard"],
                   gaze_data_source='predicted', save_dir=img_output_dir)
    plot_distances(final_avg_distances_ground_truth,
                   gaze_on_keyboard_ratio=metrics_ground_truth["mean_time_ratio_on_keyboard"],
                   gaze_data_source='ground_truth', save_dir=img_output_dir)

    generate_keyboard_heatmap_based_on_count(copy.deepcopy(ground_truth_all_keys_on_keyboard), save_dir=img_output_dir,
                                             position='keyboard')
    generate_keyboard_heatmap_based_on_count(copy.deepcopy(ground_truth_all_keys_on_text_entry),
                                             save_dir=img_output_dir, position='text_entry')
    if analysis_data_choice == 'both':
        process_key_vs_position(copy.deepcopy(ground_truth_all_keys_on_keyboard), gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='keyboard')
        process_key_vs_position(copy.deepcopy(ground_truth_all_keys_on_text_entry), gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='text_entry')

        generate_proofreading_heatmap(copy.deepcopy(ground_truth_all_keys_on_keyboard), save_dir=img_output_dir,
                                      gaze_data_source='ground_truth', position='keyboard')
        generate_proofreading_heatmap(copy.deepcopy(ground_truth_all_keys_on_text_entry), save_dir=img_output_dir,
                                      gaze_data_source='ground_truth', position='text_entry')

        process_iki_vs_position(ground_truth_ikis_on_keyboard, iki_mean_list, gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='keyboard')
        process_iki_vs_position(ground_truth_ikis_on_text_entry, iki_mean_list, gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='text_entry')

        process_distance_vs_position(ground_truth_all_distances_on_keyboard, gaze_data_source='ground_truth',
                                     save_dir=img_output_dir, position='keyboard')
        process_distance_vs_position(ground_truth_all_distances_on_text_entry, gaze_data_source='ground_truth',
                                     save_dir=img_output_dir, position='text_entry')

        plot_iki_vs_distance(ikis_distance, save_dir=img_output_dir)

    else:
        process_key_vs_position(copy.deepcopy(ground_truth_all_keys_on_keyboard), gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='keyboard')
        process_key_vs_position(copy.deepcopy(ground_truth_all_keys_on_text_entry), gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='text_entry')

        generate_proofreading_heatmap(copy.deepcopy(ground_truth_all_keys_on_keyboard), save_dir=img_output_dir,
                                      gaze_data_source='ground_truth', position='keyboard')
        generate_proofreading_heatmap(copy.deepcopy(ground_truth_all_keys_on_text_entry), save_dir=img_output_dir,
                                      gaze_data_source='ground_truth', position='text_entry')

        process_iki_vs_position(ground_truth_ikis_on_keyboard, iki_mean_list, gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='keyboard')
        process_iki_vs_position(ground_truth_ikis_on_text_entry, iki_mean_list, gaze_data_source='ground_truth',
                                save_dir=img_output_dir, position='text_entry')

        process_distance_vs_position(ground_truth_all_distances_on_keyboard, gaze_data_source='ground_truth',
                                     save_dir=img_output_dir, position='keyboard')
        process_distance_vs_position(ground_truth_all_distances_on_text_entry, gaze_data_source='ground_truth',
                                     save_dir=img_output_dir, position='text_entry')

        process_key_vs_position(copy.deepcopy(prediction_all_keys_on_keyboard), gaze_data_source='predicted',
                                save_dir=img_output_dir, position='keyboard')
        process_key_vs_position(copy.deepcopy(prediction_all_keys_on_text_entry), gaze_data_source='predicted',
                                save_dir=img_output_dir, position='text_entry')

        generate_proofreading_heatmap(copy.deepcopy(prediction_all_keys_on_keyboard), save_dir=img_output_dir,
                                      gaze_data_source='predicted', position='keyboard')
        generate_proofreading_heatmap(copy.deepcopy(prediction_all_keys_on_text_entry), save_dir=img_output_dir,
                                      gaze_data_source='predicted', position='text_entry')

        process_iki_vs_position(prediction_ikis_on_keyboard, iki_mean_list, gaze_data_source='predicted',
                                save_dir=img_output_dir, position='keyboard')
        process_iki_vs_position(prediction_ikis_on_text_entry, iki_mean_list, gaze_data_source='predicted',
                                save_dir=img_output_dir, position='text_entry')

        process_distance_vs_position(prediction_all_distances_on_keyboard, gaze_data_source='predicted',
                                     save_dir=img_output_dir, position='keyboard')
        process_distance_vs_position(prediction_all_distances_on_text_entry, gaze_data_source='predicted',
                                     save_dir=img_output_dir, position='text_entry')


def get_iki_vs_distance(typing_df):
    # record iki and distance, iki is the time between two key press, distance is the distance between two key press
    ikis = {}
    for i in range(1, len(typing_df)):
        iki = typing_df.iloc[i]['trailtime'] - typing_df.iloc[i - 1]['trailtime']
        distance = np.sqrt(
            (typing_df.iloc[i]['x'] - typing_df.iloc[i - 1]['x']) ** 2 + (
                    typing_df.iloc[i]['y'] - typing_df.iloc[i - 1]['y']) ** 2)
        if iki not in ikis:
            ikis[iki] = []
        ikis[iki].append(distance)

    return ikis


def plot_iki_vs_distance(ikis_distance, save_dir=None):
    # Set bin size and maximum IKI for plotting
    bin_size = 50  # Adjust bin size as necessary
    min_iki = 50  # Set min IKI to 50
    max_iki = 950  # Max iki for distance plot

    # Bins for iki (< 1000)
    bins_iki = np.arange(min_iki, max_iki + bin_size, bin_size)

    # Initialize binned dictionary
    binned_ikis = {bin: [] for bin in bins_iki}

    # Bin the data
    for iki, distances in ikis_distance.items():
        bin_iki = (iki // bin_size) * bin_size
        if min_iki <= iki <= max_iki:
            binned_ikis[bin_iki].extend(distances)

    # Calculate the average distance and standard deviation for each bin (for bins < 1000)
    binned_iki_keys = sorted(binned_ikis.keys())
    distance_mean = [np.mean(binned_ikis[bin]) if binned_ikis[bin] else 0 for bin in binned_iki_keys]
    distance_std = [np.std(binned_ikis[bin]) if binned_ikis[bin] else 0 for bin in binned_iki_keys]

    # Ensure error bars are calculated correctly
    distance_mean = np.array(distance_mean)
    distance_std = np.array(distance_std)

    asymmetric_error = [distance_std, distance_std]  # Symmetric error bars

    # Plot distance vs iki (bins < 1000)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=binned_iki_keys, y=distance_mean, color='#4682B4')  # Set bar color

    # Add symmetric error bars
    ax.errorbar(x=np.arange(len(binned_iki_keys)), y=distance_mean, yerr=asymmetric_error, fmt='none', c='black',
                capsize=5)

    # Formatting
    ax.set_xlabel('IKI (ms)')
    ax.set_ylabel('Distance (pixel)')
    ax.set_title('Average Distance by IKI')
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05)
    ax.set_xticklabels(binned_iki_keys, rotation=45)

    # Save or show the plot
    if save_dir:
        plt.savefig(osp.join(save_dir, 'average_distance_by_iki.png'))
    plt.show()


def process_distance_vs_position(all_distances, gaze_data_source='ground_truth', save_dir=None, position='keyboard'):
    plt.rcParams.update({
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 14,  # X and Y label font size
        'xtick.labelsize': 14,  # X tick labels font size
        'ytick.labelsize': 14,  # Y tick labels font size
        'legend.fontsize': 16,  # Legend font size
        'text.color': 'black',  # Text color
    })
    # Group distances into bins
    bin_size = 50  # Adjust bin size as necessary
    max_distance_proofreading = 950  # Max distance for proofreading rate plot
    min_distance_proofreading = 50
    max_distance_count = 1200  # Max distance for count plot
    min_distance_count = 50
    max_count = 200

    # Bins for proofreading rate (< 1000) and full bins for distance count (< 1500)
    bins_proofreading = np.arange(min_distance_proofreading, max_distance_proofreading, bin_size)
    bins_count = np.arange(min_distance_count, max_distance_count + bin_size, bin_size)

    # Initialize binned dictionaries
    binned_distances_proofreading = {bin: [] for bin in bins_proofreading}
    binned_distances_count = {bin: [] for bin in bins_count}

    # Bin the data
    for distance, proofreading in all_distances.items():
        bin_distance = (distance // bin_size) * bin_size
        if min_distance_proofreading <= distance <= max_distance_proofreading:  # For proofreading rate plot (bins < 1000)
            binned_distances_proofreading[bin_distance].extend(proofreading)
        if min_distance_count <= distance <= max_distance_count:  # For full count plot (bins < 1500)
            binned_distances_count[bin_distance].extend(proofreading)

    # Calculate the average proofreading rate for each bin (for bins < 1000)
    binned_distance_keys_proofreading = sorted(binned_distances_proofreading.keys())
    proofreading_rate = [np.mean(binned_distances_proofreading[bin]) if binned_distances_proofreading[bin] else 0 for
                         bin in binned_distance_keys_proofreading]
    proofreading_std = [np.std(binned_distances_proofreading[bin]) if binned_distances_proofreading[bin] else 0 for bin
                        in binned_distance_keys_proofreading]

    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = np.clip(proofreading_rate + proofreading_std, -np.inf, 1)
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]

    # Plot proofreading rate (bins < 1000)
    plt.figure(figsize=(12, 8))
    plt.ylim(0, 1)
    ax = sns.barplot(x=binned_distance_keys_proofreading, y=proofreading_rate)
    ax.errorbar(x=np.arange(len(binned_distance_keys_proofreading)), y=proofreading_rate, yerr=asymmetric_error,
                fmt='none', c='black', capsize=5)
    ax.set_xlabel('Distance(pixel)')
    if position == 'keyboard':
        ax.set_ylabel('Gaze on Keyboard Ratio')
        ax.set_title('Average Gaze on Keyboard Ratio by Distance')
    else:
        ax.set_ylabel('Gaze on Text Entry Ratio')
        ax.set_title('Average Gaze on Text Entry Ratio by Distance')
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.075)
    ax.set_xticklabels(binned_distance_keys_proofreading, rotation=45)
    if position == 'keyboard':
        if save_dir:
            plt.savefig(osp.join(save_dir, 'average_gaze_on_keyboard_ratio_by_distance_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'average_gaze_on_keyboard_ratio_by_distance_' + gaze_data_source + '.png'))
    else:
        if save_dir:
            plt.savefig(osp.join(save_dir, 'average_gaze_on_text_entry_ratio_by_distance_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'average_gaze_on_text_entry_ratio_by_distance_' + gaze_data_source + '.png'))
    plt.show()

    # Calculate the distance count (for full range up to 1500)
    binned_distance_keys_count = sorted(binned_distances_count.keys())
    distance_count = [len(binned_distances_count[bin]) for bin in binned_distance_keys_count]

    # Plot distance count bar chart (bins < 1500)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=binned_distance_keys_count, y=distance_count)
    plt.xlabel('Distance(pixel)')
    plt.ylabel('Count')
    plt.ylim(0, max_count)
    plt.title('Distance Count')
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.1)
    plt.xticks(rotation=45)

    if save_dir:
        plt.savefig(osp.join(save_dir, 'distance_count_by_bins_' + gaze_data_source + '.png'))
    else:
        plt.savefig(osp.join(FIG_DIR, 'distance_count_by_bins_' + gaze_data_source + '.png'))
    plt.show()


def generate_proofreading_heatmap(all_keys, keyboard_image_path=keyboard_image_path,
                                  how_we_type_key_coordinate=how_we_type_key_coordinate_resized,
                                  save_dir=None, gaze_data_source='ground_truth', position='keyboard'):
    # Sort the keys based on their row order for visualization
    first_row_char = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']
    second_row_char = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']
    third_row_char = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    backspace_char = ['<']  # Backspace should have its own color
    fourth_row_char = [' ']  # Space is on the fourth row

    # Change the key '<' to B
    if '<' in all_keys:
        all_keys['<'] = all_keys.pop('B')

    # Ensure that all keys exist in the input data, even if they have 0 proofreading events
    for key in first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char:
        if key not in all_keys:
            all_keys[key] = []

    # Initialize pygame
    pygame.init()

    # Combine the rows and filter out keys not in all_keys
    ordered_keys = first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char

    # Compute proofreading rates for each key
    proofreading_rate = {key: np.mean(all_keys[key]) if len(all_keys[key]) > 0 else 0 for key in ordered_keys}

    # Load the keyboard image
    keyboard_image = pygame.image.load(keyboard_image_path)

    # Get the size of the keyboard image
    keyboard_width, keyboard_height = keyboard_image.get_size()

    # Set the size of the window to match the keyboard image (optional if we want display)
    screen = pygame.Surface((keyboard_width, keyboard_height))  # Using surface instead of display

    # Normalize proofreading rates to range [0, 1] for the heatmap
    max_rate = max(proofreading_rate.values()) if proofreading_rate else 1  # Prevent division by zero
    max_rate = max(max_rate, 0.000001)
    normalized_rates = {key: rate / max_rate for key, rate in proofreading_rate.items()}

    # Create a surface for the heatmap overlay for the entire keyboard area
    heatmap_surface = pygame.Surface((keyboard_width, keyboard_height), pygame.SRCALPHA)

    text_surface = pygame.Surface((keyboard_width, keyboard_height), pygame.SRCALPHA)

    # Set heatmap color (slightly lighter green this time for proofreading rates)
    base_color = (0, 255, 0)  # Green for proofreading rate
    max_alpha = 150  # Lighter max alpha to keep keyboard visible

    # Font for displaying rates (bold, black text)
    font = pygame.font.SysFont(None, 48, bold=False)

    # Draw rectangles on the heatmap surface based on proofreading rate data
    for key, coords in how_we_type_key_coordinate.items():
        if key in normalized_rates:  # Only draw if the key exists in normalized rates
            # Calculate the alpha based on normalized proofreading rate (frequency of proofreading)
            alpha = int(normalized_rates[key] * max_alpha)
            # Apply the color with transparency
            color_with_alpha = (*base_color, alpha)

            # Use correct interpretation of [x1, y1, x2, y2] for pygame.Rect
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1

            # Draw the rectangle over the key area
            pygame.draw.rect(heatmap_surface, color_with_alpha, pygame.Rect(x1, y1, width, height))

    # Render the proofreading rates on the text surface
    i = 0
    for key, coords in how_we_type_key_coordinate.items():
        if key in normalized_rates:  # Only draw if the key exists in normalized rates
            # Render proofreading rate in the center of each key
            rate_text = font.render(f'{proofreading_rate[key]:.2f}', True, (0, 0, 0))  # Black bold text
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1
            text_rect = rate_text.get_rect(center=(x1 + width // 2, y1 + height * 0.9))

            # Blit the text on the text surface
            text_surface.blit(rate_text, text_rect)
            i += 1

    # Blit the keyboard image
    screen.blit(keyboard_image, (0, 0))
    # Blit the heatmap surface on top with transparency
    screen.blit(heatmap_surface, (0, 0))
    screen.blit(text_surface, (0, 0))

    # Save the rendered surface as an image if save_dir is provided
    if save_dir:
        if position == 'keyboard':
            output_path = os.path.join(save_dir, 'gaze_on_keyboard_ratio_heatmap_' + gaze_data_source + '.png')
            print(f"gaze_on_keyboard_ratio heatmap saved to {output_path}")
        else:
            output_path = os.path.join(save_dir, 'gaze_on_text_entry_ratio_heatmap_' + gaze_data_source + '.png')
            print(f"gaze_on_text_entry_ratio heatmap saved to {output_path}")

        pygame.image.save(screen, output_path)

    # Quit pygame
    pygame.quit()


def generate_keyboard_heatmap_based_on_count(all_keys, keyboard_image_path=keyboard_image_path,
                                             how_we_type_key_coordinate=how_we_type_key_coordinate_resized,
                                             save_dir=None, gaze_data_source='ground_truth', position='keyboard'):
    # Sort the keys based on their row order for visualization
    first_row_char = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']
    second_row_char = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']
    third_row_char = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    backspace_char = ['<']  # Backspace should have its own color
    fourth_row_char = [' ']  # Space is on the fourth row

    # Change the key '<' to B
    # if < exists:
    if '<' in all_keys:
        all_keys['<'] = all_keys.pop('B')

    # Ensure that all keys exist in the input data, even if they have 0 presses
    for key in first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char:
        if key not in all_keys:
            all_keys[key] = []

    # Initialize pygame
    pygame.init()

    # Combine the rows and filter out keys not in all_keys
    ordered_keys = first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char
    key_counts = [len(all_keys[key]) for key in ordered_keys]  # Count of key presses

    # Load the keyboard image
    keyboard_image = pygame.image.load(keyboard_image_path)

    # Get the size of the keyboard image
    keyboard_width, keyboard_height = keyboard_image.get_size()

    # Set the size of the window to match the keyboard image (optional if we want display)
    screen = pygame.Surface((keyboard_width, keyboard_height))  # Using surface instead of display

    # Normalize key counts to range [0, 1] for the heatmap
    max_count = max(key_counts) if key_counts else 1  # Prevent division by zero
    normalized_counts = {key: count / max_count for key, count in zip(ordered_keys, key_counts)}

    # Create a surface for the heatmap overlay for the entire keyboard area
    heatmap_surface = pygame.Surface((keyboard_width, keyboard_height), pygame.SRCALPHA)
    text_surface = pygame.Surface((keyboard_width, keyboard_height), pygame.SRCALPHA)

    # Set heatmap color (slightly lighter blue)
    base_color = (0, 0, 255)  # Blue
    max_alpha = 150  # Lighter max alpha to keep keyboard visible

    # Font for displaying counts (bold, black text)
    font = pygame.font.SysFont(None, 48, bold=False)

    def calculate_alpha(count, normalized_count, min_threshold=0.25, low_count=5):
        """
        Calculate alpha for the key. The alpha increases with count, but for very low counts,
        it starts at a slightly higher base.
        """
        if count == 0:
            return 0  # Completely transparent for zero count
        elif count < low_count:
            # Increase alpha slightly faster for low counts
            return int(min_threshold * max_alpha + (max_alpha * 0.5 * normalized_count))
        else:
            # Use regular alpha for higher counts
            return int(min_threshold * max_alpha + (max_alpha * normalized_count))

    # Draw rectangles on the heatmap surface based on key press data
    i = 0
    for key, coords in how_we_type_key_coordinate.items():
        if key in normalized_counts:  # Only draw if the key exists in normalized counts
            count = key_counts[i]
            normalized_count = normalized_counts[key]

            # Calculate the alpha based on adjusted count scaling
            alpha = calculate_alpha(count, normalized_count)

            # Apply the color with transparency
            color_with_alpha = (*base_color, alpha)

            # Use correct interpretation of [x1, y1, x2, y2] for pygame.Rect
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1

            # Draw the rectangle over the key area
            pygame.draw.rect(heatmap_surface, color_with_alpha, pygame.Rect(x1, y1, width, height))
        i += 1

    i = 0
    for key, coords in how_we_type_key_coordinate.items():
        if key in normalized_counts:  # Only draw if the key exists in normalized counts
            # Render key count in the center of each key
            count_text = font.render(str(key_counts[i]), True, (0, 0, 0))  # Black bold text
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1
            text_rect = count_text.get_rect(center=(x1 + width // 2, y1 + height * 0.9))

            # Blit the text on the text surface
            text_surface.blit(count_text, text_rect)
            i += 1

    # Blit the keyboard image
    screen.blit(keyboard_image, (0, 0))
    # Blit the heatmap surface on top with transparency
    screen.blit(heatmap_surface, (0, 0))
    screen.blit(text_surface, (0, 0))
    # Save the rendered surface as an image if save_dir is provided
    if save_dir:
        output_path = os.path.join(save_dir,
                                   'key_count_for_each_key_heatmap' + gaze_data_source + '.png')
        pygame.image.save(screen, output_path)
        print(f"Keyboard heatmap saved to {output_path}")

    # Quit pygame
    pygame.quit()


def plot_key_counts(all_keys, gaze_data_source='ground_truth', save_dir=None, position='keyboard'):
    # Sort the keys based on their row order for visualization
    first_row_char = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']
    second_row_char = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']
    third_row_char = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    backspace_char = ['B']  # Backspace should have its own color
    fourth_row_char = [' ']  # Space is on the fourth row

    for key in first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char:
        if key not in all_keys:
            all_keys[key] = []

    # Combine the rows and filter out keys not in all_keys
    ordered_keys = [key for key in
                    (first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char)]

    # Count the number of key presses for each key
    key_counts = [len(all_keys[key]) for key in ordered_keys]  # Count of key presses

    # Change " " to "space" and "B" to "backspace" for better visualization
    ordered_keys = [key.replace(" ", "space").replace("B", "backspace") for key in ordered_keys]

    # Define colors for each row and special keys (5 colors)
    row_colors = ['#ADD8E6',  # Light blue for 1st row
                  '#87CEEB',  # Slightly darker blue for 2nd row
                  '#4682B4',  # Medium blue for 3rd row
                  '#1E90FF',  # Darker blue for backspace
                  '#00008B']  # Darkest blue for space
    key_colors = []
    for key in ordered_keys:
        if key in ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']:
            key_colors.append(row_colors[0])  # First row
        elif key in ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']:
            key_colors.append(row_colors[1])  # Second row
        elif key in ['z', 'x', 'c', 'v', 'b', 'n', 'm']:
            key_colors.append(row_colors[2])  # Third row
        elif key == 'backspace':
            key_colors.append(row_colors[3])  # Backspace with its own color
        elif key == 'space':
            key_colors.append(row_colors[4])  # Space with its own color

    # Plot key counts
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=ordered_keys, y=key_counts, palette=key_colors)
    plt.xlabel('Key')
    plt.ylabel('Key Count')
    if position == 'keyboard':
        plt.title('Key Count for Each Key on Keyboard')
    else:
        plt.title('Key Count for Each Key on Text Entry')
    plt.xticks(rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.175, right=0.975, left=0.075)

    # Save the figure
    if save_dir:
        plt.savefig(osp.join(save_dir, 'key_count_for_each_key' + gaze_data_source + '.png'))
    else:
        plt.savefig(osp.join(FIG_DIR, 'key_count_for_each_key' + gaze_data_source + '.png'))
    plt.show()


def get_distance_vs_proofreading(gaze_df, typing_df, position='keyboard'):
    distances = {}

    if position == 'keyboard':
        y_limit = 1230
    else:
        y_limit = 225

    for i in range(1, len(typing_df)):
        typing_row_prev = typing_df.iloc[i - 1]
        typing_row_curr = typing_df.iloc[i]

        # Calculate the distance between the previous and current row
        distance = np.linalg.norm(
            [typing_row_curr['x'] - typing_row_prev['x'], typing_row_curr['y'] - typing_row_prev['y']])

        trialtime_prev = typing_row_prev['trailtime']
        trialtime_curr = typing_row_curr['trailtime']

        # Get gaze data within this interval
        window_gaze_df = gaze_df[(gaze_df['trailtime'] >= trialtime_prev) & (gaze_df['trailtime'] <= trialtime_curr)]
        if not window_gaze_df.empty:
            if window_gaze_df.iloc[-1]['temp_index'] != len(gaze_df) - 1:
                window_gaze_df = pd.concat([window_gaze_df,
                                            gaze_df[
                                                gaze_df['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] + 1]])
        else:
            window_gaze_df = gaze_df[(gaze_df['trailtime'] < trialtime_prev)]
            if window_gaze_df.empty:
                continue
            else:
                # get the last gaze data
                window_gaze_df = window_gaze_df.iloc[-1:]
                # get the gaze row
                gaze_row = window_gaze_df.iloc[0]
                if position == 'keyboard':
                    if gaze_row['y'] >= y_limit:
                        proofreading_ratio = 1
                    else:
                        proofreading_ratio = 0
                else:
                    if gaze_row['y'] <= y_limit:
                        proofreading_ratio = 1
                    else:
                        proofreading_ratio = 0

                if distance not in distances:
                    distances[distance] = []
                distances[distance].append(proofreading_ratio)
                continue

        if window_gaze_df.empty:
            proofreading_ratio = 0
        else:
            proofreading_time = 0
            keyboard_time = 0

            for _, gaze_row in window_gaze_df.iterrows():
                if position == 'keyboard':
                    if gaze_row['y'] >= y_limit:
                        # Proofreading
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            proofreading_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            proofreading_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            proofreading_time += gaze_row['duration']
                    else:
                        # Keyboard
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            keyboard_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            keyboard_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            keyboard_time += gaze_row['duration']
                else:
                    if gaze_row['y'] <= y_limit:
                        # Proofreading
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            proofreading_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            proofreading_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            proofreading_time += gaze_row['duration']
                    else:
                        # Keyboard
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            keyboard_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            keyboard_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            keyboard_time += gaze_row['duration']

            # Calculate proofreading ratio
            if proofreading_time + keyboard_time > 1e-6:
                proofreading_ratio = proofreading_time / (proofreading_time + keyboard_time)
            else:
                proofreading_ratio = 0.0

        # Store the proofreading ratio in distances dictionary
        if distance not in distances:
            distances[distance] = []
        distances[distance].append(proofreading_ratio)

    return distances


def get_iki_vs_proofreading(gaze_df, typing_df, position='keyboard'):
    iki = {}
    iki_mean = typing_df['duration'].iloc[1:].mean()
    for i in range(1, len(typing_df)):
        typing_row_prev = typing_df.iloc[i - 1]
        typing_row_curr = typing_df.iloc[i]

        trialtime_prev = typing_row_prev['trailtime']
        trialtime_curr = typing_row_curr['trailtime']

        # Calculate IKI
        inter_key_interval = trialtime_curr - trialtime_prev

        if inter_key_interval >= 3000:
            continue

        if position == 'keyboard':
            y_limit = 1230
        else:
            y_limit = 225

        window_gaze_df = gaze_df[(gaze_df['trailtime'] >= trialtime_prev) & (gaze_df['trailtime'] <= trialtime_curr)]
        if not window_gaze_df.empty:
            if window_gaze_df.iloc[-1]['temp_index'] != len(gaze_df) - 1:
                window_gaze_df = pd.concat([window_gaze_df,
                                            gaze_df[
                                                gaze_df['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] + 1]])
        else:
            window_gaze_df = gaze_df[(gaze_df['trailtime'] < trialtime_prev)]
            if window_gaze_df.empty:
                continue
            else:
                # get the last gaze data
                window_gaze_df = window_gaze_df.iloc[-1:]
                # get the gaze row
                gaze_row = window_gaze_df.iloc[0]
                if position == 'keyboard':
                    if gaze_row['y'] >= y_limit:
                        proofreading_ratio = 1
                    else:
                        proofreading_ratio = 0
                else:
                    if gaze_row['y'] <= y_limit:
                        proofreading_ratio = 1
                    else:
                        proofreading_ratio = 0

                if inter_key_interval not in iki:
                    iki[inter_key_interval] = []
                iki[inter_key_interval].append(proofreading_ratio)
                continue

        if window_gaze_df.empty:
            proofreading_ratio = 0
        else:
            proofreading_time = 0
            keyboard_time = 0

            for _, gaze_row in window_gaze_df.iterrows():
                if position == 'keyboard':
                    if gaze_row['y'] >= y_limit:
                        # Proofreading
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            proofreading_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            proofreading_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            proofreading_time += gaze_row['duration']
                    else:
                        # Keyboard
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            keyboard_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            keyboard_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            keyboard_time += gaze_row['duration']
                else:
                    if gaze_row['y'] <= y_limit:
                        # Proofreading
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            proofreading_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            proofreading_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            proofreading_time += gaze_row['duration']
                    else:
                        # Keyboard
                        if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                            keyboard_time += gaze_row['trailtime'] - trialtime_prev
                        elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                            'trailtime'] > trialtime_curr:
                            keyboard_time += trialtime_curr - window_gaze_df.iloc[-2]['trailtime']
                        else:
                            keyboard_time += gaze_row['duration']

            # Calculate proofreading ratio
            if proofreading_time + keyboard_time > 1e-6:
                proofreading_ratio = proofreading_time / (proofreading_time + keyboard_time)
            else:
                proofreading_ratio = 0.0

        # Store the proofreading ratio in iki
        if inter_key_interval not in iki:
            iki[inter_key_interval] = []
        iki[inter_key_interval].append(proofreading_ratio)

    return iki, iki_mean


def process_iki_vs_position(all_ikis, iki_mean_list, gaze_data_source='ground_truth', save_dir=None,
                            position='keyboard'):
    # Set global font sizes
    plt.rcParams.update({
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 16,  # X and Y label font size
        'xtick.labelsize': 14,  # X tick labels font size
        'ytick.labelsize': 14,  # Y tick labels font size
        'legend.fontsize': 16,  # Legend font size
        'text.color': 'black',  # Text color
    })
    # Group IKIs into bins of 50
    bin_size = 50
    max_iki_proofreading = 900  # Set max IKI to 1000 for proofreading rate plot
    min_iki_proofreading = 100
    max_iki_count = 2000  # Set max IKI to 2000 for count plot
    min_iki_count = 50

    # Bins for proofreading rate (< 1000) and full bins for IKI count
    bins_proofreading = np.arange(min_iki_proofreading, max_iki_proofreading, bin_size)
    bins_count = np.arange(min_iki_count, max_iki_count + bin_size, bin_size)

    # Initialize binned dictionaries
    binned_ikis_proofreading = {bin: [] for bin in bins_proofreading}
    binned_ikis_count = {bin: [] for bin in bins_count}

    # Bin the data
    for iki, proofreading in all_ikis.items():
        bin_proofreading = (iki // bin_size) * bin_size
        if min_iki_proofreading <= iki <= max_iki_proofreading and iki:  # For proofreading rate plot (bins < 1000)
            binned_ikis_proofreading[bin_proofreading].extend(proofreading)
        if min_iki_count <= iki <= max_iki_count:  # For full count plot (bins < 2000)
            binned_ikis_count[bin_proofreading].extend(proofreading)

    # Calculate the average proofreading rate for each bin (for bins < 1000)
    binned_iki_keys_proofreading = sorted(binned_ikis_proofreading.keys())
    proofreading_rate = [np.mean(binned_ikis_proofreading[bin]) if binned_ikis_proofreading[bin] else 0 for bin in
                         binned_iki_keys_proofreading]
    proofreading_std = [np.std(binned_ikis_proofreading[bin]) if binned_ikis_proofreading[bin] else 0 for bin in
                        binned_iki_keys_proofreading]

    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = np.clip(proofreading_rate + proofreading_std, -np.inf, 1)
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]

    # Plot proofreading rate (bins < 1000)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=binned_iki_keys_proofreading, y=proofreading_rate)
    ax.errorbar(x=np.arange(len(binned_iki_keys_proofreading)), y=proofreading_rate, yerr=asymmetric_error, fmt='none',
                c='black', capsize=5)
    ax.set_xlabel('IKI(ms)')
    if position == 'keyboard':
        ax.set_ylabel('Gaze on Keyboard Ratio')
        ax.set_title('Average Gaze on Keyboard Ratio by IKI')
    else:
        ax.set_ylabel('Gaze on Text Entry Ratio')
        ax.set_title('Average Gaze on Text Entry Ratio by IKI')
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.075)
    ax.set_xticklabels(binned_iki_keys_proofreading, rotation=45)
    if position == 'keyboard':
        if save_dir:
            plt.savefig(osp.join(save_dir, 'average_gaze_on_keyboard_ratio_by_iki_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'average_gaze_on_keyboard_ratio_by_iki_' + gaze_data_source + '.png'))
    else:
        if save_dir:
            plt.savefig(osp.join(save_dir, 'average_gaze_on_text_entry_ratio_by_iki_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'average_gaze_on_text_entry_ratio_by_iki_' + gaze_data_source + '.png'))
    plt.show()

    # Calculate the IKI count (for full range up to 2000)
    binned_iki_keys_count = sorted(binned_ikis_count.keys())
    iki_count = [len(binned_ikis_count[bin]) for bin in binned_iki_keys_count]

    # Plot IKI count bar chart (bins < 2000)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=binned_iki_keys_count, y=iki_count)
    plt.xlabel('IKI(ms)')
    plt.ylabel('Count')
    plt.title('IKI Count')
    plt.xticks(rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.125, right=0.95, left=0.075)
    # Add avg IKI and std to the bottom right
    plt.text(0.95, 0.05, f'Average IKI: {np.mean(iki_mean_list):.2f}', horizontalalignment='right',
             verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.text(0.95, 0.1, f'IKI std: {np.std(iki_mean_list):.2f}', horizontalalignment='right',
             verticalalignment='bottom', transform=plt.gca().transAxes)
    if save_dir:
        plt.savefig(osp.join(save_dir, 'iki_count_by_bins_' + gaze_data_source + '.png'))
    else:
        plt.savefig(osp.join(FIG_DIR, 'iki_count_by_bins_' + gaze_data_source + '.png'))
    plt.show()


def get_key_vs_proofreading(gaze_df, typing_df, position='keyboard'):
    trailtime = 0
    last_trailtime = 0
    keys = {}
    for index, row in gaze_df.iterrows():
        trailtime += row['duration']
        gaze_df.loc[index, 'trailtime'] = trailtime
        gaze_df.loc[index, 'temp_index'] = index

    # skip the first row of the typing_df, since the first row is the start of the typing
    key = typing_df.iloc[0]['key']
    keys[key] = []
    if position == 'keyboard':
        y_limit = 1230
        if gaze_df.iloc[0]['y'] >= y_limit:
            keys[key].append(1.0)
        else:
            keys[key].append(0.0)
    else:
        y_limit = 225
        if gaze_df.iloc[0]['y'] <= y_limit:
            keys[key].append(1.0)
        else:
            keys[key].append(0.0)

    for i in range(1, len(typing_df)):
        typing_row = typing_df.iloc[i]
        trailtime = typing_row['trailtime']
        # get all the gaze data between last_trailtime and trailtime
        window_gaze_df = gaze_df[(gaze_df['trailtime'] >= last_trailtime) & (gaze_df['trailtime'] <= trailtime)]
        # extend the window_gaze_df with the previous gaze_df of the first row and the next gaze_df of the last row
        # first make sure the first row and the last row exist in the gaze_df
        if not window_gaze_df.empty:
            if window_gaze_df.iloc[-1]['temp_index'] != len(gaze_df) - 1:
                window_gaze_df = pd.concat([window_gaze_df,
                                            gaze_df[
                                                gaze_df['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] + 1]])

        if window_gaze_df.empty:
            continue
        key = typing_row['key']
        proofreading_time = 0
        keyboard_time = 0
        for _, gaze_row in window_gaze_df.iterrows():
            if key not in keys:
                keys[key] = []
            if position == 'keyboard':
                if gaze_row['y'] >= y_limit:
                    is_proofreading = 1
                    if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                        proofreading_time += gaze_row['trailtime'] - last_trailtime
                    elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                        'trailtime'] > trailtime:
                        proofreading_time += trailtime - window_gaze_df.iloc[-2]['trailtime']
                    else:
                        proofreading_time += gaze_row['duration']
                else:
                    if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                        keyboard_time += gaze_row['trailtime'] - last_trailtime
                    elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                        'trailtime'] > trailtime:
                        keyboard_time += trailtime - window_gaze_df.iloc[-2]['trailtime']
                    else:
                        keyboard_time += gaze_row['duration']
            else:
                if gaze_row['y'] <= y_limit:
                    is_proofreading = 1
                    if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                        proofreading_time += gaze_row['trailtime'] - last_trailtime
                    elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                        'trailtime'] > trailtime:
                        proofreading_time += trailtime - window_gaze_df.iloc[-2]['trailtime']
                    else:
                        proofreading_time += gaze_row['duration']
                else:
                    if gaze_row['temp_index'] == window_gaze_df.iloc[0]['temp_index']:
                        keyboard_time += gaze_row['trailtime'] - last_trailtime
                    elif gaze_row['temp_index'] == window_gaze_df.iloc[-1]['temp_index'] and gaze_row[
                        'trailtime'] > trailtime:
                        keyboard_time += trailtime - window_gaze_df.iloc[-2]['trailtime']
                    else:
                        keyboard_time += gaze_row['duration']
        if proofreading_time + keyboard_time > 1e-6:
            keys[key].append(proofreading_time / (proofreading_time + keyboard_time))
        else:
            keys[key].append(0.0)
        last_trailtime = trailtime
    return keys


def process_key_vs_position(all_keys, gaze_data_source='ground_truth', save_dir=None, position='keyboard'):
    # Set global font sizes
    plt.rcParams.update({
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 14,  # X and Y label font size
        'xtick.labelsize': 12,  # X tick labels font size
        'ytick.labelsize': 14,  # Y tick labels font size
        'legend.fontsize': 16,  # Legend font size
        'text.color': 'black',  # Text color
    })
    # sort the keys based on their row order for visualization
    first_row_char = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']
    second_row_char = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']
    third_row_char = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    backspace_char = ['B']  # Backspace should have its own color
    fourth_row_char = [' ']  # Space is on the fourth row

    # Combine the rows and filter out keys not in all_keys
    ordered_keys = [key for key in
                    (first_row_char + second_row_char + third_row_char + backspace_char + fourth_row_char) if
                    key in all_keys]

    # Filter the proofreading data based on the ordered keys
    proofreading_rate = [np.mean(all_keys[key]) for key in ordered_keys]
    proofreading_std = [np.std(all_keys[key]) for key in ordered_keys]

    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = proofreading_rate + proofreading_std
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]

    # Change " " to "space" and "B" to "backspace" for better visualization
    ordered_keys = [key.replace(" ", "space").replace("B", "backspace") for key in ordered_keys]

    # Define colors for each row and special keys (5 colors)
    row_colors = ['#ADD8E6',  # Light blue for 1st row
                  '#87CEEB',  # Slightly darker blue for 2nd row
                  '#4682B4',  # Medium blue for 3rd row
                  '#1E90FF',  # Darker blue for backspace
                  '#00008B']  # Darkest blue for space
    key_colors = []
    for key in ordered_keys:
        if key in ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']:
            key_colors.append(row_colors[0])  # First row
        elif key in ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']:
            key_colors.append(row_colors[1])  # Second row
        elif key in ['z', 'x', 'c', 'v', 'b', 'n', 'm']:
            key_colors.append(row_colors[2])  # Third row
        elif key == 'backspace':
            key_colors.append(row_colors[3])  # Backspace with its own color
        elif key == 'space':
            key_colors.append(row_colors[4])  # Space with its own color

    # Plot proofreading rate with error bars
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=ordered_keys, y=proofreading_rate, palette=key_colors)
    ax.errorbar(x=np.arange(len(ordered_keys)), y=proofreading_rate, yerr=asymmetric_error, fmt='none', c='black',
                capsize=5)
    plt.xlabel('Key')
    if position == 'keyboard':
        plt.ylabel('Gaze on Keyboard Ratio')
        plt.title('Gaze on Keyboard Ratio for Each Key', fontsize=16)
    else:
        plt.ylabel('Gaze on Text Entry Ratio')
        plt.title('Gaze on Text Entry Ratio for Each Key', fontsize=16)
    plt.xticks(rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.18125, right=0.975, left=0.075)
    plt.ylim(0, 1)

    # Save the figure
    if position == 'keyboard':
        if save_dir:
            plt.savefig(osp.join(save_dir, 'gaze_on_keyboard_ratio_for_each_key_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'gaze_on_keyboard_ratio_for_each_key_' + gaze_data_source + '.png'))
    else:
        if save_dir:
            plt.savefig(osp.join(save_dir, 'gaze_on_text_entry_ratio_for_each_key_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR, 'gaze_on_text_entry_ratio_for_each_key_' + gaze_data_source + '.png'))
    plt.show()

    # plot the proofreading rate of "_", "B" and other keys
    summary_keys = {"1st row": [], "2nd row": [], "3rd row": [], "backspace": [], "space": []}
    for key, proofreading in all_keys.items():
        if key == " ":
            summary_keys["space"].extend(proofreading)
        elif key == "B":
            summary_keys["backspace"].extend(proofreading)
        elif key in first_row_char:
            summary_keys["1st row"].extend(proofreading)
        elif key in second_row_char:
            summary_keys["2nd row"].extend(proofreading)
        elif key in third_row_char:
            summary_keys["3rd row"].extend(proofreading)

    keys = list(summary_keys.keys())
    proofreading_rate = [np.mean(summary_keys[key]) for key in keys]
    proofreading_std = [np.std(summary_keys[key]) for key in keys]

    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = proofreading_rate + proofreading_std
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]

    key_colors = [
        row_colors[0],  # 1st row color
        row_colors[1],  # 2nd row color
        row_colors[2],  # 3rd row color
        row_colors[3],  # Space color
        row_colors[4],  # Backspace color
    ]

    # Plot proofreading rate with error bars
    plt.figure(figsize=(5, 6))
    ax = sns.barplot(x=keys, y=proofreading_rate, palette=key_colors)
    ax.errorbar(x=np.arange(len(keys)), y=proofreading_rate, yerr=asymmetric_error, fmt='none', c='black', capsize=5)
    plt.xlabel('Key')
    if position == 'keyboard':
        plt.ylabel('Gaze on Keyboard Ratio')
        plt.title('Gaze on Keyboard Ratio for keys', fontsize=16)
    else:
        plt.ylabel('Gaze on Text Entry Ratio')
        plt.title('Gaze on Text Entry Ratio for keys', fontsize=16)
    plt.xticks(rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.18125, right=0.99, left=0.15)
    plt.ylim(0, 1)

    # Save the figure
    if position == 'keyboard':
        if save_dir:
            plt.savefig(osp.join(save_dir,
                                 'gaze_on_keyboard_ratio_for_space_backspace_other_keys_' + gaze_data_source + '.png'))
        else:
            plt.savefig(
                osp.join(FIG_DIR, 'gaze_on_keyboard_ratio_for_space_backspace_other_keys_' + gaze_data_source + '.png'))
    else:
        if save_dir:
            plt.savefig(osp.join(save_dir,
                                 'gaze_on_text_entry_ratio_for_space_backspace_other_keys_' + gaze_data_source + '.png'))
        else:
            plt.savefig(osp.join(FIG_DIR,
                                 'gaze_on_text_entry_ratio_for_space_backspace_other_keys_' + gaze_data_source + '.png'))
    plt.show()


def distance_and_cosine_similarity_analysis(index, typing_data, gaze_log, screen_size=(1080, 1920), fps=30,
                                            model_type='transformer', data_use='human', amortized_inference=True,
                                            gaze_data_source='ground_truth'):
    # if the gaze_log is not df, change it to df, the columns are x, y, duration
    if not isinstance(gaze_log, pd.DataFrame):
        gaze_log = pd.DataFrame(gaze_log, columns=['x', 'y', 'duration'])
    typing_data.loc[typing_data.index[0], 'duration'] = 0
    # compute the trailtime for both typing and gaze at each row
    typing_data['trailtime'] = typing_data['duration'].cumsum()
    gaze_log['trailtime'] = gaze_log['duration'].cumsum()

    distances = {}
    similarities = {}
    gaze_on_keyboard_ratio = 0
    gaze_on_keyboard_ratio_list = []

    for _, typing_row in typing_data.iterrows():
        is_proofreading = []
        trailtime = typing_row['trailtime']
        window_gaze_df = gaze_log[(gaze_log['trailtime'] >= trailtime + tail_offset) &
                                  (gaze_log['trailtime'] <= trailtime + head_offset)]
        if window_gaze_df.empty:
            continue
        for _, gaze_row in window_gaze_df.iterrows():
            if gaze_row['y'] < 0.6 * screen_size[0]:
                is_proofreading.append(1)
                continue
            is_proofreading.append(0)
            offset = gaze_row['trailtime'] - trailtime
            dist = np.linalg.norm([gaze_row['x'] - typing_row['x'], gaze_row['y'] - typing_row['y']])
            gaze_vec = np.array([gaze_row['x'], gaze_row['y']]).reshape(1, -1)
            touch_vec = np.array([typing_row['x'], typing_row['y']]).reshape(1, -1)

            try:
                sim = cosine_similarity(gaze_vec, touch_vec)[0][0]
            except:
                continue
            if offset not in distances:
                distances[offset] = []
            distances[offset].append(dist)

            if offset not in similarities:
                similarities[offset] = []
            similarities[offset].append(sim)
        gaze_on_keyboard_ratio_list.append(1 - np.mean(is_proofreading))

    if gaze_on_keyboard_ratio_list:
        gaze_on_keyboard_ratio = np.mean(gaze_on_keyboard_ratio_list)

    return distances, similarities, gaze_on_keyboard_ratio


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Typing Log and Gaze Movements")
    parser.add_argument("--model_type", choices=["transformer"], default="transformer",
                        help="Type of model to use for prediction")
    parser.add_argument("--max_pred_len", type=int, default=32, help="Maximum number of gaze data points to predict")
    parser.add_argument("--loss_type", type=str, choices=['combined'], default='combined',
                        help="Loss function to use for training, use the default value as 'combined")
    parser.add_argument("--data_use", type=str, choices=['both', 'human'], default='both',
                        help="Use human data, simulated data, or both")
    parser.add_argument("--fpath_header", type=str, default='final_distribute',
                        help='File path header for data use, currently use the default value')
    parser.add_argument("--amortized-inference", action="store_true", help="Use amortized inference", default=True)
    parser.add_argument("--user_index", type=str, default=None, choices=['129', '130', '131', '132', '133', None],
                        help="Specific user for analysis")
    args = parser.parse_args()
    print("Visualizing with data use: ", args.data_use)
    print("Using amortized inference:", args.amortized_inference)
    analysis(args.model_type, 32, args.loss_type, args.data_use, args.fpath_header, args.amortized_inference,
             user_index=args.user_index, analysis_data_choice='test')


if __name__ == "__main__":
    main()
