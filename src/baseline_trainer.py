import sys
import os
import logging
import datetime
import time

# Detect the project root and add it to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.metrics import multi_match, dtw_scanpaths, sted_scanpaths
from sklearn.model_selection import KFold
import argparse
from data.data import load_and_preprocess_data
import os.path as osp
from config import GAZE_INFERENCE_DIR
from src.nets import (
    TypingGazeDataset,
    TransformerModel,
    multi_match_loss,
    finger_guiding_distance_loss,
    proofreading_loss,
)
from src.summary import Logger

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

saved_model_dir = osp.join(GAZE_INFERENCE_DIR, 'src', 'outputs')


def create_padding_mask(seq):
    seq = seq == 0
    return seq


def trim_to_length(data, length):
    return data.groupby('index').apply(lambda x: x.head(length)).reset_index(drop=True)


def train_model(
        model_type='transformer',
        include_duration=True,
        include_key=False,
        k_folds=5,
        max_pred_len=32,
        use_k_fold=True,
        num_epochs=1500,
        loss_type='combined',
        data_use='human',
        continue_training=False,
        start_epoch=0,
        pretrain_padding=False,
        pretrain_epochs=500,
        fpath_header='train',
        writer=None,
):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        include_key, include_duration=True, max_pred_len=max_pred_len, data_use=data_use, fpath_header=fpath_header,
        calculate_params=False)
    if loss_type == 'custom':
        lr = 0.00005
    else:
        lr = 0.0001
    if data_use != 'human':
        use_k_fold = False
        # lr = 0.00001
    input_dim = X_train.shape[2]  # Update input_dim to include the encoded keys if included
    num_epochs = num_epochs // k_folds if use_k_fold else num_epochs

    def initialize_model():
        return TransformerModel(input_dim=input_dim, output_dim=output_dim, dropout=0.1).to(device)

    # Initialize the model, loss function, and optimizer
    output_dim = 3 if include_duration else 2  # Set output dimension based on whether duration is included
    model = initialize_model()
    if use_k_fold:
        weight_decay_value = 5e-4
        count_loss_ratio = 0.8
    else:
        weight_decay_value = 4e-4
        count_loss_ratio = 0.64
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)
    mse_loss = nn.MSELoss(reduction='none')

    def lr_lambda(epoch):
        return 0.97 ** (epoch // 100)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Save the model and scalers
    model_filename = f'{model_type}_{loss_type}_{data_use}_model_with_key_with_duration.pth' if include_key and include_duration else \
        f'{model_type}_{loss_type}_{data_use}_model_with_key_without_duration.pth' if include_key else \
            f'{model_type}_{loss_type}_{data_use}_model_without_key_with_duration.pth' if include_duration else \
                f'{model_type}_{loss_type}_{data_use}_model_without_key_without_duration.pth'

    if not osp.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    if use_k_fold:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            k_fold_lr = lr * 0.9 ** fold
            optimizer = optim.Adam(model.parameters(), lr=k_fold_lr, weight_decay=weight_decay_value)
            print(f'Fold {fold + 1}/{k_folds}')
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            masks_x_fold_train, masks_x_fold_val = masks_x_train[train_idx], masks_x_train[val_idx]
            masks_y_fold_train, masks_y_fold_val = masks_y_train[train_idx], masks_y_train[val_idx]

            train_dataset = TypingGazeDataset(X_fold_train, y_fold_train, masks_x_fold_train, masks_y_fold_train,
                                              indices_train)
            val_dataset = TypingGazeDataset(X_fold_val, y_fold_val, masks_x_fold_val, masks_y_fold_val, indices_train)
            test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # Training loop
            for epoch in tqdm(range(num_epochs)):
                model.train()
                running_loss = 0.0
                running_gaze_padding_loss = 0.0
                running_gaze_count_loss = 0.0
                running_gaze_distance_loss = 0.0
                running_proofreading_duration_loss = 0.0
                running_proofreading_count_loss = 0.0
                for inputs, targets, masks_x, masks_y, _ in train_loader:
                    inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                        device), masks_y.to(device)
                    src_mask = create_padding_mask(masks_x) if model_type == "transformer" else None
                    optimizer.zero_grad()
                    gaze_mean, gaze_log_std, padding_outputs = model(inputs,
                                                                     src_mask) if model_type == "transformer" else model(
                        inputs)
                    # Compute loss with masking
                    gaze_std = torch.exp(gaze_log_std)
                    epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                    gaze_outputs = gaze_mean + gaze_std * epsilon
                    if loss_type == 'mse':
                        gaze_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                        gaze_loss = gaze_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                        gaze_loss = gaze_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                        padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                        padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                        loss = gaze_loss + padding_loss  # Combine the two losses
                    elif loss_type == 'combined':
                        padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                        padding_loss = padding_loss.sum() / masks_y.sum()
                        typing_gaze_distance_loss, typing_gaze_count_loss = finger_guiding_distance_loss(
                            gaze_outputs[:, 1:max_pred_len],
                            targets[:max_pred_len],
                            inputs,
                            scaler_X,
                            scaler_y,
                            masks_x,
                            masks_y[:, 1:max_pred_len],
                            padding_outputs[:max_pred_len],
                            max_pred_len=max_pred_len,
                        )
                        proofreading_duration_loss, proofreading_count_loss = proofreading_loss(
                            gaze_outputs[:, 1:max_pred_len],
                            targets[:max_pred_len],
                            scaler_y,
                            masks_y[:, 1:max_pred_len],
                            padding_outputs[:max_pred_len],
                            max_pred_len=max_pred_len,
                        )
                        gaze_multimatch_loss, _, _, _, _, _ = multi_match_loss(gaze_outputs[:max_pred_len],
                                                                               targets[:max_pred_len],
                                                                               scaler_y=scaler_y,
                                                                               masks_y=masks_y,
                                                                               predict_masks_y=padding_outputs[
                                                                                               :max_pred_len],
                                                                               pos_weight=0.5,
                                                                               dur_weight=0.5)
                        gaze_mse_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                        gaze_mse_loss = gaze_mse_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                        gaze_mse_loss = gaze_mse_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                        loss = (
                                gaze_multimatch_loss
                                + padding_loss
                                + gaze_mse_loss
                                + typing_gaze_distance_loss
                                + typing_gaze_count_loss
                                + proofreading_duration_loss
                                + proofreading_count_loss
                        )
                    else:
                        raise ValueError("Unsupported loss function. Choose 'mse' or 'combined")

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # running_gaze_padding_loss += (
                    #         padding_loss.item() + gaze_multimatch_loss.item() + gaze_mse_loss.item()
                    # )
                    running_gaze_padding_loss += (
                            padding_loss.item() + gaze_multimatch_loss.item()
                    )
                    running_gaze_count_loss += typing_gaze_count_loss.item()
                    running_gaze_distance_loss += typing_gaze_distance_loss.item()
                    running_proofreading_duration_loss += proofreading_duration_loss.item()
                    running_proofreading_count_loss += proofreading_count_loss.item()

                avg_running_loss = running_loss / len(train_loader)
                avg_gaze_padding_loss = running_gaze_padding_loss / len(train_loader)
                avg_gaze_count_loss = running_gaze_count_loss / len(train_loader)
                avg_gaze_distance_loss = running_gaze_distance_loss / len(train_loader)
                avg_proofreading_duration_loss = running_proofreading_duration_loss / len(train_loader)
                avg_proofreading_count_loss = running_proofreading_count_loss / len(train_loader)

                logging.info(
                    f'Fold {fold + 1}/{k_folds}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_running_loss:.4f}, '
                    f'Gaze Loss: {avg_gaze_padding_loss:.4f}, '
                    f'Gaze Count Loss: {avg_gaze_count_loss:.4f}, '
                    f'Gaze Distance Loss: {avg_gaze_distance_loss:.4f} '
                    f'Proofreading Duration Loss: {avg_proofreading_duration_loss:.4f} '
                    f'Proofreading Count Loss: {avg_proofreading_count_loss:.4f} '
                )

                # Log to TensorBoard
                if writer:
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Loss', avg_running_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Gaze_Loss', avg_gaze_padding_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Gaze_Count_Loss', avg_gaze_count_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Gaze_Distance_Loss', avg_gaze_distance_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Proofreading_Duration_Loss',
                                      avg_proofreading_duration_loss,
                                      epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Train/Proofreading_Count_Loss', avg_proofreading_count_loss,
                                      epoch)

            # Validation
            model.eval()
            val_loss = 0.0
            val_gaze_loss = 0.0
            val_gaze_distance_loss = 0.0
            val_gaze_count_loss = 0.0
            val_proofreading_duration_loss = 0.0
            val_proofreading_count_loss = 0.0
            val_loaders = [val_loader, test_loader]
            for loader in val_loaders:
                with torch.no_grad():
                    for inputs, targets, masks_x, masks_y, _ in loader:
                        inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                            device), masks_y.to(device)
                        src_mask = create_padding_mask(masks_x) if model_type == "transformer" else None
                        gaze_mean, gaze_log_std, padding_outputs = model(inputs,
                                                                         src_mask) if model_type == "transformer" else model(
                            inputs)
                        # Compute loss with masking
                        gaze_std = torch.exp(gaze_log_std)
                        epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                        gaze_outputs = gaze_mean + gaze_std * epsilon
                        # Compute loss with masking
                        if loss_type == 'mse':
                            gaze_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                            gaze_loss = gaze_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                            gaze_loss = gaze_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                            padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                            padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                            loss = gaze_loss + padding_loss  # Combine the two losses
                        elif loss_type == 'combined':
                            padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                            padding_loss = padding_loss.sum() / masks_y.sum()
                            typing_gaze_distance_loss, typing_gaze_count_loss = finger_guiding_distance_loss(
                                gaze_outputs[:, 1:max_pred_len],
                                targets[:max_pred_len],
                                inputs,
                                scaler_X,
                                scaler_y,
                                masks_x,
                                masks_y[:, 1:max_pred_len],
                                padding_outputs[:max_pred_len],
                                max_pred_len=max_pred_len,
                            )
                            proofreading_duration_loss, proofreading_count_loss = proofreading_loss(
                                gaze_outputs[:, 1:max_pred_len],
                                targets[:max_pred_len],
                                scaler_y,
                                masks_y[:, 1:max_pred_len],
                                padding_outputs[:max_pred_len],
                                max_pred_len=max_pred_len,
                            )
                            gaze_multimatch_loss, _, _, _, _, _ = multi_match_loss(gaze_outputs[:max_pred_len],
                                                                                   targets[:max_pred_len],
                                                                                   scaler_y=scaler_y,
                                                                                   masks_y=masks_y,
                                                                                   predict_masks_y=padding_outputs[
                                                                                                   :max_pred_len],
                                                                                   pos_weight=0.5,
                                                                                   dur_weight=0.5)
                            gaze_mse_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                            gaze_mse_loss = gaze_mse_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                            gaze_mse_loss = gaze_mse_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                            loss = (
                                    gaze_multimatch_loss
                                    + padding_loss
                                    + gaze_mse_loss
                                    + typing_gaze_distance_loss
                                    + typing_gaze_count_loss
                                    + proofreading_duration_loss
                                    + proofreading_count_loss
                            )
                        else:
                            raise ValueError("Unsupported loss function. Choose 'mse' or 'combined")

                        val_loss += loss.item()
                        val_gaze_loss += gaze_multimatch_loss.item() + gaze_mse_loss.item() + padding_loss.item()
                        val_gaze_distance_loss += typing_gaze_distance_loss.item()
                        val_gaze_count_loss += typing_gaze_count_loss.item()
                        val_proofreading_duration_loss += proofreading_duration_loss.item()
                        val_proofreading_count_loss += proofreading_count_loss.item()

                avg_val_loss = val_loss / len(loader)
                avg_val_gaze_loss = val_gaze_loss / len(loader)
                avg_val_gaze_distance_loss = val_gaze_distance_loss / len(loader)
                avg_val_gaze_count_loss = val_gaze_count_loss / len(loader)
                avg_val_proofreading_duration_loss = val_proofreading_duration_loss / len(loader)
                avg_val_proofreading_count_loss = val_proofreading_count_loss / len(loader)

                logging.info(
                    f'Validation Loss for fold {fold + 1}/{k_folds}: {avg_val_loss:.4f} '
                    f'Gaze Loss: {avg_val_gaze_loss:.4f} '
                    f'Gaze Distance Loss: {avg_val_gaze_distance_loss:.4f} '
                    f'Gaze Count Loss: {avg_val_gaze_count_loss:.4f} '
                    f'Proofreading Duration Loss: {avg_val_proofreading_duration_loss:.4f} '
                    f'Proofreading Count Loss: {avg_val_proofreading_count_loss:.4f} '
                )

                # Log to TensorBoard
                if writer:
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Loss', avg_val_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Gaze_Loss', avg_val_gaze_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Gaze_Distance_Loss', avg_val_gaze_distance_loss,
                                      epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Gaze_Count_Loss', avg_val_gaze_count_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Proofreading_Duration_Loss',
                                      avg_val_proofreading_duration_loss, epoch)
                    writer.add_scalar(f'Fold_{fold + 1}/Validation/Proofreading_Count_Loss',
                                      avg_val_proofreading_count_loss, epoch)

            logging.info(f'Saving for Fold {fold + 1}/{k_folds}')
            torch.save(model.state_dict(), osp.join(saved_model_dir, model_filename))
            torch.save(scaler_X, 'outputs/scaler_X.pth')
            torch.save(scaler_y, 'outputs/scaler_y.pth')

        # Save the model and scalers after the final fold
        torch.save(model.state_dict(), osp.join(saved_model_dir, model_filename))
        torch.save(scaler_X, 'outputs/scaler_X.pth')
        torch.save(scaler_y, 'outputs/scaler_y.pth')
    else:
        # Use a simple train-test split instead of k-fold cross-validation
        test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        train_dataset = TypingGazeDataset(X_train, y_train, masks_x_train, masks_y_train, indices_train)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Use a larger batch size

        if continue_training and osp.exists(osp.join(saved_model_dir, model_filename)):
            logging.info("Loading previously saved model and scalers...")
            model.load_state_dict(torch.load(osp.join(saved_model_dir, model_filename)))
            scaler_X = torch.load('outputs/scaler_X.pth')
            scaler_y = torch.load('outputs/scaler_y.pth')
        else:
            start_epoch = 0
        # Training loop
        if pretrain_padding:
            for epoch in tqdm(range(start_epoch, start_epoch + pretrain_epochs)):
                model.train()
                running_padding_loss = 0.0
                for inputs, targets, masks_x, masks_y, _ in train_loader:
                    inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                        device), masks_y.to(device)
                    src_mask = create_padding_mask(masks_x) if model_type == "transformer" else None
                    optimizer.zero_grad()

                    _, _, padding_outputs = model(inputs, src_mask) if model_type == "transformer" else model(inputs)

                    padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                    padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                    padding_loss.backward()
                    optimizer.step()
                    running_padding_loss += padding_loss.item()

                avg_padding_loss = running_padding_loss / len(train_loader)
                logging.info(
                    f'Epoch {epoch + 1}/{start_epoch + pretrain_epochs}, Padding Loss: {avg_padding_loss:.4f}'
                )

                # Log to TensorBoard
                if writer:
                    writer.add_scalar('Pretrain_Padding_Loss', avg_padding_loss, epoch)

                if epoch % 100 == 0:
                    logging.info(f'Saving model after pretrain epoch {epoch + 1}')
                    torch.save(model.state_dict(), osp.join(saved_model_dir, model_filename))
                    torch.save(scaler_X, 'outputs/scaler_X.pth')
                    torch.save(scaler_y, 'outputs/scaler_y.pth')
        else:
            pretrain_epochs = 0

        for epoch in tqdm(range(start_epoch + pretrain_epochs, num_epochs)):
            if epoch > 5000:
                distance_loss_ratio = 6
            else:
                distance_loss_ratio = 5
            model.train()
            running_loss = 0.0
            running_gaze_padding_loss = 0.0
            running_gaze_count_loss = 0.0
            running_gaze_distance_loss = 0.0
            running_proofreading_duration_loss = 0.0
            running_proofreading_count_loss = 0.0
            for inputs, targets, masks_x, masks_y, _ in train_loader:
                inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                    device), masks_y.to(device)
                src_mask = create_padding_mask(masks_x) if model_type == "transformer" else None
                optimizer.zero_grad()
                gaze_mean, gaze_log_std, padding_outputs = model(inputs,
                                                                 src_mask) if model_type == "transformer" else model(
                    inputs)
                # Compute loss with masking
                gaze_std = torch.exp(gaze_log_std)
                epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                gaze_outputs = gaze_mean + gaze_std * epsilon
                # Compute loss with masking
                if loss_type == 'mse':
                    gaze_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                    gaze_loss = gaze_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                    gaze_loss = gaze_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                    padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(padding_outputs[:max_pred_len],
                                                                                        masks_y[:max_pred_len].float())
                    padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                    loss = gaze_loss + padding_loss  # Combine the two losses
                elif loss_type == 'combined':
                    padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                    padding_loss = padding_loss.sum() / masks_y.sum()
                    typing_gaze_distance_loss, typing_gaze_count_loss = finger_guiding_distance_loss(
                        gaze_outputs[:, 1:max_pred_len],
                        targets[:max_pred_len],
                        inputs,
                        scaler_X,
                        scaler_y,
                        masks_x,
                        masks_y[:, 1:max_pred_len],
                        padding_outputs[:max_pred_len],
                        max_pred_len=max_pred_len,
                        distance_loss_ratio=distance_loss_ratio,
                    )
                    proofreading_duration_loss, proofreading_count_loss = proofreading_loss(
                        gaze_outputs[:, 1:max_pred_len],
                        targets[:max_pred_len],
                        scaler_y,
                        masks_y[:, 1:max_pred_len],
                        padding_outputs[:max_pred_len],
                        max_pred_len=max_pred_len,
                        count_loss_ratio=count_loss_ratio,
                    )
                    """At the beginning decrease the proofreading influence"""
                    if epoch < 1000:
                        proofreading_duration_loss_rate = 0.5
                        proofreading_count_loss_rate = 0.5
                    elif epoch < 2500:
                        proofreading_duration_loss_rate = 0.75
                        proofreading_count_loss_rate = 0.75
                    else:
                        proofreading_duration_loss_rate = 2
                        proofreading_count_loss_rate = 2
                    proofreading_duration_loss *= proofreading_duration_loss_rate
                    proofreading_count_loss *= proofreading_count_loss_rate

                    gaze_multimatch_loss, _, _, _, _, _ = multi_match_loss(gaze_outputs[:max_pred_len],
                                                                           targets[:max_pred_len],
                                                                           scaler_y=scaler_y,
                                                                           masks_y=masks_y,
                                                                           predict_masks_y=padding_outputs[
                                                                                           :max_pred_len],
                                                                           pos_weight=0.5,
                                                                           dur_weight=0.5)
                    gaze_mse_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                    gaze_mse_loss = gaze_mse_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                    gaze_mse_loss = gaze_mse_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                    loss = (
                            gaze_multimatch_loss
                            + padding_loss
                            + gaze_mse_loss
                            + typing_gaze_distance_loss
                            + typing_gaze_count_loss
                            + proofreading_duration_loss
                            + proofreading_count_loss
                    )
                else:
                    raise ValueError("Unsupported loss function. Choose 'mse' or 'combined")

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_gaze_padding_loss += (
                        padding_loss.item() + gaze_multimatch_loss.item() + gaze_mse_loss.item()
                )
                running_gaze_count_loss += typing_gaze_count_loss.item()
                running_gaze_distance_loss += typing_gaze_distance_loss.item()
                running_proofreading_duration_loss += proofreading_duration_loss.item()
                running_proofreading_count_loss += proofreading_count_loss.item()

            avg_running_loss = running_loss / len(train_loader)
            avg_gaze_padding_loss = running_gaze_padding_loss / len(train_loader)
            avg_gaze_count_loss = running_gaze_count_loss / len(train_loader)
            avg_gaze_distance_loss = running_gaze_distance_loss / len(train_loader)
            avg_proofreading_duration_loss = running_proofreading_duration_loss / len(train_loader)
            avg_proofreading_count_loss = running_proofreading_count_loss / len(train_loader)

            logging.info(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_running_loss:.4f}, '
                f'Gaze Loss: {avg_gaze_padding_loss:.4f}, '
                f'Gaze Count Loss: {avg_gaze_count_loss:.4f}, '
                f'Gaze Distance Loss: {avg_gaze_distance_loss:.4f} '
                f'Proofreading Duration Loss: {avg_proofreading_duration_loss:.4f} '
                f'Proofreading Count Loss: {avg_proofreading_count_loss:.4f} '
            )

            # Log to TensorBoard
            if writer:
                writer.add_scalar('Train/Loss', avg_running_loss, epoch)
                writer.add_scalar('Train/Gaze_Loss', avg_gaze_padding_loss, epoch)
                writer.add_scalar('Train/Gaze_Count_Loss', avg_gaze_count_loss, epoch)
                writer.add_scalar('Train/Gaze_Distance_Loss', avg_gaze_distance_loss, epoch)
                writer.add_scalar('Train/Proofreading_Duration_Loss', running_proofreading_duration_loss, epoch)
                writer.add_scalar('Train/Proofreading_Count_Loss', running_proofreading_count_loss, epoch)

            scheduler.step()

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                logging.info(f'Saving model after epoch {epoch + 1} \n')
                # torch.save(model.state_dict(), osp.join(saved_model_dir, model_filename))
                # torch.save(scaler_X, 'outputs/scaler_X.pth')
                # torch.save(scaler_y, 'outputs/scaler_y.pth')

                # Test the model
                model.eval()
                test_loss = 0.0
                test_gaze_loss = 0.0
                test_gaze_distance_loss = 0.0
                test_gaze_count_loss = 0.0
                test_proofreading_duration_loss = 0.0
                test_proofreading_count_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, masks_x, masks_y, _ in test_loader:
                        inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                            device), masks_y.to(device)
                        src_mask = create_padding_mask(masks_x) if model_type == "transformer" else None
                        gaze_mean, gaze_log_std, padding_outputs = model(inputs,
                                                                         src_mask) if model_type == "transformer" else model(
                            inputs)
                        gaze_std = torch.exp(gaze_log_std)
                        epsilon = torch.randn_like(gaze_mean)  # Same shape as gaze_mean
                        gaze_outputs = gaze_mean + gaze_std * epsilon
                        # Compute loss with masking
                        if loss_type == 'mse':
                            gaze_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                            gaze_loss = gaze_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                            gaze_loss = gaze_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                            padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                            padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                            loss = gaze_loss + padding_loss  # Combine the two losses
                        elif loss_type == 'combined':
                            padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                            padding_loss = padding_loss.sum() / masks_y.sum()
                            typing_gaze_distance_loss, typing_gaze_count_loss = finger_guiding_distance_loss(
                                gaze_outputs[:, 1:max_pred_len],
                                targets[:max_pred_len],
                                inputs,
                                scaler_X,
                                scaler_y,
                                masks_x,
                                masks_y[:, 1:max_pred_len],
                                padding_outputs[:max_pred_len],
                                max_pred_len=max_pred_len,
                            )
                            proofreading_duration_loss, proofreading_count_loss = proofreading_loss(
                                gaze_outputs[:, 1:max_pred_len],
                                targets[:max_pred_len],
                                scaler_y,
                                masks_y[:, 1:max_pred_len],
                                padding_outputs[:max_pred_len],
                                max_pred_len=max_pred_len,
                                count_loss_ratio=count_loss_ratio,
                            )
                            gaze_multimatch_loss, _, _, _, _, _ = multi_match_loss(gaze_outputs[:max_pred_len],
                                                                                   targets[:max_pred_len],
                                                                                   scaler_y=scaler_y,
                                                                                   masks_y=masks_y,
                                                                                   predict_masks_y=padding_outputs[
                                                                                                   :max_pred_len],
                                                                                   pos_weight=0.5,
                                                                                   dur_weight=0.5)
                            gaze_mse_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                            gaze_mse_loss = gaze_mse_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                            gaze_mse_loss = gaze_mse_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                            loss = (
                                    gaze_multimatch_loss
                                    + padding_loss
                                    + gaze_mse_loss
                                    + typing_gaze_distance_loss
                                    + typing_gaze_count_loss
                                    + proofreading_duration_loss
                                    + proofreading_count_loss
                            )
                        else:
                            raise ValueError("Unsupported loss function. Choose 'mse' or 'combined")

                        test_loss += loss.item()
                        test_gaze_loss += gaze_multimatch_loss.item() + gaze_mse_loss.item() + padding_loss.item()
                        test_gaze_distance_loss += typing_gaze_distance_loss.item()
                        test_gaze_count_loss += typing_gaze_count_loss.item()
                        test_proofreading_duration_loss += proofreading_duration_loss.item()
                        test_proofreading_count_loss += proofreading_count_loss.item()

                avg_test_loss = test_loss / len(test_loader)
                avg_test_gaze_loss = test_gaze_loss / len(test_loader)
                avg_test_gaze_distance_loss = test_gaze_distance_loss / len(test_loader)
                avg_test_gaze_count_loss = test_gaze_count_loss / len(test_loader)
                avg_test_proofreading_duration_loss = test_proofreading_duration_loss / len(test_loader)
                avg_test_proofreading_count_loss = test_proofreading_count_loss / len(test_loader)

                logging.info(
                    f'Test Loss: {avg_test_loss:.4f}, '
                    f'Gaze Loss: {avg_test_gaze_loss:.4f}, '
                    f'Gaze Distance Loss: {avg_test_gaze_distance_loss:.4f}, '
                    f'Gaze Count Loss: {avg_test_gaze_count_loss:.4f} '
                    f'Proofreading Duration Loss: {avg_test_proofreading_duration_loss:.4f} '
                    f'Proofreading Count Loss: {avg_test_proofreading_count_loss:.4f} '
                )

                # # Log to TensorBoard
                # if writer:
                #     writer.add_scalar('Test/Loss', avg_test_loss, epoch)
                #     writer.add_scalar('Test/Gaze_Loss', avg_test_gaze_loss, epoch)
                #     writer.add_scalar('Test/Gaze_Distance_Loss', avg_test_gaze_distance_loss, epoch)
                #     writer.add_scalar('Test/Gaze_Count_Loss', avg_test_gaze_count_loss, epoch)
                #     writer.add_scalar('Test/Proofreading_Duration_Loss', avg_test_proofreading_duration_loss, epoch)
                #     writer.add_scalar('Test/Proofreading_Count_Loss', avg_test_proofreading_count_loss, epoch)

                torch.save(model.state_dict(), osp.join(saved_model_dir, model_filename))
                torch.save(scaler_X, 'outputs/scaler_X.pth')
                torch.save(scaler_y, 'outputs/scaler_y.pth')


def test_model(
        model_type='transformer',
        include_duration=True,
        include_key=False,
        max_pred_len=32,
        loss_type='combined',
        log_index=[],
        data_use='human',
        fpath_header='train',
        use_best_model=False,
):
    X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
        include_key, include_duration=True, max_pred_len=max_pred_len, data_use=data_use, fpath_header=fpath_header,
        calculate_params=False)

    if use_best_model:
        saved_model_dir = osp.join(GAZE_INFERENCE_DIR, 'scr', 'best_outputs')
    else:
        saved_model_dir = osp.join(GAZE_INFERENCE_DIR, 'scr', 'outputs')

    test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset = TypingGazeDataset(X_train, y_train, masks_x_train, masks_y_train, indices_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # Process one trail at a time

    # Initialize the model
    input_dim = X_train.shape[2]  # Update input_dim to include the encoded keys if included
    output_dim = 3 if include_duration else 2  # Set output dimension based on whether duration is included

    def initialize_model():
        return TransformerModel(input_dim=input_dim, output_dim=output_dim, dropout=0.1).to(device)

    model = initialize_model()
    model_filename = f'{model_type}_{loss_type}_{data_use}_model_with_key_with_duration.pth' if include_key and include_duration else \
        f'{model_type}_{loss_type}_{data_use}_model_with_key_without_duration.pth' if include_key else \
            f'{model_type}_{loss_type}_{data_use}_model_without_key_with_duration.pth' if include_duration else \
                f'{model_type}_{loss_type}_{data_use}_model_without_key_without_duration.pth'
    last_modified_time = time.ctime(os.path.getmtime(osp.join(saved_model_dir, model_filename)))
    print("Loading model from {}".format(osp.join(saved_model_dir, model_filename)))
    print(f"Model was last modified on: {last_modified_time}")
    model.load_state_dict(torch.load(osp.join(saved_model_dir, model_filename), map_location=device))
    model.eval()

    mse_loss = nn.MSELoss(reduction='none')

    if data_use != 'human':
        X_train, X_test, y_train, y_test, masks_x_train, masks_x_test, masks_y_train, masks_y_test, indices_train, indices_test, scaler_X, scaler_y, typing_data, gaze_data = load_and_preprocess_data(
            include_key, include_duration=True, max_pred_len=max_pred_len, data_use='human',
            fpath_header=fpath_header,
            calculate_params=False)
        test_dataset = TypingGazeDataset(X_test, y_test, masks_x_test, masks_y_test, indices_test)
        human_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataset = TypingGazeDataset(X_train, y_train, masks_x_train, masks_y_train, indices_train)
        human_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # Process one trail at a time

        # Initialize the model
        input_dim = X_train.shape[2]  # Update input_dim to include the encoded keys if included
        output_dim = 3 if include_duration else 2  # Set output dimension based on whether duration is included
    if data_use == "simulated":
        loader_list = [train_loader, test_loader, human_test_loader]
    elif data_use == "human":
        loader_list = [train_loader, test_loader]
    else:
        loader_list = [test_loader]

    with torch.no_grad():
        for data_loader in loader_list:
            logger = Logger(typing_data, gaze_data)
            running_loss = 0.0
            running_gaze_loss = 0.0
            running_gaze_count_loss = 0.0
            running_gaze_distance_loss = 0.0
            running_proofreading_duration_loss = 0.0
            running_proofreading_count_loss = 0.0
            print("Testing in {} data".format("training" if data_loader == train_loader else "testing"))
            for inputs, targets, masks_x, masks_y, index in tqdm(data_loader):
                try:
                    # get the max trailtime in typing_data
                    current_typing_data = typing_data[typing_data['index'] == index[0]]
                    max_trailtime = current_typing_data['trailtime'].max()
                    inputs, targets, masks_x, masks_y = inputs.to(device), targets.to(device), masks_x.to(
                        device), masks_y.to(device)
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

                    # Calculate similarity metrics
                    overall_similarity, pos_sim, dur_sim, shape_sim, direction_sim, length_sim = multi_match(
                        valid_gaze_outputs, valid_targets,
                        return_components=True,
                        include_duration=include_duration)
                    dtwd_dist = dtw_scanpaths(valid_gaze_outputs, valid_targets)
                    sted_dist = sted_scanpaths(valid_gaze_outputs, valid_targets)
                    logger.record_scanpath_level_metrics(index, pos_sim, dur_sim, shape_sim, direction_sim,
                                                         length_sim,
                                                         overall_similarity, dtwd_dist, sted_dist)
                    if str(index[0]) in log_index:
                        print(f'Index: {index[0]}')
                        print(f'Position Similarity: {pos_sim}')
                        print(f'Duration Similarity: {dur_sim}')
                        print(f'Shape Similarity: {shape_sim}')
                        print(f'Direction Similarity: {direction_sim}')
                        print(f'Length Similarity: {length_sim}')
                        print(f'Overall Similarity: {overall_similarity}')
                        print(f'Average Position-Duration Similarity: {(pos_sim + dur_sim) / 2}')
                        print(f'DTWD Score: {dtwd_dist}')
                        print(f'STED Score: {sted_dist}')

                    # Compute loss with masking
                    if loss_type == 'mse':
                        gaze_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                        gaze_loss = gaze_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                        gaze_loss = gaze_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                        padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                        padding_loss = padding_loss.sum() / masks_y.sum()  # Average only over non-padded elements

                        loss = gaze_loss + padding_loss  # Combine the two losses
                    elif loss_type == 'combined':
                        padding_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            padding_outputs[:max_pred_len], masks_y[:max_pred_len].float())
                        padding_loss = padding_loss.sum() / masks_y.sum()
                        typing_gaze_distance_loss, typing_gaze_count_loss = finger_guiding_distance_loss(
                            gaze_outputs[1:max_pred_len],
                            targets[:max_pred_len],
                            inputs,
                            scaler_X,
                            scaler_y,
                            masks_x,
                            masks_y[1:max_pred_len],
                            padding_outputs[:max_pred_len],
                            max_pred_len=max_pred_len,
                        )
                        proofreading_duration_loss, proofreading_count_loss = proofreading_loss(
                            gaze_outputs[1:max_pred_len],
                            targets[:max_pred_len],
                            scaler_y,
                            masks_y[1:max_pred_len],
                            padding_outputs[:max_pred_len],
                            max_pred_len=max_pred_len,
                        )
                        gaze_multimatch_loss, _, _, _, _, _ = multi_match_loss(gaze_outputs[:max_pred_len],
                                                                               targets[:max_pred_len],
                                                                               scaler_y=scaler_y,
                                                                               masks_y=masks_y,
                                                                               predict_masks_y=padding_outputs,
                                                                               pos_weight=0.5,
                                                                               dur_weight=0.5)
                        gaze_mse_loss = mse_loss(gaze_outputs[:max_pred_len], targets[:max_pred_len])
                        gaze_mse_loss = gaze_mse_loss * masks_y.unsqueeze(-1)  # Apply mask to gaze loss
                        gaze_mse_loss = gaze_mse_loss.sum() / masks_y.sum()  # Average only over non-padded elements
                        gaze_loss = gaze_multimatch_loss + gaze_mse_loss
                        loss = gaze_multimatch_loss + padding_loss + gaze_mse_loss + \
                               typing_gaze_distance_loss + typing_gaze_count_loss + \
                               proofreading_duration_loss + proofreading_count_loss

                        running_gaze_loss += gaze_loss.item()
                        running_gaze_count_loss += typing_gaze_count_loss.item()
                        running_gaze_distance_loss += typing_gaze_distance_loss.item()
                        running_proofreading_duration_loss += proofreading_duration_loss.item()
                        running_proofreading_count_loss += proofreading_count_loss.item()
                    else:
                        raise ValueError("Unsupported loss function. Choose 'mse' or 'combined")

                    logger.record_loss_item(loss.item(), gaze_loss.item(), padding_loss.item())
                    logger.build_dataframes(valid_gaze_outputs, valid_targets, index)
                except:
                    print(f'Error in index {index[0]}')
                    continue
            avg_pos_sim, avg_dur_sim, avg_overall_sim = logger.print_inferred_data_result()
            print("Average gaze count loss: ", running_gaze_count_loss / len(data_loader))
            print("Average gaze distance loss: ", running_gaze_distance_loss / len(data_loader))
            print("Average proofreading duration loss: ", running_proofreading_duration_loss / len(data_loader))
            print("Average proofreading count loss: ", running_proofreading_count_loss / len(data_loader))
    # get the ground truth gaze data metrics
    # np array concat test and trian
    indices = np.concatenate([indices_test])
    logger.print_ground_truth_metrics(indices=indices)
    logger.print_best_result()

    return avg_pos_sim, avg_dur_sim, avg_overall_sim


def main():
    parser = argparse.ArgumentParser(description="Train or Test Baseline Model for Gaze Prediction")

    parser.add_argument("--model_type", type=str, choices=['transformer'],
                        default='transformer', help="Type of model to train/test")
    parser.add_argument("--train", action="store_true", help="Train the model", default=False)
    parser.add_argument("--test", action="store_true", help="Test the model", default=False)
    parser.add_argument("--k_folds", type=int, default=12, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--max_pred_len", type=int, default=32, help="Maximum number of gaze data points to predict")
    parser.add_argument("--use_k_fold", action="store_true", help="Use k-fold cross-validation", default=True)
    parser.add_argument("--num_epochs", type=int, default=6000, help="Number of epochs to train the model")
    parser.add_argument("--all", action="store_true", help="Train and test all the model", default=False)
    parser.add_argument("--loss_type", type=str, choices=['mse', 'combined'], default='combined',
                        help="Loss function to use for training")
    parser.add_argument("--data_use", type=str, choices=['both', 'human', 'simulated'], default='human',
                        help="Use human data, simulated data, or both")
    parser.add_argument("--fpath_header", type=str, default='final_distribute', help='File path header for data use')
    parser.add_argument("--continue-training", action="store_true", help="Continue Train the model", default=True)
    parser.add_argument("--start_epoch", type=int, help="Starting epoch for continue training", default=0)
    parser.add_argument("--pretrain-padding", action="store_true", help="Pretrain the Padding", default=False)
    parser.add_argument("--pretrain-epochs", type=int, help="pretraining epoch for padding training", default=200)
    parser.add_argument("--use-best-model", action="store_true", help="Use the best model for testing", default=True)

    args = parser.parse_args()

    # Set up logging only if training is True
    if args.train:
        log_dir = os.path.join(GAZE_INFERENCE_DIR, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        log_filepath = os.path.join(log_dir, log_filename)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
        )
        logging.info("Logging is set up.")

    # Set up TensorBoard writer for training, but not if only testing
    writer = None
    if args.train:
        tensorboard_dir = os.path.join(GAZE_INFERENCE_DIR, 'tensorboard')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)

    if args.train:
        logging.info("Training Baseline Model")
        logging.info("current model type: {}".format(args.model_type))
        logging.info("current loss type: {}".format(args.loss_type))
        logging.info("current data source: {}".format(args.data_use))
        logging.info("use best model: {}".format(args.use_best_model))

    if args.train:
        args.use_best_model = False

    if args.train:
        train_model(
            args.model_type,
            k_folds=args.k_folds,
            max_pred_len=args.max_pred_len,
            use_k_fold=args.use_k_fold,
            num_epochs=args.num_epochs,
            loss_type=args.loss_type,
            data_use=args.data_use,
            continue_training=args.continue_training,
            start_epoch=args.start_epoch,
            pretrain_padding=args.pretrain_padding,
            pretrain_epochs=args.pretrain_epochs,
            fpath_header=args.fpath_header,
            writer=writer,
        )
    if args.test:
        test_model(
            args.model_type,
            max_pred_len=args.max_pred_len,
            loss_type=args.loss_type,
            log_index=[],
            data_use=args.data_use,
            fpath_header=args.fpath_header,
            use_best_model=args.use_best_model,
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
