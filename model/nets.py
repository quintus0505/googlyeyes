import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import timm
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class AmortizedInferenceMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(AmortizedInferenceMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        return out


class TypingGazeDataset(Dataset):
    def __init__(self, X, y, masks_x, masks_y, indices):
        self.X = X
        self.y = y
        self.masks_x = masks_x
        self.masks_y = masks_y
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.masks_x[idx], dtype=torch.float32),
                torch.tensor(self.masks_y[idx], dtype=torch.float32),
                self.indices[idx])


class TypingGazeInferenceDataset(Dataset):
    def __init__(self, X, y, masks_x, masks_y, indices, user_params):
        self.X = X
        self.y = y
        self.masks_x = masks_x
        self.masks_y = masks_y
        self.indices = indices
        self.user_params = user_params  # Add user parameters

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.masks_x[idx], dtype=torch.float32),
                torch.tensor(self.masks_y[idx], dtype=torch.float32),
                self.indices[idx],
                torch.tensor(self.user_params[idx], dtype=torch.float32))  # Return user parameters


class AmortizedInferenceLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout=0.1):
        super(AmortizedInferenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.padding_fc = nn.Linear(hidden_dim, 1)  # Predict padding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)

        gaze_output = self.fc(x)  # Gaze predictions
        # make sure the padding prediction is positive
        padding_output = self.padding_fc(x).squeeze(-1)  # Padding predictions
        # make the padding 0 or 1
        padding_output = torch.sigmoid(padding_output)

        return gaze_output, padding_output


class TransformerInferenceModelV1(nn.Module):
    def __init__(self, input_dim, output_dim, user_param_dim=3, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerInferenceModelV1, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.user_param_encoder = nn.Linear(user_param_dim, d_model)
        self.decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)

    def forward(self, src, user_params, src_mask=None):
        src_encoded = self.encoder(src)  # Shape: (batch_size, seq_len, d_model)

        # Encode user parameters
        user_params_encoded = self.user_param_encoder(user_params)

        # # Expand user_params_encoded to match the sequence length
        # user_params_encoded = user_params_encoded.repeat(1, src_encoded.size(1),
        #                                                  1)  # Shape: (batch_size, seq_len, d_model)

        # Combine the encoded user parameters with the source encoding
        src_combined = src_encoded + user_params_encoded

        src_combined = self.dropout_encoder(src_combined)
        src_combined = src_combined.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model]

        # Pass through the Transformer
        output = self.transformer(src_combined, src_combined, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)  # Change shape back to [batch_size, seq_len, d_model]
        output = self.dropout_transformer(output)

        gaze_output = self.decoder(output)  # Gaze predictions
        padding_output = self.padding_decoder(output).squeeze(-1)  # Padding predictions
        padding_output = torch.sigmoid(padding_output)  # Ensure the padding output is between 0 and 1
        return gaze_output, padding_output


class TransformerInferenceModelV2(nn.Module):
    def __init__(self, input_dim, output_dim, user_param_dim=3, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerInferenceModelV2, self).__init__()
        # Encoder for the input features
        self.encoder = nn.Linear(input_dim, d_model)
        # Encoder for the user parameters
        self.user_param_encoder = nn.Linear(user_param_dim, d_model)
        # Adjusted d_model size for concatenated input
        concatenated_dim = d_model * 2
        # Transformer and Decoder now take the concatenated dimension as input
        self.transformer = nn.Transformer(concatenated_dim, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward,
                                          dropout=dropout)
        self.decoder = nn.Linear(concatenated_dim, output_dim)
        self.padding_decoder = nn.Linear(concatenated_dim, 1)  # Predict padding

        # Dropout layers
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)

    def forward(self, src, user_params, src_mask=None):
        # Encode the source input
        src_encoded = self.encoder(src)  # Shape: (batch_size, seq_len, d_model)
        # Encode the user parameters
        user_params_encoded = self.user_param_encoder(user_params)  # Shape: (batch_size, seq_len, d_model)

        # Concatenate the encoded input and user parameters
        src_combined = torch.cat((src_encoded, user_params_encoded), dim=-1)  # Shape: (batch_size, seq_len, 2*d_model)

        # Apply dropout
        src_combined = self.dropout_encoder(src_combined)
        # Prepare for transformer input: [seq_len, batch_size, 2*d_model]
        src_combined = src_combined.permute(1, 0, 2)

        # Pass through the Transformer
        output = self.transformer(src_combined, src_combined, src_key_padding_mask=src_mask)
        # Convert back to [batch_size, seq_len, 2*d_model]
        output = output.permute(1, 0, 2)
        # Apply dropout after transformer
        output = self.dropout_transformer(output)

        # Decode the output
        gaze_output = self.decoder(output)  # Shape: (batch_size, seq_len, output_dim)
        padding_output = self.padding_decoder(output).squeeze(-1)  # Shape: (batch_size, seq_len)
        padding_output = torch.sigmoid(padding_output)  # Ensure the padding output is between 0 and 1
        return gaze_output, padding_output


class TransformerInferenceModelV3(nn.Module):
    def __init__(self, input_dim, output_dim, user_param_dim=3, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerInferenceModelV3, self).__init__()
        # Encoder for the input features
        self.encoder = nn.Linear(input_dim, d_model)
        # Encoder for the user parameters
        self.user_param_encoder = nn.Linear(user_param_dim, d_model)
        # Transformer for processing the sequence
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)
        # Decoder layers for gaze and padding predictions
        self.decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding

        # Dropout layers
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, src, user_params, src_mask=None):
        # Encode the source input (x, y, duration)
        src_encoded = self.encoder(src)  # Shape: (batch_size, seq_len, d_model)

        # Encode the user parameters and use only the first one for attention
        user_params_encoded = self.user_param_encoder(user_params[:, 0, :])  # Shape: (batch_size, d_model)
        user_params_encoded = user_params_encoded.unsqueeze(1)  # Shape: (batch_size, 1, d_model)

        # Apply dropout to the encoded input
        src_encoded = self.dropout_encoder(src_encoded)
        # Prepare for transformer input: [seq_len, batch_size, d_model]
        src_encoded = src_encoded.permute(1, 0, 2)

        # Attention mechanism (use user_params_encoded as query, and src_encoded as key and value)
        attn_output, _ = self.attn(user_params_encoded.permute(1, 0, 2),  # Query: (1, batch_size, d_model)
                                   src_encoded,  # Key: (seq_len, batch_size, d_model)
                                   src_encoded,  # Value: (seq_len, batch_size, d_model)
                                   key_padding_mask=src_mask)  # Apply padding mask if provided

        # Concatenate the attention output with the original input features
        src_combined = attn_output.permute(1, 0, 2) + src_encoded.permute(1, 0,
                                                                          2)  # Shape: (batch_size, seq_len, d_model)

        # Pass through the transformer
        src_combined = src_combined.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
        output = self.transformer(src_combined, src_combined, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, d_model)
        output = self.dropout_transformer(output)

        # Decode the output
        gaze_output = self.decoder(output)  # Shape: (batch_size, seq_len, output_dim)
        padding_output = self.padding_decoder(output).squeeze(-1)  # Shape: (batch_size, seq_len)
        padding_output = torch.sigmoid(padding_output)  # Ensure the padding output is between 0 and 1
        return gaze_output, padding_output


class TransformerInferenceModel(nn.Module):
    def __init__(self, input_dim, output_dim, user_param_dim=3, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerInferenceModel, self).__init__()
        # Encoder for the input features
        self.encoder = nn.Linear(input_dim, d_model)
        # Encoder for the user parameters
        self.user_param_encoder = nn.Linear(user_param_dim, d_model)
        # Transformer for processing the sequence
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)
        # Decoder layers for gaze and padding predictions
        self.output_mean_decoder = nn.Linear(d_model, output_dim)
        self.output_log_std_decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding

        # Dropout layers
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, src, user_params, src_mask=None):
        # Encode the source input (x, y, duration)
        src_encoded = self.encoder(src)  # Shape: (batch_size, seq_len, d_model)
        src_encoded = self.dropout_encoder(src_encoded)
        src_encoded = src_encoded.permute(1, 0, 2)

        # Encode the user parameters and use only the first one for attention
        user_params_encoded = self.user_param_encoder(user_params[:, 0, :])  # Shape: (batch_size, d_model)
        user_params_encoded = user_params_encoded.unsqueeze(1)  # Shape: (batch_size, 1, d_model)
        user_params_encoded = user_params_encoded.permute(1, 0, 2)  # Shape: (1, batch_size, d_model)

        # Attention mechanism (use user_params_encoded as query, and src_encoded as key and value)
        attn_output, _ = self.attn(user_params_encoded,  # Query: (1, batch_size, d_model)
                                   src_encoded,  # Key: (seq_len, batch_size, d_model)
                                   src_encoded,  # Value: (seq_len, batch_size, d_model)
                                   key_padding_mask=src_mask)  # Apply padding mask if provided

        # Concatenate the attention output with the original input features
        src_combined = attn_output.permute(1, 0, 2) + src_encoded.permute(1, 0,
                                                                          2)  # Shape: (batch_size, seq_len, d_model)

        # Pass through the transformer
        src_combined = src_combined.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
        output = self.transformer(src_combined, src_encoded, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, d_model)
        output = self.dropout_transformer(output)

        # Decode the output
        gaze_mean = self.output_mean_decoder(output)
        gaze_log_std = self.output_log_std_decoder(output)
        padding_output = self.padding_decoder(output).squeeze(-1)  # Shape: (batch_size, seq_len)
        padding_output = torch.sigmoid(padding_output)  # Ensure the padding output is between 0 and 1
        return gaze_mean, gaze_log_std, padding_output


class TransformerAutoRegressiveModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1):
        super(TransformerAutoRegressiveModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)
        self.d_model = d_model
        self.output_dim = output_dim

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, max_len=32, is_training=False,
                teacher_forcing_ratio=1.0):
        # Encode the input typing log
        src = self.encoder(src)
        src = self.dropout_encoder(src)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        memory = self.transformer.encoder(src, src_key_padding_mask=src_mask)

        # Non-autoregressive padding prediction based on the full input (src)
        # This is similar to your non-autoregressive TransformerModel
        padding_output = self.padding_decoder(memory.permute(1, 0, 2)).squeeze(-1)  # [batch_size, seq_len]
        padding_output = torch.sigmoid(padding_output)  # Sigmoid to normalize between 0 and 1 (0 or 1 prediction)

        gaze_outputs = []

        if is_training and tgt is not None:
            tgt_input = self.encoder(tgt)
            tgt_input = self.dropout_encoder(tgt_input)
            tgt_input = tgt_input.permute(1, 0, 2)  # [seq_len, batch_size, d_model]

            # Decoding loop for autoregressive gaze prediction
            for i in range(max_len):
                if i == 0:
                    # For the first step, we always use the ground truth
                    current_input = tgt_input[i].unsqueeze(0)
                else:
                    # Decide whether to use the ground truth or the predicted output
                    if np.random.rand() < teacher_forcing_ratio:
                        current_input = tgt_input[i].unsqueeze(0)
                    else:
                        current_input = tgt_output[-1].unsqueeze(0)

                tgt_output = self.transformer.decoder(current_input, memory, memory_key_padding_mask=src_mask,
                                                      tgt_key_padding_mask=tgt_mask)
                tgt_output = self.dropout_transformer(tgt_output)

                gaze_output = self.decoder(tgt_output[-1])
                gaze_outputs.append(gaze_output)

            gaze_outputs = torch.stack(gaze_outputs, dim=1)  # [batch_size, seq_len, output_dim]

        else:
            # Autoregressive inference (autoregressively predict the gaze sequence)
            tgt_input = src[0].unsqueeze(0)  # Start with the first input from the source sequence
            for i in range(max_len):
                tgt_output = self.transformer.decoder(tgt_input, memory, memory_key_padding_mask=src_mask,
                                                      tgt_key_padding_mask=tgt_mask)
                tgt_output = self.dropout_transformer(tgt_output)

                gaze_output = self.decoder(tgt_output[-1])
                gaze_outputs.append(gaze_output)

                # Update the input with the new gaze point prediction for the next step
                tgt_input = torch.cat([tgt_input, tgt_output[-1].unsqueeze(0)], dim=0)

            gaze_outputs = torch.stack(gaze_outputs, dim=1)  # [batch_size, seq_len, output_dim]

        return gaze_outputs, padding_output


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.output_mean_decoder = nn.Linear(d_model, output_dim)
        self.output_log_std_decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src = self.encoder(src)
        src = self.dropout_encoder(src)
        src = src.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model]
        output = self.transformer(src, src, src_key_padding_mask=src_mask)
        output = output.permute(1, 0, 2)  # Change shape back to [batch_size, seq_len, d_model]
        output = self.dropout_transformer(output)

        gaze_mean = self.output_mean_decoder(output)
        gaze_log_std = self.output_log_std_decoder(output)
        padding_output = self.padding_decoder(output).squeeze(-1)  # Padding predictions
        # padding_output = torch.clamp(padding_output, min=0)
        # make the padding 0 or 1
        padding_output = torch.sigmoid(padding_output)
        return gaze_mean, gaze_log_std, padding_output


class TransformerModelTeacherForcing(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1):
        super(TransformerModelTeacherForcing, self).__init__()

        # Linear layers for projecting continuous features (x, y, duration) to d_model
        self.encoder = nn.Linear(input_dim, d_model)
        self.decoder = nn.Linear(d_model, output_dim)
        self.encoder_activation = nn.Tanh()
        self.decoder_activation = nn.Tanh()
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding

        # Transformer for sequence modeling
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.dropout_encoder = nn.Dropout(dropout)
        self.dropout_transformer = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Apply the linear projection to map (x, y, duration) to d_model
        # src = self.dropout_encoder(self.encoder(src))
        # tgt = self.dropout_encoder(self.encoder(tgt))

        src = self.encoder(src)
        tgt = self.encoder(tgt)
        src = self.encoder_activation(src)
        tgt = self.encoder_activation(tgt)
        src = self.dropout_encoder(src)
        tgt = self.dropout_encoder(tgt)

        # Permute to [seq_len, batch_size, d_model] for transformer
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Pass through the transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, d_model]

        # Predict gaze (x, y, duration) output and padding
        gaze_output = self.decoder(output)  # Predict new gaze coordinates and duration
        padding_output = self.padding_decoder(output).squeeze(-1)  # Predict padding
        padding_output = torch.sigmoid(padding_output)

        return gaze_output, padding_output


class TransformerModelTeacherForcingRL(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, input_dropout=0.15, output_dropout=0.15, transformer_dropout=0.1):
        super(TransformerModelTeacherForcingRL, self).__init__()

        # Linear layers for projecting continuous features (x, y, duration) to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(input_dim, d_model)

        # Activation function for projections
        self.activation = nn.Tanh()  # You can use ReLU, Tanh, etc.

        # Transformer for sequence modeling
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=transformer_dropout)

        self.input_dropout = nn.Dropout(input_dropout)  # Dropout for the input projection
        self.output_dropout = nn.Dropout(output_dropout)  # Dropout for the output layer

        # Output layers for mean and log_std of gaze positions
        self.output_mean = nn.Linear(d_model, output_dim)
        self.output_log_std = nn.Linear(d_model, output_dim)

        # Output layer for padding prediction
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding

        # Activation for padding prediction
        # self.padding_activation = nn.Sigmoid()  # If padding prediction is binary (0 or 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Apply the linear projection to map (x, y, duration) to d_model
        src = self.input_dropout(self.activation(self.input_projection(src)))
        tgt = self.output_dropout(self.activation(self.output_projection(tgt)))

        # Permute to [seq_len, batch_size, d_model] for transformer
        src = src.permute(1, 0, 2)  # [seq_len_src, batch_size, d_model]
        tgt = tgt.permute(1, 0, 2)  # [seq_len_tgt, batch_size, d_model]

        # Pass through the transformer
        transformer_output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                              src_key_padding_mask=src_padding_mask,
                                              tgt_key_padding_mask=tgt_padding_mask)

        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, seq_len, d_model]

        # Apply output dropout before predicting gaze positions
        transformer_output = self.output_dropout(transformer_output)

        # Output mean and log_std for gaze positions
        gaze_mean = self.output_mean(transformer_output)
        gaze_log_std = self.output_log_std(transformer_output)

        # Predict padding
        padding_output = self.padding_decoder(transformer_output).squeeze(-1)
        return gaze_mean, gaze_log_std, padding_output


class InverseEyeTracker(nn.Module):
    def __init__(self, input_dim, output_dim, human_param_dim=3, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1):
        super(InverseEyeTracker, self).__init__()
        self.human_param_dim = human_param_dim
        self.d_model = d_model

        # Linear layers for processing input sequence and human parameters
        self.encoder = nn.Linear(input_dim, d_model)
        self.human_param_encoder = nn.Linear(human_param_dim, d_model)

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Transformer layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=dropout)

        # Decoders for gaze prediction and padding prediction
        self.decoder = nn.Linear(d_model, output_dim)
        self.padding_decoder = nn.Linear(d_model, 1)  # Predict padding
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, human_params, src_mask=None):
        # src: (batch_size, seq_len, input_dim)
        # human_params: (batch_size, human_param_dim)

        # Encode the input sequence and human parameters
        src = self.encoder(src)  # (batch_size, seq_len, d_model)
        human_params = self.human_param_encoder(human_params)  # (batch_size, d_model)

        # Repeat human parameters for attention across all timesteps
        human_params = human_params.unsqueeze(1).repeat(1, src.size(1), 1)  # (batch_size, seq_len, d_model)
        human_params = human_params.permute(1, 0, 2)  # (seq_len, batch_size, d_model)

        # Apply attention: query is the input sequence, key/value are the human parameters
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model) for attention
        attn_output, _ = self.attention(src, human_params, human_params, key_padding_mask=src_mask)

        # Add the attention output back to the original input sequence representation
        src = src + attn_output  # (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # Pass through transformer layers
        output = self.transformer(src, src, src_key_padding_mask=src_mask)
        output = self.dropout(output)

        # Decode the output
        gaze_output = self.decoder(output)  # Gaze predictions
        padding_output = self.padding_decoder(output).squeeze(-1)  # Padding predictions
        padding_output = torch.clamp(padding_output, min=0)

        return gaze_output, padding_output


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.padding_fc = nn.Linear(hidden_dim, 1)  # Predict padding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)

        gaze_output = self.fc(x)  # Gaze predictions
        # make sure the padding prediction is positive
        padding_output = self.padding_fc(x).squeeze(-1)  # Padding predictions
        # make the padding 0 or 1
        padding_output = torch.sigmoid(padding_output)

        return gaze_output, padding_output


class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout=0.1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeiTModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=32, model_name='vit_tiny_patch16_224', pretrained=True):
        super(DeiTModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=3, kernel_size=1)  # Convert to 3 channels
        self.deit = timm.create_model(model_name, pretrained=pretrained)
        self.deit.head = nn.Linear(self.deit.head.in_features, output_dim * seq_len)  # Adjust output dimension
        self.seq_len = seq_len
        self.output_dim = output_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, feature_dim, seq_len]
        x = self.conv1d(x)  # Apply 1D convolution to get [batch_size, 3, seq_len]
        x = x.unsqueeze(-1)  # Add height dimension to get [batch_size, 3, seq_len, 1]
        x = F.interpolate(x, size=(224, 224), mode='bilinear',
                          align_corners=False)  # Resize to [batch_size, 3, 224, 224]
        x = self.deit(x)  # Feed into DeiT model
        x = x.view(x.size(0), self.seq_len, self.output_dim)  # Reshape to [batch_size, seq_len, output_dim]
        return x


class T5ModelWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=32):
        super(T5ModelWrapper, self).__init__()
        self.input_proj = nn.Linear(input_dim, 512)  # Project input_dim to T5's expected input dimension
        self.t5 = T5EncoderModel(T5Config(d_model=512, num_layers=6))
        self.output_proj = nn.Linear(512, output_dim)  # Project T5's output dimension to the expected output_dim
        self.seq_len = seq_len
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x)  # Project input to T5's expected input dimension
        x = self.t5(inputs_embeds=x).last_hidden_state  # Use T5 encoder
        x = self.output_proj(x)  # Project to the expected output_dim
        return x  # Output shape should be [batch_size, seq_len, output_dim]


# Create a mask for target sequences (to prevent seeing future tokens)
def create_target_mask(size):
    mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Create padding mask
def create_padding_mask(seq, pad_token):
    return (seq == pad_token).transpose(0, 1)


def _compute_position_similarity(X, Y):
    X = X / torch.max(X)
    Y = Y / torch.max(Y)
    # X = X / min(torch.max(X), torch.tensor(1080))
    # Y = Y / min(torch.max(Y), torch.tensor(1920))
    distance_matrix = torch.cdist(X, Y, p=2)

    # Approximate linear sum assignment using differentiable softmin
    row_ind = torch.argmin(distance_matrix, dim=1)
    col_ind = torch.argmin(distance_matrix, dim=0)

    similarity = torch.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


def _compute_trailtime_similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    max_trailtime = max(X.max(), Y.max())
    X = X / max_trailtime
    Y = Y / max_trailtime
    distance_matrix = torch.abs(X[:, None] - Y[None, :])

    # Approximate linear sum assignment using differentiable softmin
    row_ind = torch.argmin(distance_matrix, dim=1)
    col_ind = torch.argmin(distance_matrix, dim=0)

    similarity = torch.exp(-distance_matrix[row_ind, col_ind].mean())
    return similarity


def _compute_shape_similarity(X, Y):
    X_vectors = X[1:] - X[:-1]
    Y_vectors = Y[1:] - Y[:-1]
    X_angles = torch.atan2(X_vectors[:, 1], X_vectors[:, 0])
    Y_angles = torch.atan2(Y_vectors[:, 1], Y_vectors[:, 0])
    angle_diff = torch.abs(X_angles[:, None] - Y_angles[None, :])
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)

    # Approximate linear sum assignment using differentiable softmin
    row_ind = torch.argmin(angle_diff, dim=1)
    col_ind = torch.argmin(angle_diff, dim=0)

    similarity = torch.exp(-angle_diff[row_ind, col_ind].mean())
    return similarity


def _compute_direction_similarity(X, Y):
    X_direction = X[-1] - X[0]
    Y_direction = Y[-1] - Y[0]
    X_norm = X_direction / torch.norm(X_direction)
    Y_norm = Y_direction / torch.norm(Y_direction)
    dot_product = torch.dot(X_norm, Y_norm)
    similarity = (dot_product + 1) / 2  # scale to [0, 1]
    return similarity


def _compute_length_similarity(X, Y):
    X_lengths = torch.norm(X[1:] - X[:-1], dim=1)
    Y_lengths = torch.norm(Y[1:] - Y[:-1], dim=1)
    X_total = X_lengths.sum()
    Y_total = Y_lengths.sum()
    length_diff = torch.abs(X_total - Y_total)
    max_length = torch.max(X_total, Y_total)
    similarity = torch.exp(-length_diff / max_length)
    return similarity


def multi_match(X, Y, dimensions=2, return_components=False, include_duration=True, device='cpu', pos_weight=1.0,
                dur_weight=0.5, shape_weight=0.5, dir_weight=0.5, length_weight=0.5):
    if dimensions not in [2, 3]:
        raise ValueError('Invalid number of dimensions')

    Xpos = X[:, :dimensions].to(device)
    Ypos = Y[:, :dimensions].to(device)
    if include_duration:
        Xdur = X[:, dimensions:dimensions + 1].to(device)
        Ydur = Y[:, dimensions:dimensions + 1].to(device)

    pos_sim = _compute_position_similarity(Xpos, Ypos)
    if include_duration:
        dur_sim = _compute_trailtime_similarity(Xdur, Ydur)
    else:
        dur_sim = torch.tensor(0.0, device=device)

    # shape_sim = _compute_shape_similarity(Xpos, Ypos)
    # dir_sim = _compute_direction_similarity(Xpos, Ypos)
    length_sim = _compute_length_similarity(Xpos, Ypos)
    shape_sim = torch.tensor(0.0, device=device)
    dir_sim = torch.tensor(0.0, device=device)
    # length_sim = torch.tensor(0.0, device=device)

    # total_sim = ((pos_weight * pos_sim) + (dur_weight * dur_sim) + (shape_weight * shape_sim) + (
    #         dir_weight * dir_sim) + (
    #                      length_weight * length_sim)) / (
    #                     pos_weight + dur_weight + shape_weight + dir_weight + length_weight)
    total_sim = (pos_weight * pos_sim + dur_weight * dur_sim + length_sim * length_weight) / (
            pos_weight + dur_weight + length_weight)
    if return_components:
        return total_sim, pos_sim, dur_sim, shape_sim, dir_sim, length_sim
    return total_sim


def time_duration_penalty(outputs, threshold=5000, penalty_weight=0.01):
    # Apply ReLU to penalize only values greater than the threshold
    penalty = torch.relu(outputs[:, -1] - threshold)
    regularization_loss = penalty.mean() * penalty_weight
    # # Also penalize the negative values
    # penalty = torch.relu(-outputs[:, -1])
    # also penalize the values less than 100
    penalty = torch.relu(100 - outputs[:, -1])
    regularization_loss += penalty.mean() * penalty_weight
    return regularization_loss


def position_penalty(outputs, penalty_weight=0.01):
    # X max 1080, Y max 1920
    penalty = torch.relu(outputs[:, 0] - 1080) + torch.relu(outputs[:, 1] - 1920)
    regularization_loss = penalty.mean() * penalty_weight
    penalty = torch.relu(-outputs[:, 0]) + torch.relu(-outputs[:, 1])
    regularization_loss += penalty.mean() * penalty_weight
    return regularization_loss


def proofreading_loss(outputs, targets, scaler_y, masks_y, predict_masks_y, max_pred_len=32,
                      y_limit=350, duration_loss_ratio=1, count_loss_ratio=0.8,
                      is_debug=False):
    """
    Computes the count and duration losses for proofreading behavior.

    Args:
        outputs (Tensor): Predicted gaze data, shape (batch_size, seq_len, feature_dim).
        targets (Tensor): Ground truth gaze data, shape (batch_size, seq_len, feature_dim).
        scaler_y (scaler): Scaler used to normalize gaze data.
        masks_y (Tensor): Mask for gaze data, shape (batch_size, seq_len).
        predict_masks_y (Tensor): Predicted mask for gaze data, shape (batch_size, seq_len).
        max_pred_len (int): Maximum sequence length to consider.
        y_limit (float): Threshold for the y-coordinate of gaze positions (proofreading threshold).
        duration_loss_ratio (float): Weight for the duration loss.
        count_loss_ratio (float): Weight for the count loss.
        is_debug (bool): Whether to print debug information.

    Returns:
        total_duration_loss (Tensor): Averaged duration loss over the batch.
        total_count_loss (Tensor): Averaged count loss over the batch.
    """
    # Ensure tensors are 3D for batch processing
    duration_loss_normalizer = 1e-4
    count_loss_normalizer = 1 / max_pred_len
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        masks_y = masks_y.unsqueeze(0)
        predict_masks_y = predict_masks_y.unsqueeze(0)
    batch_size = outputs.shape[0]
    predict_masks_y = torch.round(predict_masks_y)

    # Extract the scaler parameters
    scaler_y_mean = torch.tensor(scaler_y.mean_, device=outputs.device, dtype=outputs.dtype)
    scaler_y_scale = torch.tensor(scaler_y.scale_, device=outputs.device, dtype=outputs.dtype)
    total_duration_loss = 0.0
    total_count_loss = 0.0
    for i in range(batch_size):
        outputs_i = outputs[i]
        targets_i = targets[i]
        masks_y_i = masks_y[i]
        predict_masks_y_i = predict_masks_y[i]

        # Perform the inverse scaling using PyTorch operations
        outputs_inv = outputs_i * scaler_y_scale + scaler_y_mean
        targets_inv = targets_i * scaler_y_scale + scaler_y_mean

        # Select valid data based on masks
        valid_outputs = outputs_inv[:max_pred_len]
        valid_targets = targets_inv[:max_pred_len]
        valid_masks_y = masks_y_i[:max_pred_len]
        valid_predict_masks_y = predict_masks_y_i[:max_pred_len]
        valid_targets = valid_targets[1:] if valid_targets.size(0) > 1 else valid_targets

        # Apply masks to select valid entries
        valid_outputs = valid_outputs[valid_masks_y == 1]
        valid_targets = valid_targets[valid_masks_y == 1]

        # Check if we have valid data to proceed
        if valid_outputs.size(0) == 0 or valid_targets.size(0) == 0:
            continue  # Skip this iteration if there's no valid data

        # Extract y-coordinates and durations
        gaze_y_pred = valid_outputs[:, 1]  # y-coordinate
        durations_pred = valid_outputs[:, 2]  # durations

        gaze_y_gt = valid_targets[:, 1]  # y-coordinate
        durations_gt = valid_targets[:, 2]  # durations

        # Use soft masks for proofreading points (y < y_limit)
        s_y = 10.0  # Controls softness of y-coordinate mask transitions

        # Compute soft masks
        soft_mask_pred = 1 - torch.sigmoid((gaze_y_pred - y_limit) / s_y)  # (N_pred,)
        soft_mask_gt = 1 - torch.sigmoid((gaze_y_gt - y_limit) / s_y)  # (N_gt,)

        # Compute soft counts and total durations
        soft_count_pred = soft_mask_pred.sum()
        soft_count_gt = soft_mask_gt.sum()

        total_duration_pred = (durations_pred * soft_mask_pred).sum()
        total_duration_gt = (durations_gt * soft_mask_gt).sum()

        # Compute the losses
        count_loss = torch.abs(soft_count_pred - soft_count_gt)
        # TODO: stupid total_duration_gt * 1.2
        duration_loss = torch.abs(total_duration_pred - total_duration_gt)

        # ** Debugging Code Start **
        if is_debug:
            with torch.no_grad():
                # Create hard masks for debugging
                hard_mask_pred = (gaze_y_pred <= y_limit)
                hard_mask_gt = (gaze_y_gt <= y_limit)

                # Counts
                count_pred_hard = hard_mask_pred.sum().item()
                count_gt_hard = hard_mask_gt.sum().item()

                # Total durations
                total_duration_pred_hard = durations_pred[hard_mask_pred].sum().item()
                total_duration_gt_hard = durations_gt[hard_mask_gt].sum().item()

                print(f"Batch item {i}:")
                print(f" Predicted proofreading count (hard): {count_pred_hard}")
                print(f" Ground truth proofreading count (hard): {count_gt_hard}")
                print(f" Predicted proofreading total duration (hard): {total_duration_pred_hard:.2f} ms")
                print(f" Ground truth proofreading total duration (hard): {total_duration_gt_hard:.2f} ms")
                print(f" Soft count (predicted): {soft_count_pred.item():.2f}")
                print(f" Soft count (ground truth): {soft_count_gt.item():.2f}")
                print(f" Total duration (predicted): {total_duration_pred.item():.2f} ms")
                print(f" Total duration (ground truth): {total_duration_gt.item():.2f} ms")

        total_count_loss += count_loss
        total_duration_loss += duration_loss
        # ** Debugging Code End **

    total_duration_loss = total_duration_loss / batch_size
    total_count_loss = total_count_loss / batch_size

    total_duration_loss *= duration_loss_normalizer
    total_count_loss *= count_loss_normalizer

    return total_duration_loss * count_loss_ratio, total_count_loss * duration_loss_ratio


def finger_guiding_distance_loss(outputs, targets, typing_data, scaler_x, scaler_y,
                                 masks_x, masks_y, predict_masks_y, max_pred_len=32,
                                 y_limit=960, time_window=350,
                                 distance_loss_ratio=5, count_loss_ratio=1,
                                 is_debug=False):
    """
    Computes the distance and count losses for proofreading, using soft masks to ensure differentiability.

    Args:
        outputs (Tensor): Predicted gaze data, shape (batch_size, seq_len, feature_dim).
        targets (Tensor): Ground truth gaze data, shape (batch_size, seq_len, feature_dim).
        typing_data (Tensor): Typing data, shape (batch_size, seq_len, feature_dim).
        scaler_x (scaler): Scaler used to normalize typing data.
        scaler_y (scaler): Scaler used to normalize gaze data.
        masks_x (Tensor): Mask for typing data, shape (batch_size, seq_len).
        masks_y (Tensor): Mask for gaze data, shape (batch_size, seq_len).
        predict_masks_y (Tensor): Predicted mask for gaze data, shape (batch_size, seq_len).
        max_pred_len (int): Maximum sequence length to consider.
        y_limit (float): Threshold for the y-coordinate of gaze positions.
        time_window (float): Time window (in milliseconds) for considering gaze points before typing events.
        distance_loss_ratio (float): Weight for the distance loss.
        count_loss_ratio (float): Weight for the count loss.
        is_debug (bool): Whether to print debug information.

    Returns:
        total_distance_loss (Tensor): Averaged distance loss over the batch.
        total_count_loss (Tensor): Averaged count loss over the batch.
    """
    distance_loss_normalizer = 1e-3  # Normalize distance loss to avoid large values
    count_loss_normalizer = 100
    # Ensure tensors are 3D for batch processing
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        masks_y = masks_y.unsqueeze(0)
        masks_x = masks_x.unsqueeze(0)
        predict_masks_y = predict_masks_y.unsqueeze(0)
        typing_data = typing_data.unsqueeze(0)
    batch_size = outputs.shape[0]
    predict_masks_y = torch.round(predict_masks_y)

    # Extract the scaler parameters
    scaler_y_mean = torch.tensor(scaler_y.mean_, device=outputs.device, dtype=outputs.dtype)
    scaler_y_scale = torch.tensor(scaler_y.scale_, device=outputs.device, dtype=outputs.dtype)
    scaler_x_mean = torch.tensor(scaler_x.mean_, device=outputs.device, dtype=outputs.dtype)
    scaler_x_scale = torch.tensor(scaler_x.scale_, device=outputs.device, dtype=outputs.dtype)

    total_distance_loss = 0.0
    total_count_loss = 0.0
    for i in range(batch_size):
        outputs_i = outputs[i]
        targets_i = targets[i]
        masks_y_i = masks_y[i]
        masks_x_i = masks_x[i]
        typing_data_i = typing_data[i]
        predict_masks_y_i = predict_masks_y[i]

        # Perform the inverse scaling using PyTorch operations
        outputs_inv = outputs_i * scaler_y_scale + scaler_y_mean
        targets_inv = targets_i * scaler_y_scale + scaler_y_mean
        typing_inv = typing_data_i * scaler_x_scale + scaler_x_mean

        # Select valid data based on masks
        valid_outputs = outputs_inv[:max_pred_len]
        valid_targets = targets_inv[:max_pred_len]
        valid_masks_y = masks_y_i[:max_pred_len]
        valid_masks_x = masks_x_i[:max_pred_len]
        valid_typing = typing_inv[:max_pred_len]
        valid_predict_masks_y = predict_masks_y_i[:max_pred_len]

        # Adjust for trail time starting point if necessary
        valid_targets_first_trailtime = valid_targets[0, 2] if valid_targets.size(0) > 0 else 0.0
        valid_targets = valid_targets[1:] if valid_targets.size(0) > 1 else valid_targets

        # Apply masks to select valid entries
        valid_outputs = valid_outputs[valid_masks_y == 1]
        valid_targets = valid_targets[valid_masks_y == 1]
        valid_typing = valid_typing[valid_masks_x == 1]

        # Check if we have valid data to proceed
        if valid_outputs.size(0) == 0 or valid_targets.size(0) == 0 or valid_typing.size(0) == 0:
            continue  # Skip this iteration if there's no valid data

        # Compute trail times (cumulative sum of durations)
        typing_trailtime = torch.cumsum(valid_typing[:, 2], dim=0)
        outputs_trailtime = torch.cumsum(valid_outputs[:, 2], dim=0) + valid_targets_first_trailtime
        targets_trailtime = torch.cumsum(valid_targets[:, 2], dim=0) + valid_targets_first_trailtime

        # Extract positions and y-coordinates
        typing_positions = valid_typing[:, :2]  # (N_typing, 2)
        typing_times = typing_trailtime  # (N_typing,)

        gaze_positions_pred = valid_outputs[:, :2]  # (N_gaze_pred, 2)
        gaze_times_pred = outputs_trailtime  # (N_gaze_pred,)
        gaze_y_pred = valid_outputs[:, 1]  # y-coordinate

        gaze_positions_gt = valid_targets[:, :2]  # (N_gaze_gt, 2)
        gaze_times_gt = targets_trailtime  # (N_gaze_gt,)
        gaze_y_gt = valid_targets[:, 1]  # y-coordinate

        # Prepare for broadcasting
        typing_times_expanded = typing_times.unsqueeze(1)  # (N_typing, 1)
        gaze_times_pred_expanded = gaze_times_pred.unsqueeze(0)  # (1, N_gaze_pred)
        gaze_times_gt_expanded = gaze_times_gt.unsqueeze(0)  # (1, N_gaze_gt)

        # Parameters for soft masks
        s_time = 50.0  # Controls softness of time mask transitions
        s_y = 50.0  # Controls softness of y-coordinate mask transitions

        # Compute time differences
        delta_t_pred = typing_times_expanded - gaze_times_pred_expanded  # (N_typing, N_gaze_pred)
        delta_t_gt = typing_times_expanded - gaze_times_gt_expanded  # (N_typing, N_gaze_gt)

        # Compute time-based soft masks
        w_time_pred = torch.sigmoid(- (delta_t_pred - 0) / s_time) * torch.sigmoid(
            (delta_t_pred - time_window) / s_time)
        w_time_gt = torch.sigmoid(- (delta_t_gt - 0) / s_time) * torch.sigmoid((delta_t_gt - time_window) / s_time)

        # Compute y-coordinate-based soft masks
        gaze_y_pred_expanded = gaze_y_pred.unsqueeze(0)  # (1, N_gaze_pred)
        gaze_y_gt_expanded = gaze_y_gt.unsqueeze(0)  # (1, N_gaze_gt)
        w_y_pred = torch.sigmoid((gaze_y_pred_expanded - y_limit) / s_y)
        w_y_gt = torch.sigmoid((gaze_y_gt_expanded - y_limit) / s_y)

        # Combined soft masks
        soft_mask_pred = w_time_pred * w_y_pred  # Shape: (N_typing, N_gaze_pred)
        soft_mask_gt = w_time_gt * w_y_gt  # Shape: (N_typing, N_gaze_gt)

        # ** Debugging Code Start **
        # Create hard masks for debugging (do not use in loss computation)
        if is_debug:
            with torch.no_grad():
                # Hard masks based on the selection conditions
                hard_mask_pred = (
                        (delta_t_pred >= 0) & (delta_t_pred <= time_window) & (gaze_y_pred_expanded >= y_limit))
                hard_mask_gt = ((delta_t_gt >= 0) & (delta_t_gt <= time_window) & (gaze_y_gt_expanded >= y_limit))

                # Convert hard masks to numpy arrays for inspection
                hard_mask_pred_np = hard_mask_pred.cpu().numpy()
                hard_mask_gt_np = hard_mask_gt.cpu().numpy()

                # For each typing point, find the indices of selected gaze points
                selected_indices_pred = [np.where(hard_mask_pred_np[i])[0] for i in range(hard_mask_pred_np.shape[0])]
                selected_indices_gt = [np.where(hard_mask_gt_np[i])[0] for i in range(hard_mask_gt_np.shape[0])]

                # Compute distances between typing positions and gaze positions (for hard masks)
                delta_pos_pred = typing_positions.unsqueeze(1) - gaze_positions_pred.unsqueeze(
                    0)  # (N_typing, N_gaze_pred, 2)
                distances_pred = torch.norm(delta_pos_pred, dim=2)  # (N_typing, N_gaze_pred)
                delta_pos_gt = typing_positions.unsqueeze(1) - gaze_positions_gt.unsqueeze(
                    0)  # (N_typing, N_gaze_gt, 2)
                distances_gt = torch.norm(delta_pos_gt, dim=2)  # (N_typing, N_gaze_gt)

                # For each typing point, compute average distances using hard masks
                avg_distances_pred_list = []
                avg_distances_gt_list = []
                for idx in range(len(typing_times)):
                    pred_indices = selected_indices_pred[idx]
                    gt_indices = selected_indices_gt[idx]

                    # Get distances for selected gaze points
                    distances_pred_selected = distances_pred[idx, pred_indices]
                    distances_gt_selected = distances_gt[idx, gt_indices]

                    # Compute average distances
                    avg_distance_pred = distances_pred_selected.mean().item() if len(
                        distances_pred_selected) > 0 else float('nan')
                    avg_distance_gt = distances_gt_selected.mean().item() if len(distances_gt_selected) > 0 else float(
                        'nan')

                    avg_distances_pred_list.append(avg_distance_pred)
                    avg_distances_gt_list.append(avg_distance_gt)

                # Print or log the selected indices and average distances for debugging
                gt_count = 0
                pred_count = 0
                total_avg_distance_pred = 0.0
                total_avg_distance_gt = 0.0
                print(f"Batch item {i}:")
                for idx, (typing_time, pred_indices, gt_indices, avg_dist_pred, avg_dist_gt) in enumerate(
                        zip(typing_times.cpu().numpy(),
                            selected_indices_pred,
                            selected_indices_gt,
                            avg_distances_pred_list,
                            avg_distances_gt_list)):
                    print(f" Typing time {typing_time:.2f} ms:")
                    print(f"  Selected predicted gaze indices: {pred_indices}")
                    print(f"  Selected ground truth gaze indices: {gt_indices}")
                    print(f"  Avg distance (predicted): {avg_dist_pred:.2f}")
                    print(f"  Avg distance (ground truth): {avg_dist_gt:.2f}")
                    gt_count += len(gt_indices)
                    pred_count += len(pred_indices)
                    if len(pred_indices) > 0:
                        total_avg_distance_pred += avg_dist_pred
                    if len(gt_indices) > 0:
                        total_avg_distance_gt += avg_dist_gt
                print(f" Total selected predicted gaze points: {pred_count}")
                print(f" Total selected ground truth gaze points: {gt_count}")
                print(f" Total avg distance (predicted): {total_avg_distance_pred / len(typing_times):.2f}")
                print(f" Total avg distance (ground truth): {total_avg_distance_gt / len(typing_times):.2f}")
        # ** Debugging Code End **

        # Compute distances between typing positions and gaze positions
        delta_pos_pred = typing_positions.unsqueeze(1) - gaze_positions_pred.unsqueeze(0)  # (N_typing, N_gaze_pred, 2)
        distances_pred = torch.norm(delta_pos_pred, dim=2)  # (N_typing, N_gaze_pred)

        delta_pos_gt = typing_positions.unsqueeze(1) - gaze_positions_gt.unsqueeze(0)  # (N_typing, N_gaze_gt, 2)
        distances_gt = torch.norm(delta_pos_gt, dim=2)  # (N_typing, N_gaze_gt)

        # Apply soft masks to distances
        distances_pred_masked = distances_pred * soft_mask_pred
        distances_gt_masked = distances_gt * soft_mask_gt

        # Sum distances and soft counts per typing point
        sum_distances_pred = distances_pred_masked.sum()
        soft_count_pred = soft_mask_pred.sum()

        sum_distances_gt = distances_gt_masked.sum()
        soft_count_gt = soft_mask_gt.sum()

        # Avoid division by zero
        avg_distances_pred = sum_distances_pred / (soft_count_pred + 1e-8)
        avg_distances_gt = sum_distances_gt / (soft_count_gt + 1e-8)

        # Compute the loss
        distance_loss = torch.abs(avg_distances_pred - avg_distances_gt / 2)
        count_loss = torch.abs(soft_count_pred - soft_count_gt)

        total_distance_loss += distance_loss
        total_count_loss += count_loss

    total_distance_loss = total_distance_loss / batch_size
    total_count_loss = total_count_loss / batch_size

    # normalize the total_distance_loss and total_count_loss
    total_distance_loss *= distance_loss_normalizer
    total_count_loss *= count_loss_normalizer
    return total_distance_loss * count_loss_ratio, total_count_loss * distance_loss_ratio


def multi_match_loss(outputs, targets, scaler_y, masks_y, predict_masks_y, max_pred_len=32, include_duration=True,
                     pos_weight=0.5, dur_weight=0.5, shape_weight=0.5, dir_weight=0.5, length_weight=0.5,
                     time_penalty_weight=0.05, position_penalty_weight=0.05):
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        masks_y = masks_y.unsqueeze(0)
        predict_masks_y = predict_masks_y.unsqueeze(0)
    batch_size = outputs.shape[0]
    predict_masks_y = torch.round(predict_masks_y)

    loss_value_list = []
    pos_sim_list = []
    dur_sim_list = []
    shape_sim_list = []
    dir_sim_list = []
    length_sim_list = []

    # Extract the scaler parameters
    scaler_y_mean = torch.tensor(scaler_y.mean_, device=outputs.device)
    scaler_y_scale = torch.tensor(scaler_y.scale_, device=outputs.device)

    for i in range(batch_size):
        outputs_i = outputs[i]
        targets_i = targets[i]
        masks_y_i = masks_y[i]
        predict_masks_y_i = predict_masks_y[i]

        # Perform the inverse scaling using PyTorch operations
        outputs_inv = (outputs_i * scaler_y_scale) + scaler_y_mean
        targets_inv = (targets_i * scaler_y_scale) + scaler_y_mean

        valid_outputs = outputs_inv[:max_pred_len]
        valid_targets = targets_inv[:max_pred_len]
        valid_masks_y = masks_y_i[:max_pred_len]
        # currently for training using the ground truth masks y
        valid_predict_masks_y = predict_masks_y_i[:max_pred_len]

        valid_outputs = valid_outputs[valid_masks_y == 1]
        valid_targets = valid_targets[valid_masks_y == 1]

        time_penalty_threshold = valid_targets[:, -1].max() * 1.5

        loss_value, pos_sim, dur_sim, shape_sim, dir_sim, length_sim = multi_match(valid_outputs,
                                                                                   valid_targets,
                                                                                   return_components=True,
                                                                                   include_duration=include_duration,
                                                                                   device=outputs.device,
                                                                                   pos_weight=pos_weight,
                                                                                   dur_weight=dur_weight,
                                                                                   shape_weight=shape_weight,
                                                                                   dir_weight=dir_weight,
                                                                                   length_weight=length_weight)
        loss_value = 1.0 - loss_value

        # Add the time duration regularization term
        loss_value += time_duration_penalty(valid_outputs, threshold=time_penalty_threshold,
                                            penalty_weight=time_penalty_weight)
        loss_value += position_penalty(valid_outputs, penalty_weight=position_penalty_weight)
        loss_value_list.append(loss_value)
        pos_sim_list.append(pos_sim)
        dur_sim_list.append(dur_sim)
        shape_sim_list.append(shape_sim)
        dir_sim_list.append(dir_sim)
        length_sim_list.append(length_sim)

    loss_value_tensor = torch.stack(loss_value_list)

    return loss_value_tensor.mean(), pos_sim_list, dur_sim_list, shape_sim_list, dir_sim_list, length_sim_list
