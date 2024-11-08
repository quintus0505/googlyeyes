import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from config import DEFAULT_DATA_DIR
from googlyeyes.amortized_inference.configs.typing_config import default_typing_config
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from copy import deepcopy
from googlyeyes.amortized_inference.simulators import TypingSimulator
import warnings
from config import GAZE_INFERENCE_DIR
from googlyeyes.model.nets import AmortizedInferenceMLP

model_save_path = os.path.join(GAZE_INFERENCE_DIR, "model", "outputs", "amortized_inference.pth")

warnings.simplefilter("ignore", UserWarning)


# from .typing.supervisor.supervisor_agent import SupervisorAgent
def train_fold(train_loader, model, criterion, optimizer, num_epochs, device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for stats, params in train_loader:
            stats, params = stats.to(device), params.to(device)
            stats = stats.view(stats.size(0), -1).float()
            params = params.float()

            # Forward pass
            outputs = model(stats)
            loss = criterion(outputs, params)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_fold(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for stats, params in test_loader:
            stats, params = stats.to(device), params.to(device)
            stats = stats.view(stats.size(0), -1).float()
            params = params.float()

            outputs = model(stats)
            loss = criterion(outputs, params)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')
    return avg_loss


class TypingDataset(object):
    """
    The class 'TypingValidDataset' handles the creation and retrieval of a validation dataset with touchscreen typing simulator.
    """

    def __init__(self, n_param=2000, sim_per_param=50, model='CRTypist_v1', fpath_header="valid"):
        """
        Initialize the dataset object with a specified number of total user (different parameter sets), episodes,
        and a simulation configuration.
        """
        self.n_param = n_param
        self.sim_per_param = sim_per_param
        self.max_sim_per_param = 2000
        self.sim_config = deepcopy(default_typing_config["simulator"])
        self.sim_config["seed"] = 121
        self.model = model
        self.fpath = os.path.join(
            DEFAULT_DATA_DIR,
            f"{fpath_header}_{self.n_param}_param_{self.sim_per_param}ep.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the TypingSimulator.
        """
        params_list = []
        stats_list = []

        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
                # recover the list of arrays
                params_list = [np.expand_dims(self.dataset['params'][i], axis=0) for i in
                               range(self.dataset['params'].shape[0])]
                stats_list = [np.expand_dims(self.dataset['stats'][i], axis=0) for i in
                              range(self.dataset['stats'].shape[0])]

            if len(params_list) >= self.n_param / 2:
                print("Get enough data in the existing dataset with total {} parameters".format(len(params_list)))
                return

            else:
                print(
                    "Not enough data in the existing dataset. Continuing to collect more data starting from {}".format(
                        len(params_list)))

        print(f"[ valid dataset ] {self.fpath}")
        self.simulator = TypingSimulator(self.sim_config, model=self.model)

        try:
            while len(params_list) < self.n_param:
                print(f"Collecting data with index {len(params_list)}")
                args = self.simulator.simulate(1, self.sim_per_param, verbose=False,
                                               n_max_eval_sentence=self.max_sim_per_param)
                if args[1].shape[1] < self.sim_per_param:
                    continue
                params_list.append(args[0])
                stats_list.append(args[1])

                if len(params_list) % 2 == 0:  # Save progress every 10 iterations
                    self._save_progress(params_list, stats_list)
                    print(f"Progress saved: {len(params_list)} params collected")

        except KeyboardInterrupt:
            print("Interrupted! Saving progress with total {} params".format(len(params_list)))
            self._save_progress(params_list, stats_list)
            raise

        params_arr = np.concatenate(params_list, axis=0)
        stats_arr = np.concatenate(stats_list, axis=0)
        self.dataset = dict(
            params=params_arr,
            stats=stats_arr,
        )
        self._save_progress(params_list, stats_list, final=True)

    def _save_progress(self, params_list, stats_list, final=False):
        """
        Save the current progress of dataset creation to a file.
        """
        params_arr = np.concatenate(params_list, axis=0)
        stats_arr = np.concatenate(stats_list, axis=0)
        self.dataset = dict(
            params=params_arr,
            stats=stats_arr,
        )
        with open(self.fpath, "wb") as f:
            pickle.dump(self.dataset, f, protocol=4)
        if final:
            print("Final dataset saved successfully")

    def sample(self, n_trial, n_user=None):
        """
        Return a sample from the dataset with the given number of trials and users.
        If the number of users is not specified, it defaults to the total number of parameters in the dataset.
        """
        if n_user is None:
            n_user = self.dataset["params"].shape[0]
        params = list()
        stats = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats.append(self.dataset["stats"][user][:n_trial])
        return np.array(params, dtype=np.float32), np.array(stats, dtype=np.float32)


def get_training_and_testing_dataset():
    """
    Get the training and testing dataset for the typing simulator.
    """
    training_dataset = TypingDataset(n_param=5000, sim_per_param=50)
    additional_new_training_dataset = TypingDataset(n_param=4000, sim_per_param=25, fpath_header="new_train")
    additional_training_dataset = TypingDataset(n_param=3000, sim_per_param=50, fpath_header="train")
    additional_distribute_optimal_training_dataset = TypingDataset(n_param=250, sim_per_param=25,
                                                                   fpath_header="distribute_optimal")

    # get the min sim_per_param number (length) from all the datasets
    min_sim_per_param_num = min(training_dataset.dataset['stats'].shape[1],
                                additional_training_dataset.dataset['stats'].shape[1],
                                additional_new_training_dataset.dataset['stats'].shape[1],
                                additional_distribute_optimal_training_dataset.dataset['stats'].shape[1])

    for dataset in [training_dataset, additional_training_dataset, additional_new_training_dataset]:
        if dataset.dataset['stats'].shape[1] > min_sim_per_param_num:
            dataset.dataset['stats'] = dataset.dataset['stats'][:, :min_sim_per_param_num]

    min_status_dim = min(training_dataset.dataset['stats'].shape[-1],
                         additional_training_dataset.dataset['stats'].shape[-1],
                         additional_new_training_dataset.dataset['stats'].shape[-1],
                         additional_distribute_optimal_training_dataset.dataset['stats'].shape[-1])

    for dataset in [training_dataset, additional_training_dataset, additional_new_training_dataset,
                    additional_distribute_optimal_training_dataset]:
        if dataset.dataset['stats'].shape[-1] > min_status_dim:
            dataset.dataset['stats'] = dataset.dataset['stats'][..., :min_status_dim]

    training_dataset.dataset['params'] = np.concatenate(
        [training_dataset.dataset['params'], additional_training_dataset.dataset['params'],
         additional_new_training_dataset.dataset['params'],
         additional_distribute_optimal_training_dataset.dataset['params']], axis=0)
    training_dataset.dataset['stats'] = np.concatenate(
        [training_dataset.dataset['stats'], additional_training_dataset.dataset['stats'],
         additional_new_training_dataset.dataset['stats'],
         additional_distribute_optimal_training_dataset.dataset['stats']], axis=0)

    # testing_dataset = TypingDataset(n_param=100, sim_per_param=500)
    testing_dataset = TypingDataset(n_param=250, sim_per_param=25, fpath_header="val_distribute_optimal")
    # print the shape of the dataset
    print("Training dataset shape: ", training_dataset.dataset['params'].shape, training_dataset.dataset['stats'].shape)
    print("Testing dataset shape: ", testing_dataset.dataset['params'].shape, testing_dataset.dataset['stats'].shape)
    return training_dataset, testing_dataset


# Custom Dataset class
class TypingDatasetWrapper(Dataset):
    def __init__(self, params, stats):
        # self.params = np.repeat(params, stats.shape[1], axis=0)  # Repeat params for each trial
        # self.stats = stats.reshape(-1, stats.shape[2])[..., :4]  # Reshape stats and use only the first 4 dimensions
        self.params = params
        self.stats = np.mean(stats, axis=1)[:, :4]  # Use only the first 4 dimensions of averaged stats

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.stats[idx], self.params[idx]


# Training function
def train(flag='train'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == 'train':
        training_dataset, testing_dataset = get_training_and_testing_dataset()

        train_params, train_stats = training_dataset.sample(n_trial=50)
        print(f'Training dataset shape: {train_params.shape} {train_stats.shape}')
    else:
        testing_dataset = TypingDataset(n_param=250, sim_per_param=25, fpath_header="val_distribute_optimal")
    test_params, test_stats = testing_dataset.sample(n_trial=50)
    print(f'Testing dataset shape: {test_params.shape} {test_stats.shape}')

    # Define hyperparameters
    input_size = 4  # Dimensionality of the first 4 averaged stats
    hidden_size = 128  # Can be adjusted
    output_size = 3  # Use only the first 3 dimensions of params
    num_epochs = 200
    learning_rate = 10e-5
    k_folds = 5

    model = AmortizedInferenceMLP(input_size, hidden_size, output_size).to(device)
    # Set up K-Fold cross-validation
    if flag == 'train':
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        # K-Fold Cross Validation model evaluation
        # Combine train_params and train_stats into a single dataset
        dataset = TypingDatasetWrapper(train_params, train_stats)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f'FOLD {fold + 1}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = Subset(dataset, train_ids)
            test_subsampler = Subset(dataset, test_ids)

            train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_subsampler, batch_size=32, shuffle=False)

            # Initialize model, loss function, and optimizer

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train and evaluate for the current fold
            train_fold(train_loader, model, criterion, optimizer, num_epochs, device)
            avg_loss = evaluate_fold(test_loader, model, criterion, device)

            print(f'Fold {fold + 1} - Average Test Loss: {avg_loss:.4f}')
            print('--------------------------------')
            learning_rate *= 0.75
    else:
        model.load_state_dict(torch.load(model_save_path))
    # Final evaluation on the test set
    model.eval()
    test_dataset = TypingDatasetWrapper(test_params, test_stats)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    with torch.no_grad():
        total_loss = 0
        for stats, params in test_loader:
            stats, params = stats.to(device), params.to(device)
            stats = stats.view(stats.size(0), -1).float()
            params = params.float()

            outputs = model(stats)
            loss = criterion(outputs, params)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Final Test Loss: {avg_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train('train')
    # train('test')
