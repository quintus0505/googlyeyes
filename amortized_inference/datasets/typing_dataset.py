import os, pickle
import os.path as osp
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
from config import DEFAULT_DATA_DIR
from googlyeyes.amortized_inference.simulators import TypingSimulator
from googlyeyes.amortized_inference.configs import default_typing_config
import joblib

joblib.parallel_backend('loky', inner_max_num_threads=1)


class TypingUserDataset(object):
    """
    The class 'TypingUserDataset' handles the creation and retrieval of an empirical dataset for touchscreen typing tasks.
    ==> From "How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers"
        (MobileHCI 2019) by Palin et al.
    ==> https://userinterfaces.aalto.fi/typing37k/data/csv_raw_and_processed.zip
    """

    def __init__(self, min_trials=11):
        self.min_trials = min_trials
        # self.fpath = os.path.join(
        #     Path(__file__).parent.parent.parent,
        #     f"data/typing/datasets/empirical_data_trial_min{self.min_trials}.csv"
        # )
        # self.p_fpath = os.path.join(
        #     Path(__file__).parent.parent.parent,
        #     f"data/typing/datasets/empirical_data_info_min{self.min_trials}.csv"
        # )
        self.path = osp.join(DEFAULT_DATA_DIR, 'amortized_inference_test_sections.csv')
        self._get_dataset()
        self.n_user = len(self.df["PARTICIPANT_ID"].unique())

    def _get_dataset(self):
        """
        Retrieve the dataset if it exists, otherwise it processes and creates the dataset from the raw data files.
        Use empirical data where participants used "english" & "qwerty" keyboard and typed with one thumb.
        Also, only data from people with more than specific trials (min_trials) were used.
        """
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            raise FileNotFoundError(f"Dataset not found: {self.path}")

    def sample(self, n_user=None, cross_valid=False):
        """
        Sample the dataset for a given number of users (n_user),
        and optionally splits the data into training and validation sets if cross_valid is True.
        """
        if n_user is None:
            if cross_valid:
                n_user = self.n_user // 2
            else:
                n_user = self.n_user
        outputs = list()

        sampled_p = np.random.choice(self.df["PARTICIPANT_ID"].unique(), size=n_user, replace=False)
        train_p_stat = self.df.query("PARTICIPANT_ID in @sampled_p")
        if cross_valid:
            valid_p_stat = self.df.query("PARTICIPANT_ID not in @sampled_p")
        stat_arr = [train_p_stat, valid_p_stat] if cross_valid else [train_p_stat, ]

        for p_stat in stat_arr:
            output = np.array([p_stat['char_error_rate'],
                               p_stat['IKI'],
                               p_stat['WPM'],
                               p_stat['num_backspaces'],
                               # p_stat['gaze_kbd_ratio'],
                               # p_stat['gaze_shift'],
                               p_stat['WMR'],
                               p_stat['edit_before_commit']
                               ]).T
            outputs.append(output)

        if cross_valid:
            return outputs[0], outputs[1]
        else:
            return outputs[0]

    def indiv_sample(self, n_user=None, cross_valid=False, for_pop=False):
        """
        Sample the dataset for individual users with a given number of users (n_user),
        and optionally splits the data into training and validation sets if cross_valid is True.
        The for_pop parameter, if set to True, concatenates the outputs for all users.
        """
        if n_user is None:
            n_user = self.n_user
        outputs, valid_outputs, user_info = list(), list(), list()

        sampled_p = np.random.choice(self.df["PARTICIPANT_ID"].unique(), size=n_user, replace=False)
        for i in range(n_user):
            target_id = sampled_p[i]
            p_stat = self.df.query("PARTICIPANT_ID == @target_id")
            if cross_valid:
                n_trial = p_stat.shape[0] // 2
            else:
                n_trial = p_stat.shape[0]

            train_p_stat = p_stat.iloc[:n_trial]
            valid_p_stat = p_stat.iloc[n_trial:]

            indiv_output = np.array([train_p_stat['char_error_rate'],
                                     train_p_stat['IKI'],
                                     train_p_stat['WPM'],
                                     train_p_stat['num_backspaces'],
                                     # train_p_stat['gaze_kbd_ratio'],
                                     # train_p_stat['gaze_shift'],
                                     train_p_stat['WMR'],
                                     train_p_stat['edit_before_commit']
                                     ]).T
            outputs.append(indiv_output)

            if cross_valid:
                valid_indiv_output = np.array([valid_p_stat['char_error_rate'],
                                               valid_p_stat['IKI'],
                                               valid_p_stat['WPM'],
                                               valid_p_stat['num_backspaces'],
                                               # valid_p_stat['gaze_kbd_ratio'],
                                               # valid_p_stat['gaze_shift'],
                                               valid_p_stat['WMR'],
                                               valid_p_stat['edit_before_commit']
                                               ]).T
                valid_outputs.append(valid_indiv_output)

            user_info.append(dict(
                id=target_id,
                age=self.df.query("PARTICIPANT_ID == @target_id")["AGE"].values.astype(int)[0],
                gender=self.df.query("PARTICIPANT_ID == @target_id")["GENDER"].values.astype(str)[0],
            ))

        if for_pop:
            outputs = np.concatenate(outputs, axis=0)
        if cross_valid:
            valid_outputs = np.concatenate(valid_outputs, axis=0)
            return outputs, valid_outputs, user_info
        else:
            return outputs, user_info


class TypingTrainDataset(object):
    """
    The class 'TypingTrainDataset' handles the creation and retrieval of a training dataset with touchscreen typing simulator.
    It samples data from the dataset and supports various configurations.
    """

    def __init__(self, total_sim=2 ** 20, n_ep=16, sim_config=None, model='CRTypist_v1'):
        """
        Initialize with the total number of simulations, number of episodes, simulator configuration. 
        """
        if sim_config is None:
            self.sim_config = deepcopy(default_typing_config["simulator"])
        else:
            self.sim_config = sim_config
        self.sim_config["seed"] = 100
        self.total_sim = total_sim
        self.n_ep = n_ep
        self.n_param = total_sim // n_ep
        self.name = f"{self.total_sim // 1000000}M_step_{self.n_ep}ep"
        if self.sim_config["use_uniform"]:
            self.name += "_uniform"
        self.model = model
        self.fpath = os.path.join(
            DEFAULT_DATA_DIR,
            f"train_{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the TypingSimulator.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ train dataset ] {self.fpath}")
            self.simulator = TypingSimulator(self.sim_config, model=self.model)

            params_list = []
            stats_list = []
            finger_log_list = []
            vision_log_list = []
            for i in tqdm(range(self.n_param)):
                try:
                    args = self.simulator.simulate(1, self.n_ep, verbose=False)
                    params_list.append(args[0])
                    stats_list.append(args[1])
                    finger_log_list.append(args[2])
                    vision_log_list.append(args[3])
                except Exception as e:
                    print(f"Error in simulation {i}: {e}")

            params_arr = np.concatenate(params_list, axis=0)
            stats_arr = np.concatenate(stats_list, axis=0)
            finger_log_list = [item[0] for item in finger_log_list]
            vision_log_list = [item[0] for item in vision_log_list]

            self.dataset = dict(
                params=params_arr,
                stats=stats_arr,
                finger_log=finger_log_list,
                vision_log=vision_log_list
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f, protocol=4)

    def sample(self, batch_sz, sim_per_param=1):
        """
        Returns a random sample from the dataset with the specified number of parameter sets (batch size),
        and number of simulated trials per parameters (sim_per_param).
        """
        indices = np.random.choice(self.n_param, batch_sz)
        ep_indices = np.random.choice(self.n_ep, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        if sim_per_param == 1:
            return (
                np.array(self.dataset["params"][indices], dtype=np.float32),
                np.array(self.dataset["stats"][rows, cols].squeeze(1), dtype=np.float32),
            )
        else:
            return (
                np.array(self.dataset["params"][indices], dtype=np.float32),
                np.array(self.dataset["stats"][rows, cols], dtype=np.float32),
            )


class TypingValidDataset(object):
    """
    The class 'TypingValidDataset' handles the creation and retrieval of a validation dataset with touchscreen typing simulator.
    """

    def __init__(self, n_param=100, sim_per_param=500, model='CRTypist_v1', keyboard='chi', use_optimal_params=False, fpath_header="valid"):
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
        self.keyboard = keyboard
        self.fpath = os.path.join(
            DEFAULT_DATA_DIR,
            # f"valid_{self.n_param}_param_{self.sim_per_param}ep.pkl"
            f"{fpath_header}_{self.n_param}_param_{self.sim_per_param}ep.pkl"
        )
        self.use_optimal_params = use_optimal_params
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the TypingSimulator.
        """
        params_list = []
        stats_list = []
        finger_log_list = []
        vision_log_list = []

        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
                # recover the list of arrays
                params_list = [np.expand_dims(self.dataset['params'][i], axis=0) for i in
                               range(self.dataset['params'].shape[0])]
                stats_list = [np.expand_dims(self.dataset['stats'][i], axis=0) for i in
                              range(self.dataset['stats'].shape[0])]
                # add one more dimension to the finger_log_list and vision_log_list
                finger_log_list = [[item] for item in self.dataset['finger_log']]
                vision_log_list = [[item] for item in self.dataset['vision_log']]
                print("shape of params_arr: {}".format(self.dataset['params'].shape))
                print("shape of stats_arr: {}".format(self.dataset['stats'].shape))

            if len(params_list) >= self.n_param:
                return

            else:
                print(
                    "Not enough data in the existing dataset. Continuing to collect more data starting from {}".format(
                        len(params_list)))

        print(f"[ valid dataset ] {self.fpath}")
        self.simulator = TypingSimulator(self.sim_config, model=self.model, keyboard=self.keyboard,
                                         use_optimal_params=self.use_optimal_params)

        try:
            while len(params_list) < self.n_param:
                print(f"Collecting data with index {len(params_list)}")
                args = self.simulator.simulate(1, self.sim_per_param, verbose=False,
                                               n_max_eval_sentence=self.max_sim_per_param,
                                               fixed_params=self.simulator.fixed_params)
                if args[1].shape[1] < self.sim_per_param:
                    continue
                params_list.append(args[0])
                stats_list.append(args[1])
                finger_log_list.append(args[2])
                vision_log_list.append(args[3])
                if len(params_list) % 2 == 0:  # Save progress every 10 iterations
                    self._save_progress(params_list, stats_list, finger_log_list, vision_log_list)
                    print(f"Progress saved: {len(params_list)} params collected")

        except KeyboardInterrupt:
            print("Interrupted! Saving progress with total {} params".format(len(params_list)))
            self._save_progress(params_list, stats_list, finger_log_list, vision_log_list)
            raise

        params_arr = np.concatenate(params_list, axis=0)
        stats_arr = np.concatenate(stats_list, axis=0)
        finger_log_list = [item[0] for item in finger_log_list]
        vision_log_list = [item[0] for item in vision_log_list]
        self.dataset = dict(
            params=params_arr,
            stats=stats_arr,
            finger_log=finger_log_list,
            vision_log=vision_log_list
        )
        # print the shape of
        print("shape of params_arr: {}".format(params_arr.shape))
        print("shape of stats_arr: {}".format(stats_arr.shape))
        # self._save_progress(params_list, stats_list, finger_log_list, vision_log_list, final=True)

    def _save_progress(self, params_list, stats_list, finger_log_list, vision_log_list, final=False):
        """
        Save the current progress of dataset creation to a file.
        """
        params_arr = np.concatenate(params_list, axis=0)
        stats_arr = np.concatenate(stats_list, axis=0)
        finger_log_list = [item[0] for item in finger_log_list]
        vision_log_list = [item[0] for item in vision_log_list]
        self.dataset = dict(
            params=params_arr,
            stats=stats_arr,
            finger_log=finger_log_list,
            vision_log=vision_log_list
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
            n_user = self.n_param
        params = list()
        stats = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats.append(self.dataset["stats"][user][:n_trial])
        return np.array(params, dtype=np.float32), np.array(stats, dtype=np.float32)


if __name__ == "__main__":
    typing_dataset = TypingValidDataset(n_param=20, sim_per_param=50, model='CRTypist_v1', keyboard='chi',
                                        use_optimal_params=False, fpath_header="new_train")
