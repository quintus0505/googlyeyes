import os.path
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import truncnorm
import warnings
from typing_env.internal_env import InternalEnv
from typing_env.internal_env_v2 import InternalEnvHumanError
from setting import KEYS, KEYS_FIN, PLACES, PLACES_FIN, CHARS, CHARS_FIN
from models.supervisor_agent import SupervisorAgent
from config import DEFAULT_ROOT_DIR
import os.path as osp
from metrics import Metrics

warnings.simplefilter("ignore", UserWarning)

from ..configs import default_typing_config


# from .typing.supervisor.supervisor_agent import SupervisorAgent


class TypingSimulator(object):
    """
    A simulator of touchscreen typing behavior
    Touchscreen typing model from CHI 2021 paper by Jokinen et al.
    """

    def __init__(self, config=None, model="CRTypist_v1", keyboard='chi', use_optimal_params=False):
        if config is None:
            self.config = deepcopy(default_typing_config["simulator"])
        else:
            self.config = config
        if "seed" in self.config:
            self.seeding(self.config["seed"])
        else:
            self.seeding()
        self.model = model
        self.keyboard = keyboard
        memory_p = 0.32052004358604397
        finger_p = 0.4075566536411429
        vision_p = 0.86587701380511
        if use_optimal_params:
            self.fixed_params = [memory_p, finger_p, vision_p]
        else:
            self.fixed_params = None

        config_path = os.path.join(Path(__file__).parent, "typing", "configs")
        with open(os.path.join(config_path, "config.yml"), "r") as file:
            config_file = yaml.load(file, Loader=yaml.FullLoader)
        with open(os.path.join(config_path, config_file["testing_config"]), "r") as file:
            test_config = yaml.load(file, Loader=yaml.FullLoader)
        if self.keyboard == 'chi':
            keys = KEYS_FIN
            chars = CHARS_FIN
            places = PLACES_FIN
            text_path = osp.join(DEFAULT_ROOT_DIR, 'data/sentences_fin.txt')
            finger_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/finger_agent_chi.pt')
            vision_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/vision_agent_chi.pt')
            supervisor_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/supervisor_agent_chi.pt')
            img_folder = osp.join(DEFAULT_ROOT_DIR, 'kbd1k/CHI21_keyboard/')
            position_file = osp.join(DEFAULT_ROOT_DIR, 'kbd1k/chi21.csv')
        else:
            keys = KEYS
            chars = CHARS
            places = PLACES
            text_path = osp.join(DEFAULT_ROOT_DIR, 'data/sentences.txt')
            finger_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/finger_agent.pt')
            vision_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/vision_agent.pt')
            supervisor_path = osp.join(DEFAULT_ROOT_DIR, 'outputs/supervisor_agent.pt')
            img_folder = osp.join(DEFAULT_ROOT_DIR, 'kbd1k/keyboard_dataset/')
            position_file = osp.join(DEFAULT_ROOT_DIR, 'kbd1k/keyboard_label.csv')
        if self.model == "CRTypist_v1":
            self.env = InternalEnv(img_folder=img_folder,
                                   position_file=position_file,
                                   text_path=text_path,
                                   vision_path=vision_path,
                                   finger_path=finger_path,
                                   chars=chars,
                                   keys=keys,
                                   places=places,
                                   render_mode='human')
            self.agent = SupervisorAgent(env=self.env,
                                         load=supervisor_path)
        elif self.model == "CRTypist_v2":
            self.env = InternalEnvHumanError(img_folder=img_folder,
                                             position_file=position_file,
                                             text_path=text_path,
                                             vision_path=vision_path,
                                             finger_path=finger_path,
                                             chars=chars,
                                             keys=keys,
                                             places=places,
                                             render_mode='human')
            self.agent = SupervisorAgent(env=self.env,
                                         load=supervisor_path)
        # self.agent = SupervisorAgent(
        #     config_file["device_config"],
        #     test_config,
        #     train=False,
        #     verbose=False,
        #     variable_params=self.config["variable_params"],
        #     fixed_params=self.config["base_params"],
        #     concat_layers=self.config["concat_layers"],
        #     embed_net_arch=self.config["embed_net_arch"],
        # )

    def simulate(
            self,
            n_param=1,
            n_eval_sentence=1500,
            fixed_params=None,
            random_sample=False,
            return_info=False,
            verbose=False,
            n_max_eval_sentence=2000
    ):
        """
        Simulates human behavior based on given (or sampled) free parameters

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - n_eval_sentence: no. of simulation (i.e., sentences) per parameter set
        - fixed_params: ndarray of free parameter (see below) sets to simulate
        =======
        Free params in menu search model (from CHI'21 paper)
        1) obs_prob:  uniform(min=0.0, max=1.0)
        2) who_alpha: trucnorm(mean=0.6, std=0.3, min=0.4, max=0.9)
        3) who_k:     trucnorm(mean=0.12, std=0.08, min=0.04, max=0.20)
        =======
        - random_sample: set random sentenes from corpus for evaluation (simulation)
        - return_info: return other details incl. episode rewards, lengths, stats, etc.

        Outputs:
        - param_sampled: free parameter sets that are used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - outputs_normalized: behavioral outputs by simulation with normalized values (see below)
            > ndarray with size ((n_param), (n_eval_sentence), (dim. of behavior output))
        =======
        Behavioral output (per each trial)
        1) WPM
        2) Error rate
        3) No. of backspacing
        4) KSPC
        5) Length of sentence
        =======
        """
        if fixed_params is None:
            param_sampled = self.sample_param_from_distr(n_param)
        elif len(np.array(fixed_params).shape) == 1:
            # use the fixed_params as mean an 0.1 as std
            param_sampled = np.random.normal(fixed_params, 0.05, (n_param, len(fixed_params)))
            # param_sampled = np.expand_dims(np.array(fixed_params), axis=0)
        else:
            param_sampled = np.array(fixed_params)
        n_max_eval_sentence = min(n_max_eval_sentence, n_eval_sentence * 3)
        outputs, infos = list(), list()
        finger_log_outputs, vision_log_outputs = list(), list()
        loop = range(param_sampled.shape[0]) if verbose else range(param_sampled.shape[0])
        # for i in loop:
        #     output, info = self.agent.simulate(
        #         fixed_params=param_sampled[i],
        #         n_eval_sentence=n_eval_sentence,
        #         random_sample=random_sample,
        #         return_info=True,
        #     )
        #     outputs.append(output)
        #     infos.append(info)

        for i in loop:
            # print("loop: ", i)
            simulated_run = 0
            iter_output = list()
            iter_finger_logs = list()
            iter_vision_logs = list()
            for _ in tqdm(range(n_max_eval_sentence)):
                while True:
                    obs = self.env.reset(parameters=param_sampled[i])
                    done = False
                    while not done:
                        action, _states = self.agent.predict(obs, deterministic=True)
                        obs, reward, done, info = self.env.step(action)
                    metrics = Metrics(log=self.env.log, target_text=self.env.target_text,
                                      finger_log=self.env.finger_log, vision_log=self.env.vision_log)
                    if len(self.env.vision_log) == 0 or len(self.env.finger_log) == 0:
                        continue
                    try:
                        summary = metrics.summary()
                    except:
                        continue
                    if summary['char_error_rate'] <= 0.2: break  # remove outliers
                    # else: continue print("remove outlier")
                finger_log = self.env.finger_log
                vision_log = metrics.merged_vision_log
                # if any duration in vision_log < 0 continue
                if any([log['duration'] < 0 for log in vision_log]) or any([log['duration'] < 0 for log in finger_log]):
                    continue
                finger_total_time = 0
                vision_total_time = 0
                finger_duration = []
                gaze_duration = []
                for item in finger_log:
                    finger_total_time += item['duration']
                    finger_duration.append(item['duration'])
                for item in vision_log:
                    vision_total_time += item['duration']
                    gaze_duration.append(item['duration'])
                time_test = 50 * len(self.env.log) / 1000
                time = finger_total_time / 1000
                typed_text = self.env.log[-1]['typed_text']  # > is the enter
                if typed_text == "": return 0
                wpm = (len(typed_text) - 1) / time * 60 * (1 / 5)
                iki = np.mean(finger_duration)
                fixation_duration = np.mean(gaze_duration)
                iter_finger_logs.append(finger_log)
                iter_vision_logs.append(vision_log)
                iter_output.append(
                    [summary['char_error_rate'],
                     summary['IKI'],
                     summary['WPM'],
                     summary['num_backspaces'],
                     summary['gaze_kbd_ratio'],
                     summary['gaze_shift'],
                     summary['WMR'],
                     summary['edit_before_commit'],
                     summary['fixation_count'],
                     summary['fixation_duration']])
                simulated_run += 1
                if simulated_run >= n_eval_sentence:
                    break
            if simulated_run < n_max_eval_sentence:
                outputs.append(np.array(iter_output))
                finger_log_outputs.append(iter_finger_logs)
                vision_log_outputs.append(iter_vision_logs)
        # outputs_max = np.array([80, 20, 20, 2.0, 80])
        # outputs_min = np.array([0, 0, 0, 0.5, 0])
        # outputs_normalized = (np.array(outputs) - outputs_min) / (outputs_max - outputs_min) * 2 - 1
        # print("outputs shape: ", np.array(outputs).shape)
        outputs = np.array(outputs)
        if return_info:
            return param_sampled, outputs, finger_log_outputs, vision_log_outputs, infos
        else:
            return param_sampled, outputs, finger_log_outputs, vision_log_outputs

    def sample_param_from_distr(self, n_param):
        param_dim = len(self.config["param_distr"]["distr"])
        # param_distr = self.config["param_distr"]
        param_sampled = list()
        for i in range(param_dim):
            # if param_distr["distr"][i] == "truncnorm":
            #     param_sampled.append(truncnorm.rvs(
            #         (param_distr["minv"][i] - param_distr["mean"][i]) / param_distr["std"][i],
            #         (param_distr["maxv"][i] - param_distr["mean"][i]) / param_distr["std"][i],
            #         param_distr["mean"][i],
            #         param_distr["std"][i],
            #         size=n_param,
            #     ))
            # else:  # "uniform"
            #     param_sampled.append(np.random.uniform(
            #         low=param_distr["minv"][i],
            #         high=param_distr["maxv"][i],
            #         size=n_param,
            #     ))
            param_sampled.append(np.random.uniform(0, 1.0, size=n_param))
        return np.array(param_sampled).T

    def process_log(self, finger_log, vision_log):
        """
        Process logs from simulation.
        For the vision_log, combine consecutive entries with the same 'goal' by summing their durations.
        """
        # Process vision log to merge consecutive entries with the same goal
        merged_vision_log = []
        if vision_log:
            # Start with the first log entry
            current_entry = vision_log[0]

            for i in range(1, len(vision_log)):
                if vision_log[i]['goal'] == current_entry['goal']:
                    # If the goal is the same, sum the duration
                    current_entry['duration'] += vision_log[i]['duration']
                else:
                    # If the goal is different, append the current entry and start a new one
                    merged_vision_log.append(current_entry)
                    current_entry = vision_log[i]

            # Append the last entry
            merged_vision_log.append(current_entry)

        # You can now return or further process the finger_log and merged_vision_log
        return finger_log, merged_vision_log

    def seeding(self, seed=121):
        ## Not implemented
        self.seed = seed
