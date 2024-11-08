import sys
import os
# PYTHONPATH=$PYTHONPATH:/m/home/home3/33/zhuy10/unix/myproject/deep-typing-agent python3 googlyeyes/amortized_inference/amortized_inference.py
# Get the absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# Add the project root directory and other necessary directories to the Python path
sys.path.append(project_root)
# sys.path.append(os.path.join(project_root, 'deep-typing-agent'))
os.environ['PYTHONPATH'] = os.pathsep.join(
    [os.environ.get('PYTHONPATH', ''), os.path.join(project_root, 'deep-typing-agent')])

import numpy as np

from googlyeyes.amortized_inference.trainers.typing_trainer import TypingTrainer

typing_trainers = TypingTrainer(model='CRTypist_v1')

# param_sampled, outputs = typing_trainers.simulator.simulate(
#     n_param=1,
#     n_eval_sentence=30,
# )
#
# print("param_sampled: ", param_sampled)
# print("outputs: ", outputs)

# print(typing_trainers.amortizer)
