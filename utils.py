import os.path as osp
import os

DEFAULT_ROOT_DIR =osp.join((osp.abspath(osp.dirname(__file__))))
GAZE_INFERENCE_DIR = osp.join((osp.abspath(osp.dirname(__file__))))
HOW_WE_TYPE_DATA_DIR = osp.join(GAZE_INFERENCE_DIR, 'data', 'how_we_type')
HOW_WE_TYPE_FINGER_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_finger_motion_capture')
HOW_WE_TYPE_GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_gaze')
HOW_WE_TYPE_TYPING_LOG_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_typing_log')

