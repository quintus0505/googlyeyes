from config import HOW_WE_TYPE_TYPING_LOG_DATA_DIR
from string_tools import *
import os.path as osp

LOG_DIR = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')

original_sentences_columns = ['sentence_n', 'sentence']
sentences_columns = ['SENTENCE_ID', 'SENTENCE']
# systime	id	block	sentence_n	trialtime	event	layout	message	touchx	touchy
original_log_columns = ['systime', 'id', 'block', 'SENTENCE_ID', 'trialtime', 'DATA', 'layout', 'INPUT', 'touchx',
                        'touchy']
used_log_columns = ['systime', 'id', 'block', 'SENTENCE_ID', 'DATA', 'INPUT']


def load_sentences_df():
    sentences_path = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Sentences.csv')
    sentences_df = pd.read_csv(sentences_path, usecols=original_sentences_columns)
    # rename columns
    sentences_df.columns = sentences_columns
    return sentences_df


if __name__ == '__main__':
    pass
