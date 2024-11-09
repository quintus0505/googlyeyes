import os.path as osp
import pandas as pd
import os
import numpy as np
from config import HOW_WE_TYPE_TYPING_LOG_DATA_DIR, HOW_WE_TYPE_GAZE_DATA_DIR, \
    HOW_WE_TYPE_FINGER_DATA_DIR, GAZE_INFERENCE_DIR
from string_tools import *
import Levenshtein as lev

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


def load_log_df():
    # all csv file stored in LOG_DIR, load them all and concat into one dataframe, only use 'SENTENCE_ID', 'DATA' and 'INPUT'
    log_df = None
    selected_columns_id = [original_log_columns.index(col) for col in used_log_columns if col in original_log_columns]
    for file in os.listdir(LOG_DIR):
        if file.endswith("1.csv"):
            file_path = osp.join(LOG_DIR, file)
            # only use 'SENTENCE_ID', 'DATA' and 'INPUT'
            df = pd.read_csv(file_path, names=original_log_columns, usecols=selected_columns_id)
            # remove the first row
            df = df.iloc[1:]
            if log_df is None:
                log_df = df
            else:
                log_df = pd.concat([log_df, df], ignore_index=True)

    # rename
    log_df.columns = used_log_columns
    log_df['TEST_SECTION_ID'] = (log_df['SENTENCE_ID'] != log_df['SENTENCE_ID'].shift()).cumsum()
    # add a column 'ITE_AUTO' with all 0
    log_df['ITE_AUTO'] = 0
    return log_df


def get_test_sections_df(test_section_id):
    # get the test section dataframe by test_section_id
    test_section_df = log_df[log_df['TEST_SECTION_ID'] == test_section_id]
    committed_sentence = test_section_df['INPUT'].iloc[-1]
    return test_section_df, committed_sentence


if __name__ == '__main__':
    sentences_df = load_sentences_df()
    log_df = load_log_df()
    iter_count = 0
    test_section_ids = log_df['TEST_SECTION_ID'].unique()
    total_correct_count, total_inf_count, total_if_count, total_fix_count = 0, 0, 0, 0
    test_section_count = 0
    total_char_count = 0
    corrected_error_rates = []
    uncorrected_error_rates = []
    # log_index = ['129_9', '130_34']
    log_index = []
    for test_section_id in test_section_ids:
        iter_count += 1
        try:
            test_section_df, committed_sentence = get_test_sections_df(test_section_id)
            # iki is the time between two key presses
            sentence_id = int(test_section_df['SENTENCE_ID'].iloc[0])
            current_index = str(test_section_df['id'].iloc[0]) + '_' + str(sentence_id)
            # print("current_index: ", current_index)
            target_sentence = sentences_df[sentences_df['SENTENCE_ID'] == sentence_id]['SENTENCE'].iloc[0]
            reformatted_input, auto_corrected_if_count, auto_corrected_c_count, \
            auto_corrected_word_count, auto_correct_count, auto_correct_flag, \
            immediate_error_correction_count, delayed_error_correction_count, bsp_count = reformat_input(
                test_section_df)
            flagged_IS = flag_input_stream(reformatted_input)
            unique_transposition_sets = []
            _, MSD = min_string_distance(target_sentence, committed_sentence)

            alignments = []

            align(target_sentence, committed_sentence, MSD, len(target_sentence), len(committed_sentence), "", "",
                  alignments)

            all_triplets = stream_align(flagged_IS, alignments)
            all_edited_triplets = assign_position_values(all_triplets)
            all_error_lists = error_detection(all_edited_triplets)
            lev_distance = lev.distance(target_sentence, committed_sentence)
            for error_list in all_error_lists:
                inf_count, if_count, correct_count, fix_count, slips_info = count_component(error_list, verbose=False)
                if inf_count == lev_distance:
                    break
            Target = target_sentence
            Typed = committed_sentence
            Reformatted = reformatted_input
            if lev_distance != inf_count:
                print("lev_distance: ", lev_distance)
                print("inf_count: ", inf_count)
                print("test_section_id: ", test_section_id)
                continue
            # inf_count = lev_distance
            INF = inf_count
            IF = if_count
            C = correct_count
            total_correct_count += correct_count + auto_corrected_c_count - auto_corrected_word_count
            total_inf_count += inf_count
            total_if_count += if_count + auto_corrected_if_count
            total_fix_count += fix_count + auto_correct_count
            test_section_count += 1
            # uncorrected_error_rate = inf_count / len(target_sentence)
            uncorrected_error_rate = inf_count / (inf_count + if_count + correct_count)
            corrected_error_rate = if_count / (inf_count + if_count + correct_count)

            corrected_error_rates.append(corrected_error_rate)
            uncorrected_error_rates.append(uncorrected_error_rate)
            total_char_count += len(target_sentence)

            if current_index in log_index:
                print("current_index: ", current_index)
                print("Target: ", Target)
                print("Typed: ", Typed)
                print("Reformatted: ", Reformatted)
                # use systime to compute iki (num of keystrokes) / total time
                total_time = int(test_section_df['systime'].iloc[-1]) - int(test_section_df['systime'].iloc[0])
                num_of_keystrokes = len(test_section_df)
                iki = total_time / num_of_keystrokes
                word_num = len(committed_sentence.split())
                wpm = word_num / (total_time / 60) * 1000

                print("IKI: ", iki)
                print('wpm: ', wpm)
                print('error_rate: ', uncorrected_error_rate)
                print("bsp_count: ", bsp_count)
            # if iter_count % 1000 == 0:
            #     print("test_section_count: ", test_section_count)
            #     print("test_section_id: ", test_section_id)
            #     uncorrected_error_rate = total_inf_count / (total_correct_count + total_inf_count + total_if_count)
            #     corrected_error_rate = total_if_count / (total_correct_count + total_inf_count + total_if_count)
            #     print("Corrected error rate: ", corrected_error_rate)
            #     print("Uncorrected error rate: ", uncorrected_error_rate)

        except:
            pass
    uncorrected_error_rate = np.mean(uncorrected_error_rates)
    corrected_error_rate = np.mean(corrected_error_rates)
    print("Corrected error rate: ", corrected_error_rate)
    print("Corrected error rate std: ", np.std(corrected_error_rates))
    print("Uncorrected error rate: ", uncorrected_error_rate)
    print("Uncorrected error rate std: ", np.std(uncorrected_error_rates))
    print("Selected test section count: ", test_section_count)
    print("Total test section count: ", iter_count)
