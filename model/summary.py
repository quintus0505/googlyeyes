from googlyeyes.model.data import calculate_gaze_metrics
from googlyeyes.model.data import calculate_gaze_metrics
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, typing_data, gaze_data):
        self.typing_data = typing_data
        self.gaze_data = gaze_data
        self.typing_log_time = []
        self.ground_truth_gaze_log_time = []
        self.predict_gaze_log_time = []
        self.result_dict = {}
        self.multi_match_pos_sims = []
        self.multi_match_dur_sims = []
        self.multi_match_shape_sims = []
        self.multi_match_direction_sims = []
        self.multi_match_length_sims = []
        self.multi_match_overall_sims = []
        self.multi_match_avg_pos_dur_sims = []
        self.dtwd_score = []
        self.sted_score = []
        self.loss_list = []
        self.trail_loss_list = []
        self.padding_loss_list = []
        self.predict_gaze_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration'])
        self.target_gaze_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration'])
        self.typing_log_df = pd.DataFrame(columns=['index', 'x', 'y', 'duration', 'key', 'trailtime'])

    def record_scanpath_level_metrics(self, index, pos_sim, dur_sim, shape_sim, direction_sim, length_sim,
                                      overall_similarity, dtwd_dist, sted_dist):
        self.multi_match_pos_sims.append(pos_sim)
        self.multi_match_dur_sims.append(dur_sim)
        self.multi_match_shape_sims.append(shape_sim)
        self.multi_match_direction_sims.append(direction_sim)
        self.multi_match_length_sims.append(length_sim)
        self.multi_match_overall_sims.append(overall_similarity)
        self.multi_match_avg_pos_dur_sims.append((pos_sim + dur_sim) / 2)
        self.dtwd_score.append(dtwd_dist)
        self.sted_score.append(sted_dist)
        self.result_dict[index[0]] = {'dtwd_score': dtwd_dist, 'sted_score': sted_dist,
                                      'overall_similarity': overall_similarity, 'pos_similarity': pos_sim,
                                      'dur_similarity': dur_sim, 'shape_similarity': shape_sim,
                                      'direction_similarity': direction_sim, 'length_similarity': length_sim}

    def record_loss_item(self, loss, trail_loss, padding_loss):
        self.loss_list.append(loss)
        self.trail_loss_list.append(trail_loss)
        self.padding_loss_list.append(padding_loss)

    def build_dataframes(self, valid_gaze_outputs, valid_targets, index):
        typing_log = self.typing_data[self.typing_data['index'] == index[0]]
        temp_list = []
        target_gaze_list = []
        typing_list = []
        for i, (x, y, duration) in enumerate(valid_gaze_outputs):
            temp_list.append([index[0], x, y, duration])
        for i, (x, y, duration) in enumerate(valid_targets):
            target_gaze_list.append([index[0], x, y, duration])
        for i, (x, y, duration) in enumerate(typing_log[['x', 'y', 'duration']].values):
            typing_list.append([index[0], x, y, duration])
        self.predict_gaze_df = pd.concat([self.predict_gaze_df,
                                          pd.DataFrame(temp_list, columns=['index', 'x', 'y', 'duration'])],
                                         ignore_index=True)
        self.target_gaze_df = pd.concat([self.target_gaze_df,
                                         pd.DataFrame(target_gaze_list,
                                                      columns=['index', 'x', 'y', 'duration'])],
                                        ignore_index=True)
        self.typing_log_df = pd.concat([self.typing_log_df,
                                        pd.DataFrame(typing_list, columns=['index', 'x', 'y', 'duration'])],
                                       ignore_index=True)

        self.predict_gaze_log_time.append(np.sum([duration for _, _, _, duration in temp_list]) / 1000)
        self.ground_truth_gaze_log_time.append(
            np.sum([duration for _, _, _, duration in target_gaze_list]) / 1000)
        self.typing_log_time.append(np.sum([duration for _, _, _, duration in typing_list]) / 1000)

    def print_inferred_data_result(self):
        avg_pos_sim = np.mean(self.multi_match_pos_sims)
        std_pos_sim = np.std(self.multi_match_pos_sims)
        avg_dur_sim = np.mean(self.multi_match_dur_sims)
        std_dur_sim = np.std(self.multi_match_dur_sims)
        avg_shape_sim = np.mean(self.multi_match_shape_sims)
        std_shape_sim = np.std(self.multi_match_shape_sims)
        avg_direction_sim = np.mean(self.multi_match_direction_sims)
        std_direction_sim = np.std(self.multi_match_direction_sims)
        avg_length_sim = np.mean(self.multi_match_length_sims)
        std_length_sim = np.std(self.multi_match_length_sims)
        avg_overall_sim = np.mean(self.multi_match_overall_sims)
        std_overall_sim = np.std(self.multi_match_overall_sims)
        avg_pos_dur_sim = np.mean(self.multi_match_avg_pos_dur_sims)
        std_pos_dur_sim = np.std(self.multi_match_avg_pos_dur_sims)
        avg_dtwd_score = np.mean(self.dtwd_score)
        std_dtwd_score = np.std(self.dtwd_score)
        avg_sted_score = np.mean(self.sted_score)
        std_sted_score = np.std(self.sted_score)
        avg_loss = np.mean(self.loss_list)
        std_loss = np.std(self.loss_list)
        avg_trail_loss = np.mean(self.trail_loss_list)
        std_trail_loss = np.std(self.trail_loss_list)
        avg_padding_loss = np.mean(self.padding_loss_list)
        std_padding_loss = np.std(self.padding_loss_list)

        print("#" * 50)
        print(f'Average multimatch position similarity per trail: {avg_pos_sim}')
        print(f'Average multimatch duration similarity per trail: {avg_dur_sim}')
        print(f'Average multimatch shape similarity per trail: {avg_shape_sim}')
        print(f'Average multimatch direction similarity per trail: {avg_direction_sim}')
        print(f'Average multimatch length similarity per trail: {avg_length_sim}')
        print(f'Average multimatch overall similarity per trail: {avg_overall_sim}')
        print(f'Average multimatch average position-duration similarity per trail: {avg_pos_dur_sim}')
        print(f'Average DTWD score per trail: {avg_dtwd_score}')
        print(f'Average STED score per trail: {avg_sted_score}')
        print(f'Average loss per trail: {avg_loss}')
        print(f'Average trail loss per trail: {avg_trail_loss}')
        print(f'Average padding loss per trail: {avg_padding_loss}')

        print(f'Standard deviation of multimatch position similarity per trail: {std_pos_sim}')
        print(f'Standard deviation of multimatch duration similarity per trail: {std_dur_sim}')
        print(f'Standard deviation of multimatch shape similarity per trail: {std_shape_sim}')
        print(f'Standard deviation of multimatch direction similarity per trail: {std_direction_sim}')
        print(f'Standard deviation of multimatch length similarity per trail: {std_length_sim}')
        print(f'Standard deviation of multimatch overall similarity per trail: {std_overall_sim}')
        print(f'Standard deviation of multimatch average position-duration similarity per trail: {std_pos_dur_sim}')
        print(f'Standard deviation of DTWD score per trail: {std_dtwd_score}')
        print(f'Standard deviation of STED score per trail: {std_sted_score}')
        print(f'Standard deviation of loss per trail: {std_loss}')
        print(f'Standard deviation of trail loss per trail: {std_trail_loss}')
        print(f'Standard deviation of padding per trail: {std_padding_loss}')

        metrics = calculate_gaze_metrics(self.predict_gaze_df, log_index=[])
        print(
            f'Average fixation time: {metrics["mean_fixation_duration"]} ms (std: {metrics["std_fixation_duration"]} ms)')
        print(
            f'Average number of fixations per trail: {metrics["mean_fixations"]} (std: {metrics["std_fixations"]})')
        print(
            f'Average number of gaze shifts per trail: {metrics["mean_gaze_shifts"]} (std: {metrics["std_gaze_shifts"]})')
        print(
            f'Time ratio for gaze on keyboard: {metrics["mean_time_ratio_on_keyboard"]} (std: {metrics["std_time_ratio_on_keyboard"]})')
        print(
            f'Time ratio for gaze on text entry: {metrics["mean_time_ratio_on_text_entry"]} (std: {metrics["std_time_ratio_on_text_entry"]})')
        print('average typing time: ', np.mean(self.typing_log_time))
        print('average ground truth gaze time: ', np.mean(self.ground_truth_gaze_log_time))
        print('average predict gaze time: ', np.mean(self.predict_gaze_log_time))
        print("#" * 50)

        return avg_pos_sim, avg_dur_sim, avg_overall_sim

    def print_ground_truth_metrics(self, indices):
        testing_gaze_data = self.gaze_data[self.gaze_data['index'].isin(indices)].copy()
        print("Ground truth gaze data metrics")
        metrics = calculate_gaze_metrics(testing_gaze_data)
        print(
            f'Average fixation time: {metrics["mean_fixation_duration"]} ms (std: {metrics["std_fixation_duration"]} ms)')
        print(
            f'Average number of fixations per trail: {metrics["mean_fixations"]} (std: {metrics["std_fixations"]})')
        print(
            f'Average number of gaze shifts per trail: {metrics["mean_gaze_shifts"]} (std: {metrics["std_gaze_shifts"]})')
        print(
            f'Time ratio for gaze on keyboard: {metrics["mean_time_ratio_on_keyboard"]} (std: {metrics["std_time_ratio_on_keyboard"]})')
        print(
            f'Time ratio for gaze on text entry: {metrics["mean_time_ratio_on_text_entry"]} (std: {metrics["std_time_ratio_on_text_entry"]})')
        # print('\n' * 3)

    def print_best_result(self, num_result=0):
        # the 5 smallest dtwd score's index
        print("The {} smallest DTWD score's index".format(num_result))
        small_dtwd_index = sorted(self.result_dict.items(), key=lambda x: x[1]['dtwd_score'])[:num_result]
        for index, value in small_dtwd_index:
            print(f'Index: {index}')
            print(f'DTWD Score: {value["dtwd_score"]}')
            print(f'STED Score: {value["sted_score"]}')
            print(f'Overall Similarity: {value["overall_similarity"]}')
            print(f'Position Similarity: {value["pos_similarity"]}')
            print(f'Duration Similarity: {value["dur_similarity"]}')
            print(f'Shape Similarity: {value["shape_similarity"]}')
            print(f'Direction Similarity: {value["direction_similarity"]}')
            print(f'Length Similarity: {value["length_similarity"]}')
            print(f'Average Position-Duration Similarity: {(value["pos_similarity"] + value["dur_similarity"]) / 2}')
            print("#" * 50)
        # print('\n' * 3)
        # the 5 smallest sted score's index
        print("The {} smallest STED score's index".format(num_result))
        small_sted_index = sorted(self.result_dict.items(), key=lambda x: x[1]['sted_score'])[:num_result]
        for index, value in small_sted_index:
            print(f'Index: {index}')
            print(f'DTWD Score: {value["dtwd_score"]}')
            print(f'STED Score: {value["sted_score"]}')
            print(f'Overall Similarity: {value["overall_similarity"]}')
            print(f'Position Similarity: {value["pos_similarity"]}')
            print(f'Duration Similarity: {value["dur_similarity"]}')
            print(f'Shape Similarity: {value["shape_similarity"]}')
            print(f'Direction Similarity: {value["direction_similarity"]}')
            print(f'Length Similarity: {value["length_similarity"]}')
            print(f'Average Position-Duration Similarity: {(value["pos_similarity"] + value["dur_similarity"]) / 2}')
            print("#" * 50)
        # print('\n' * 3)
        # the 5 largest overall similarity's index
        print("The {} largest overall similarity's index".format(num_result))
        large_overall_index = sorted(self.result_dict.items(), key=lambda x: x[1]['overall_similarity'], reverse=True)[
                              :num_result]
        for index, value in large_overall_index:
            print(f'Index: {index}')
            print(f'DTWD Score: {value["dtwd_score"]}')
            print(f'STED Score: {value["sted_score"]}')
            print(f'Overall Similarity: {value["overall_similarity"]}')
            print(f'Position Similarity: {value["pos_similarity"]}')
            print(f'Duration Similarity: {value["dur_similarity"]}')
            print(f'Shape Similarity: {value["shape_similarity"]}')
            print(f'Direction Similarity: {value["direction_similarity"]}')
            print(f'Length Similarity: {value["length_similarity"]}')
            print(f'Average Position-Duration Similarity: {(value["pos_similarity"] + value["dur_similarity"]) / 2}')
            print("#" * 50)
