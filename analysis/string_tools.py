import pandas as pd
import numpy as np
import textdistance
import Levenshtein as lev
import os.path as osp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def flag_input_stream(input_stream):
    count = 0
    stream_flags = list("·" * len(input_stream))
    stream_counts = list("0" * len(input_stream))
    # Backward-passing the Input Stream
    for i in range(len(input_stream) - 1, -1, -1):
        # Take note of how many deletions there are in a row
        if input_stream[i] == "<":
            stream_counts[i] = str(count)
            count += 1
        else:
            # These characters will appear so they get flagged
            if count == 0:
                stream_flags[i] = str("F")
                stream_counts[i] = str(count)
            else:
                # Do not flag these characters as they'll be deleted
                stream_counts[i] = str(count)
                count -= 1
    # Return the stream flags, the counts and the original Input Stream
    return "".join(stream_flags), "".join(stream_counts), input_stream


# Compute the minimum string distance between two strings
def min_string_distance(str1, str2):
    m, n = len(str1), len(str2)

    # Initialize the matrix with zeros
    distance_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        distance_matrix[i][0] = i
    for j in range(n + 1):
        distance_matrix[0][j] = j

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,  # Deletion
                distance_matrix[i][j - 1] + 1,  # Insertion
                distance_matrix[i - 1][j - 1] + cost  # Substitution
            )

    # Return the value of the Minimum String Distance and the dynamic programming matrix
    return distance_matrix[m][n], distance_matrix


# Compute Optimal Alignments of two Strings
def align(S1, S2, D, x, y, A1, A2, alignments):
    # If both indexes reach zero then the alignment is complete and added to the list
    if x == 0 and y == 0:
        alignments.append([A1, A2])
        return
    # Use the MSD matrix to create the optimal alignments
    if x > 0 and y > 0:
        if D[x][y] == D[x - 1][y - 1] and S1[x - 1] == S2[y - 1]:
            align(S1, S2, D, x - 1, y - 1, S1[x - 1] + A1, S2[y - 1] + A2, alignments)
        if D[x][y] == D[x - 1][y - 1] + 1:
            align(S1, S2, D, x - 1, y - 1, S1[x - 1] + A1, S2[y - 1] + A2, alignments)
    # Insert insertions and omission markers as needed
    if x > 0 and D[x][y] == D[x - 1][y] + 1:
        align(S1, S2, D, x - 1, y, S1[x - 1] + A1, "~" + A2, alignments)
    if y > 0 and D[x][y] == D[x][y - 1] + 1:
        align(S1, S2, D, x, y - 1, "~" + A1, S2[y - 1] + A2, alignments)


# Adapt the alignments to the Input Stream
def stream_align(IS, alignments_list):
    triplets = []
    for alignment_pair in alignments_list:
        new_stream = IS[2]  # Input stream
        new_flags = IS[0]  # Flags
        new_alignments = [alignment_pair[0], alignment_pair[1]]

        # The final length of the alignment triplet will have to include omissions and insertions.
        final_alignment_length = max(len(alignment_pair[1]), len(new_stream))

        for i in range(final_alignment_length):
            if i < len(alignment_pair[1]) and alignment_pair[1][i] == '~':
                # Insert spacer character
                new_stream = new_stream[:i] + "·" + new_stream[i:]
                # Necessary to set every "~" as flag.
                new_flags = new_flags[:i] + "F" + new_flags[i:]
            elif new_flags[i] != "F":
                alignment_pair[0] = alignment_pair[0][:i] + "·" + alignment_pair[0][i:]
                alignment_pair[1] = alignment_pair[1][:i] + "·" + alignment_pair[1][i:]

        triplets += [[alignment_pair[0], alignment_pair[1], [new_stream, new_flags]]]
    # Return the new alignments as a tuple and the flags
    return triplets


# Assign position values to properly conduct error detection
def assign_position_values(triplets):
    edited_triplets = []

    for triplet in triplets:
        # Get the two strings and the input stream alignments
        align_0 = triplet[0]
        align_1 = triplet[1]
        align_IS = triplet[2][0]
        # Plus the flags
        IS_flags = triplet[2][1]
        # And initialize the pos values as zero
        IS_pos = list("0" * len(align_IS))

        pos = 0
        for i in range(len(align_IS)):
            if IS_flags[i] == "F":
                pos = 0
                IS_pos[i] = str(pos)
            else:
                if align_IS[i] == "<" and pos > 0:
                    pos -= 1
                IS_pos[i] = str(pos)
                if align_IS[i] != "<":
                    pos += 1

        new_triplet = [[align_0, align_1, [align_IS, IS_flags, "".join(IS_pos)]]]
        edited_triplets += new_triplet

    # Return a copy of the given triplets now with flags and position values.
    return edited_triplets


# Look Ahead function for Error Detection
def look_ahead(string, start, count, condition_function):
    i = start
    while (i >= 0 and i < len(string)) and not condition_function(string[i]):
        i += 1  # Keep looking until the condition is met.
    while count > 0 and i < len(string):
        i += 1
        if i == len(string):
            break
        elif condition_function(string[i]):
            count -= 1
    return min(i, len(string) - 1)


# Lood Behind function for Error Detection
def look_behind(string, start, count, condition_function):
    i = start
    while (i >= 0 and i < len(string)) and not condition_function(string[i]):
        i -= 1  # Keep looking until the condition is met.
    while count > 0 and i >= 0:
        i -= 1
        if i < 0:
            break
        elif condition_function(string[i]):
            count -= 1
    return max(0, i)


# Character detection functions
def check_special_char(char):
    return char == '~' or char == '#' or char == '·'


def check_not_spacer(char):
    return char != "·"


# Error Detection Function
def error_detection(triplets):
    errors_per_triplet = []

    for triplet in triplets:
        errors = []

        P = triplet[0]  # Target Phrase Align
        T = triplet[1]  # User Input Align
        IS = triplet[2][0]  # Input Stream
        IS_flags = triplet[2][1]  # IS Flags
        IS_pos = triplet[2][2]  # IS Position values

        a = 0
        for b in range(min(len(IS), len(T))):  # 0 to |IS|-1
            if T[b] == '~':
                errors += [[0, "o", [P[b], "~"]]]  # uncorrected omission
            elif IS_flags[b] == "F" or b == len(IS) - 1:
                M = set()  # Corrected omissions set
                I = set()  # Corrected insertions set
                for i in range(a, b):  # Iterate over substring determined by flags
                    val = int(IS_pos[i])
                    if IS[i] == "<":
                        if val in M:
                            M.remove(val)
                        if val in I:
                            I.remove(val)
                    elif check_not_spacer(IS[i]):
                        target = look_ahead(P, b, val + len(M) - len(I), check_special_char)
                        next_p = look_ahead(P, target, 1, check_special_char)
                        prev_p = look_behind(P, target, 1, check_special_char)
                        next_is = look_ahead(IS, i, 1, check_not_spacer)
                        prev_is = look_behind(IS, i, 1, check_not_spacer)

                        if IS[i] == P[target]:
                            errors += [[1, "n", [IS[i], IS[i]]]]  # corrected no error
                        elif target >= len(P) - 1 or IS[next_is] == P[target] or (
                                IS[prev_is] == IS[i] and IS[prev_is] == P[i]):
                            errors += [[1, "i", ["~", IS[i]]]]  # corrected insertion
                            I.add(val)
                        elif IS[i] == P[next_p] and not check_special_char(T[target]):
                            errors += [[1, "o", [P[target], "~"]]]  # corrected omission
                            errors += [[1, "n", [IS[i], IS[i]]]]  # corrected no error
                            M.add(val)
                        else:
                            errors += [[1, "s", [P[target], IS[i]]]]  # corrected substitution

                if P[b] == "~":
                    errors += [[0, "i", ["~", T[b]]]]  # uncorrected insertion
                elif P[b] != T[b]:
                    errors += [[0, "s", [P[b], T[b]]]]  # uncorrected substitution
                elif P[b] != "·":
                    errors += [[0, "n", [T[b], T[b]]]]  # uncorreced no error
                a = b + 1
        # if len(IS) < len(P):
        #     for i in range(len(IS), len(P)):
        #         errors += [[0, "o", [P[i], "~"]]]
        # remove the end 3 elements of "eof" in errors

        errors_per_triplet += [errors]
    # Returns a series of the errors/non-errors present in the user typed phrase.
    # The series contains items of this type [0, i, [~, z]] where:
    # [1]corrected, [0]uncorrected
    # [i]nsertion, [o]mission, [t]ransposition, [s]ubstitution and [c]apitalization. [n]o error
    # [char expected, char produced]
    return errors_per_triplet


# Specify which kind of substitution error is happening (TRA, CAP or SU)
def specify_errors(error_list):
    updated_errors = []
    last_error = None
    last_last_error = None
    last_last_last_error = None
    for error in error_list:
        if ((error[2][0] != error[2][1]) and  # CAPITALIZATION ERROR
                (error[2][0].lower() == error[2][1].lower()) and
                error[1] == "s"):
            new_error = [error[0], "c", error[2]]
            updated_errors += [new_error]
            last_error = new_error
        else:
            if ((last_error != None) and  # UNCORRECTED TRANSPOSITION IF NOT RE-ALIGNED
                    (error[2][0] == last_error[2][1] and error[2][1] == last_error[2][0]) and
                    (error[0] == last_error[0]) and
                    (error[1] == last_error[1] == "s")):
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][0], error[2][1]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_error != None) and  # UNCORRECTED TRANSPOSITION NOT RE-ALIGNED CASE 1 if input is [~,l2][l1,l1][l2,~]
                  (error[2][0] == last_last_error[2][1] != '~') and
                  (last_error[2][0] == last_error[2][1]) and
                  (error[2][1] == last_last_error[2][0] == '~') and
                  (error[1] == "o" and last_error[1] == "n" and last_last_error[1] == "i")):
                updated_errors.pop()
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][0], last_error[2][0]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_error != None) and  # UNCORRECTED TRANSPOSITION NOT RE-ALIGNED CASE 2 if input is [l2,~][l1,l1][~,l2]
                  (error[2][0] == last_last_error[2][1] == '~') and
                  (last_error[2][0] == last_error[2][1]) and
                  (error[2][1] == last_last_error[2][0] != '~') and
                  (error[1] == "i" and last_error[1] == "n" and last_last_error[1] == "o")):
                updated_errors.pop()
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][1], last_error[2][0]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_last_error != None) and  # CORRECTED TRANSPOSITION (DELETE LAST 4, TWO WRONG INSERTS AND TWO CORRECTIONS -> TRANSPOSITION) if input is [~,l2][~,l1][l2,l2][l1,l1]
                  (last_last_last_error[0] == last_last_error[0] == 1) and
                  (last_last_last_error[1] and last_last_error[1] in ["i", "s"]) and
                  (last_last_last_error[2][1] == error[2][1]) and
                  (last_last_error[2][1] == last_error[2][1])):
                updated_errors.pop()
                updated_errors.pop()
                updated_errors.pop()
                new_error = [1, "t", [error[2][1], last_error[2][1]]]
                updated_errors += [new_error]
                last_error = new_error
            else:  # NO EDIT TO BE MADE
                updated_errors += [error]
                last_error = error
        last_last_last_error = last_last_error
        last_last_error = last_error
    return updated_errors


def count_errors(error_list):
    uncorr_errors = [["i", 0], ["o", 0], ["s", 0], ["t", 0], ["c", 0], ]  # "INS,OMI,SUB,TRA,CAP"
    corr_errors = [["i", 0], ["o", 0], ["s", 0], ["t", 0], ["c", 0], ]
    errors_only_list = []

    for error in error_list:
        if error[0]:
            # Char has been corrected
            for error_type in corr_errors:
                # Increase by one the counter for this kind of error
                if error_type[0] == error[1]:
                    error_type[1] += 1
            if error[1] != 'n':
                errors_only_list.append(error)
        else:
            # Char has not been corrected
            for error_type in uncorr_errors:
                # Increase by one the counter for this kind of error
                if error_type[0] == error[1]:
                    error_type[1] += 1
            if error[1] != 'n':
                errors_only_list.append(error)

    return uncorr_errors, corr_errors, errors_only_list


def count_transpositions(unique_transposition_sets, new_transposition):
    # For each transposition
    # [[exp_1, exp_2], [transp_1, transp_2], count]
    for transposition in unique_transposition_sets:
        # if the error combination is already known
        if new_transposition == transposition[1]:
            # increase the count
            transposition[2] += 1
            break
        # if the combination isn't present then add it with count set to 1
    else:
        unique_transposition_sets.append([[new_transposition[1], new_transposition[0]], new_transposition, 1])


def optimal_error_set(all_error_lists, unique_transposition_sets):
    # Create a list of already known error combinations
    # The objects of the set will have this structure
    # [[uncorr_errors, corr_errors],
    #  [sum_of_errors],
    #  appeareances]
    unique_error_sets = []

    for error_list in all_error_lists:
        # Specify the type of errors
        new_error_list = specify_errors(error_list)
        # Count the total errors per category
        uncorr_errors, corr_errors, errors_only_list = count_errors(new_error_list)
        # Sum uncorrected and corrected errors together so we can filter by highest TRA, lowest INS
        sum_of_errors = []
        for pair1, pair2 in zip(uncorr_errors, corr_errors):
            letter = pair1[0]
            sum_value = pair1[1] + pair2[1]
            sum_of_errors.append([letter, sum_value])

        new_errors = [uncorr_errors, corr_errors]

        for error_set in unique_error_sets:
            # if the error combination is already known
            if new_errors == error_set[0]:
                # increase the count
                error_set[2] += 1
                break
            # if the combination isn't present then add it with count set to 1
        else:
            unique_error_sets.append([new_errors, sum_of_errors, 1, errors_only_list])

        # Sort by max TRA errors and then by minimum sum of all other errors on the sum_of_errors
        best_set = max(unique_error_sets,
                       key=lambda x: (x[1][3][1], - (x[1][0][1] + x[1][1][1] + x[1][2][1] + x[1][4][1])))

        for error in best_set[3]:
            if error[1] == 't':
                count_transpositions(unique_transposition_sets, [error[2][0], error[2][1]])

    return best_set[0], best_set[2]  # THE ERROR WITH HIGHEST COUNT


def return_errors(target_phrase, user_phrase, input_stream, unique_transposition_sets):
    # Get alignments
    flagged_IS = flag_input_stream(input_stream)
    _, MSD = min_string_distance(target_phrase, user_phrase)
    alignments = []
    align(target_phrase, user_phrase, MSD, len(target_phrase), len(user_phrase), "", "", alignments)
    all_triplets = stream_align(flagged_IS, alignments)
    all_edited_triplets = assign_position_values(all_triplets)

    # Get error lists
    all_error_lists = error_detection(all_edited_triplets)

    best_set, occurrences = optimal_error_set(all_error_lists, unique_transposition_sets)

    return best_set[0], best_set[1]


def compute_error_rate(target_phrase, user_phrase, input_stream):
    target_phrase += 'eof'
    user_phrase += 'eof'
    input_stream += 'eof'

    flagged_IS = flag_input_stream(input_stream)
    # print("Phrase Details:")
    # print("Flags: {}\nMoves: {}\nIS   : {}\n".format(*flagged_IS))
    unique_transposition_sets = []
    _, MSD = min_string_distance(target_phrase, user_phrase)

    alignments = []

    align(target_phrase, user_phrase, MSD, len(target_phrase), len(user_phrase), "", "", alignments)

    all_triplets = stream_align(flagged_IS, alignments)
    all_edited_triplets = assign_position_values(all_triplets)
    all_error_lists = error_detection(all_edited_triplets)

    best_set, occurrences = optimal_error_set(all_error_lists, unique_transposition_sets)
    # INF, IF, C, F, slips_info = count_component(all_error_lists[-1])
    # print("INF: {} | IF: {} | C: {} | F: {}".format(INF, IF, C, F))
    # print(slips_info)
    count = 0
    new_all_error_lists = []
    uncorrected_error_rate = 0
    corrected_error_rate = 0
    for error_list in all_error_lists:
        error_list = error_list[:-3]
        # print("Alignment {}:".format(str(count + 1)))
        # print("Target: {}\nUser  : {}\n".format(alignments[count][0], alignments[count][1]))
        new_error_list = specify_errors(error_list)
        count += 1
        INF, IF, C, F, slips_info = count_component(new_error_list)

        uncorrected_error_rate = INF / len(user_phrase)
        corrected_error_rate = IF / (C + INF + IF)

        # print("INF: {} | IF: {} | C: {} | F: {}".format(INF, IF, C, F))
        # print("uncorrected error rate: ", uncorrected_error_rate)
        # print("corrected error rate: ",corrected_error_rate)
        # print(slips_info)
        new_all_error_lists.append(new_error_list)
    # uncorr_errors, corr_errors = optimal_error_set(new_all_error_lists, unique_transposition_sets)
    return uncorrected_error_rate, corrected_error_rate


def count_component(error_list):
    INF, IF, C, F = 0, 0, 0, 0
    slips_info = {'uncorrected': {'INS': 0, 'OMI': 0, 'SUB': 0, 'CAP': 0, 'TRA': 0},
                  'corrected': {'INS': 0, 'OMI': 0, 'SUB': 0, 'CAP': 0, 'TRA': 0}}
    for error in error_list:
        if error[0] == 0:
            if error[1] == "i":
                INF += 1
                slips_info['uncorrected']['INS'] += 1
            elif error[1] == "o":
                INF += 1
                slips_info['uncorrected']['OMI'] += 1
            elif error[1] == "s":
                INF += 1
                slips_info['uncorrected']['SUB'] += 1
            elif error[1] == "c":
                INF += 1
                slips_info['uncorrected']['CAP'] += 1
            elif error[1] == "t":
                INF += 1
                slips_info['uncorrected']['TRA'] += 1
            else:
                C += 1
        else:
            if error[1] == "i":
                IF += 1
                slips_info['corrected']['INS'] += 1
            elif error[1] == "o":
                IF += 1
                slips_info['corrected']['OMI'] += 1
            elif error[1] == "s":
                IF += 1
                slips_info['corrected']['SUB'] += 1
            elif error[1] == "c":
                IF += 1
                slips_info['corrected']['CAP'] += 1
            elif error[1] == "t":
                IF += 1
                slips_info['corrected']['TRA'] += 1
            else:
                # C += 1
                pass
    return INF, IF, C, F, slips_info


def compute_if_c_count_for_auto_correction(str1, str2):
    """
    :param str1: original string (word)
    :param str2: auto-corrected string (word)
    :return:
    """
    # Create a matrix to store the distances and matches
    matrix = [[[0, 0] for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len(str1) + 1):
        matrix[i][0] = [i, 0]  # Distance, Matches
    for j in range(len(str2) + 1):
        matrix[0][j] = [j, 0]  # Distance, Matches

    # Populate the matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
                matches = matrix[i - 1][j - 1][1] + 1  # Increase matches
            else:
                cost = 1
                matches = max(matrix[i - 1][j][1], matrix[i][j - 1][1], matrix[i - 1][j - 1][1])

            # Calculate distances and update matches
            dist_del = matrix[i - 1][j][0] + 1
            dist_ins = matrix[i][j - 1][0] + 1
            dist_sub = matrix[i - 1][j - 1][0] + cost

            min_dist = min(dist_del, dist_ins, dist_sub)
            matrix[i][j] = [min_dist, matches]

            # Ensure the match count does not decrease
            if min_dist == dist_sub:
                matrix[i][j][1] = max(matrix[i][j][1], matrix[i - 1][j - 1][1])

    # The last element of the matrix contains the distance and the matches
    distance, matches = matrix[-1][-1]
    char_count = len(str2)
    return distance, matches, char_count


def reformat_input(test_section_df):
    reformatted_input = ""
    pre_input = ""
    reformat_if_count = 0
    reformat_c_count = 0
    reformat_f_count = 0
    bsp_count = 0
    bsp_index_list = []
    auto_corrected_word_count = 0
    auto_capitalization_count = 0

    auto_correct_flag = False

    immediate_error_correction_count = 0
    delayed_error_correction_count = 0

    def get_bsp_adjustments(bsp_index_list, input_text):
        bsp_adjustments = [0] * len(input_text)
        bsp_running_total = 0
        for bsp_index in bsp_index_list:
            if bsp_index < len(bsp_adjustments):
                bsp_adjustments[bsp_index] = 1
        for i, adjustment in enumerate(bsp_adjustments):
            bsp_running_total += adjustment
            bsp_adjustments[i] = bsp_running_total
        return bsp_adjustments

    for index, row in test_section_df.iterrows():
        if row['INPUT'] != row['INPUT']:
            if len(pre_input) == 1:
                reformatted_input += '<'
            pre_input = ""
            current_input = ''
            continue
        current_input = row['INPUT']
        if current_input != current_input:
            current_input = ''
        if current_input == pre_input:
            continue
        if len(current_input) > len(pre_input):
            # Calculate the point where current_input diverges from pre_input
            # divergence_point = next((i for i, (c_pre, c_curr) in enumerate(zip(pre_input, current_input)) if
            #                          c_pre.lower() != c_curr.lower()), len(pre_input))
            divergence_point = 0
            for i, (c_pre, c_curr) in enumerate(zip(pre_input, current_input)):
                if c_pre.lower() != c_curr.lower():
                    divergence_point = i
                    break

            # Calculate backspace count before the divergence point
            bsp_count_before_divergence = sum(1 for bsp_index in bsp_index_list if bsp_index < divergence_point)

            # Adjust for backspaces in the reformatted_input index
            adjusted_index = divergence_point + 2 * bsp_count_before_divergence

            if current_input[:len(pre_input)].lower() == pre_input.lower():
                if current_input[:len(pre_input)] != pre_input:
                    # Auto capitalization detected
                    auto_capitalization_count += 1
                    auto_correct_flag = True
                    auto_correct_index = 0
                    for i, (c_pre, c_curr) in enumerate(zip(pre_input, current_input)):
                        if c_pre != c_curr:
                            auto_correct_index = i
                            break
                    # Correct the capitalization
                    # reformatted_input = reformatted_input[:adjusted_index] + current_input[
                    #     divergence_point] + reformatted_input[adjusted_index + 1:]
                    reformatted_input = reformatted_input[:auto_correct_index] + current_input[
                        auto_correct_index] + reformatted_input[auto_correct_index + 1:]

                    # Increment counters for auto-correction
                    reformat_if_count += 1
                    reformat_f_count += 1
                    delayed_error_correction_count += 1

                # Handle normal typing or auto capitalization (adding the rest of the input)
                reformatted_input += current_input[len(pre_input):]
            else:
                if lev.distance(pre_input, current_input) == 1:
                    # Handling mid-sentence typing or corrections
                    reformatted_input = reformatted_input[:adjusted_index] + current_input[
                        divergence_point] + reformatted_input[adjusted_index:]
                    reformat_if_count += 1
                    delayed_error_correction_count += 1
                else:  # auto correction
                    # just replace the last word in the reformatted_input as the last word in the current_input
                    if len(reformatted_input.split()) == 1:
                        reformatted_input = current_input
                        reformat_if_count += 1
                        reformat_f_count += 1
                        auto_corrected_word_count += 1
                        delayed_error_correction_count += 1
                        auto_correct_flag = True
                    # else if multiple words are typed in one keystroke
                    else:
                        # if the last char is " ":
                        if current_input[-1] == ' ':
                            reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + \
                                                current_input[:-1].rsplit(' ', 1)[1] + ' '
                        else:
                            reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + \
                                                current_input.rsplit(' ', 1)[1]
                        reformat_if_count += 1
                        reformat_f_count += 1
                        auto_corrected_word_count += 1
                        immediate_error_correction_count += 1
                        auto_correct_flag = True

        elif row['ITE_AUTO'] or len(current_input) == len(pre_input):
            #  use auto correction
            if not row['ITE_AUTO'] and current_input.lower() == pre_input.lower():
                if len(current_input) > 1:
                    auto_capitalization_count += 1
                bsp_adjustments = get_bsp_adjustments(bsp_index_list, current_input)
                for i in range(len(pre_input)):
                    if pre_input[i] != current_input[i]:
                        adjusted_index = i + 2 * bsp_adjustments[i] if bsp_adjustments[i] else i
                        reformatted_input = reformatted_input[:adjusted_index] + current_input[
                            i] + reformatted_input[adjusted_index + 1:]
                        break
                reformat_if_count += 1
                if len(current_input) == 1:
                    auto_correct_flag = False
                else:
                    auto_correct_flag = True

                pre_input = current_input

                delayed_error_correction_count += 1
                continue

            elif not row['ITE_AUTO'] and current_input[:-1] == pre_input[:-1] and len(current_input) > 1:
                # not detected as autocorrected but the log looks like that, maybe multiple input in one keystroke
                bsp_count_before = sum(1 for bsp_index in bsp_index_list if bsp_index < len(current_input) - 1)
                # replace the last character in the reformatted_input with the last character in the current_input
                reformatted_input = reformatted_input[:len(current_input) - 2 + 2 * bsp_count_before] + \
                                    current_input[-1] + \
                                    reformatted_input[len(current_input) - 1 + 2 * bsp_count_before:]
                reformat_if_count += 1
                if len(current_input) == 1 or (current_input[-1] == '.' and pre_input[-1] == ' '):
                    auto_correct_flag = False
                else:
                    auto_correct_flag = True

                pre_input = current_input

                immediate_error_correction_count += 1

                continue
            elif len(reformatted_input.split()) > 1:
                # replace the last word in the reformatted_input
                # with the last word in the current_input
                # if more than one words in the current_input, only consider the last word
                # if the last char is " "
                if current_input[-1] == ' ':
                    reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + \
                                        current_input[:-1].rsplit(' ', 1)[1] + ' '
                    word_before_modification = pre_input.rsplit(' ', 1)[1]
                    word_after_modification = current_input[:-1].rsplit(' ', 1)[1]
                else:
                    reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + current_input.rsplit(' ', 1)[1]
                    word_before_modification = pre_input.rsplit(' ', 1)[1]
                    word_after_modification = current_input.rsplit(' ', 1)[1]

                delayed_error_correction_count += 1
                auto_correct_flag = True
            else:
                # if only one word is typed
                reformatted_input = current_input
                word_before_modification = pre_input
                word_after_modification = current_input

                delayed_error_correction_count += 1
                auto_correct_flag = True

            if_count, c_count, word_count = compute_if_c_count_for_auto_correction(word_before_modification,
                                                                                   word_after_modification)
            reformat_if_count += if_count
            reformat_c_count += c_count
            reformat_f_count += 1
            auto_corrected_word_count += word_count
        else:
            # using backspace to delete
            # find where the pre_input and current_input diverge, no matter how many backspaces are used, start with
            # if pre_input.startswith(current_input):
            if len(pre_input) - len(current_input) == 1 and pre_input[:-1] == current_input:
                # if the backspace is used to delete the last character
                reformatted_input += '<'
                bsp_count += 1
                bsp_index_list.append(len(current_input) - 1)

                immediate_error_correction_count += 1
                # else:
                #     # add corresponding backspaces to delete the characters
                #     sequencial_backspaces_counts = len(pre_input) - len(current_input)
                #     for i in range(sequencial_backspaces_counts):
                #         reformatted_input += '<'
                #         bsp_count += 1
                #         bsp_index_list.append(len(current_input) + i)
            # for some cases, two or more backspaces occurred in one keystroke, we donot consider this case

            elif lev.distance(pre_input, current_input) == 1:  # let us assume no miss detected autocorrection
                # or multiple input in one keystroke
                # find if the last word in the reformatted_input is not the same
                # as the last word in the current_input
                # move the cursor to the middle of the sentence and use backspace to delete
                bsp_adjustments = get_bsp_adjustments(bsp_index_list, pre_input)
                for i in range(len(pre_input)):
                    if pre_input[i] != current_input[i]:
                        adjusted_index = i + 2 * bsp_adjustments[max(i - 1, 0)]
                        reformatted_input = reformatted_input[:adjusted_index] + '<' + reformatted_input[
                                                                                       adjusted_index:]
                        reformat_c_count += 1
                        bsp_count += 1
                        bsp_index_list.append(i)
                        break
                delayed_error_correction_count += 1
            elif pre_input.rsplit(' ', 1)[1] != current_input.rsplit(' ', 1)[1]:
                # Miss detected auto-correction
                auto_correct_flag = True
                reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + current_input.rsplit(' ', 1)[-1]
                word_before_modification = pre_input.rsplit(' ', 1)[-1]
                word_after_modification = current_input.rsplit(' ', 1)[-1]
                if_count, c_count, word_count = compute_if_c_count_for_auto_correction(
                    word_before_modification, word_after_modification)
                reformat_if_count += if_count
                reformat_c_count += c_count
                reformat_f_count += 1
                auto_corrected_word_count += word_count
                delayed_error_correction_count += 1
            else:
                # Move the cursor to the middle of the sentence and use backspace to delete
                bsp_adjustments = [0] * len(pre_input)
                bsp_running_total = 0
                for bsp_index in bsp_index_list:
                    if bsp_index < len(bsp_adjustments):
                        bsp_adjustments[bsp_index] = 1
                for i, adjustment in enumerate(bsp_adjustments):
                    bsp_running_total += adjustment
                    bsp_adjustments[i] = bsp_running_total
                for i in range(len(pre_input)):
                    if pre_input[i] != current_input[i]:
                        adjusted_index = i + 2 * bsp_adjustments[max(i - 1, 0)]
                        reformatted_input = reformatted_input[:adjusted_index] + '<' + reformatted_input[
                                                                                       adjusted_index:]
                        reformat_c_count += 1
                        bsp_count += 1
                        bsp_index_list.append(i)
                        break
                delayed_error_correction_count += 1
        pre_input = current_input

    return reformatted_input, reformat_if_count, reformat_c_count, auto_corrected_word_count, \
           reformat_f_count, auto_correct_flag, immediate_error_correction_count, delayed_error_correction_count, bsp_count


def levenshtein_with_details_and_indices(s1, s2):
    """
    Compute the Levenshtein distance between two strings (s1 and s2),
    and return the distance along with counts and indices of insertions, deletions, and substitutions.
    However, in this version, the logic that previously applied to s1 and s2 is switched.
    """
    rows = len(s2) + 1  # Switched
    cols = len(s1) + 1  # Switched
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Initialize the distance matrix
    for i in range(1, rows):
        distance_matrix[i][0] = i
    for i in range(1, cols):
        distance_matrix[0][i] = i

    # Compute Levenshtein distance
    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s2[row - 1] == s1[col - 1] else 1  # Switched
            distance_matrix[row][col] = min(distance_matrix[row - 1][col] + 1,  # Deletion
                                            distance_matrix[row][col - 1] + 1,  # Insertion
                                            distance_matrix[row - 1][col - 1] + cost)  # Substitution

    # Backtrack to find the number of insertions, deletions, and substitutions
    insertions, deletions, substitutions = 0, 0, 0
    insertion_indices, deletion_indices, substitution_indices = [], [], []
    row, col = rows - 1, cols - 1
    while row > 0 or col > 0:
        if row > 0 and col > 0 and s2[row - 1] == s1[col - 1]:  # Switched
            row -= 1
            col -= 1
        elif row > 0 and col > 0 and distance_matrix[row][col] == distance_matrix[row - 1][col - 1] + 1:
            substitutions += 1
            substitution_indices.append(row - 1)  # Changed to index in s2
            row -= 1
            col -= 1
        elif row > 0 and distance_matrix[row][col] == distance_matrix[row - 1][col] + 1:
            deletions += 1
            deletion_indices.append(row - 1)  # Changed to index in s2
            row -= 1
        elif col > 0 and distance_matrix[row][col] == distance_matrix[row][col - 1] + 1:
            insertions += 1
            insertion_indices.append(row - 1)  # Changed, indicates where in s2 the insertion in s1 should be
            col -= 1
        # Handle cases where we're at the first row or column
        elif row == 0 and col > 0:
            insertions += 1
            insertion_indices.append(row - 1)  # Changed, for clarity in context
            col -= 1
        elif col == 0 and row > 0:
            deletions += 1
            deletion_indices.append(row - 1)  # Changed, for clarity in context
            row -= 1

    distance = distance_matrix[-1][-1]
    return (distance, insertions, deletion_indices, insertion_indices, substitutions, substitution_indices)


def rebuild_committed_sentence(typed):
    committed = []
    for char in typed:
        if char != "<":
            committed.append(char)
        elif committed:
            committed.pop()  # Remove the last character due to backspace
    return ''.join(committed)


def calculate_C_INF(reference, committed):
    # Calculate INF using Levenshtein distance
    INF, insertions, deletion_indices, insertion_indices, substitutions, substitution_indices = levenshtein_with_details_and_indices(
        reference, committed)
    # Directly count matching characters in the same positions
    C = len(committed) - insertions

    return C, INF, deletion_indices, insertion_indices, substitution_indices


def compute_IF_from_indices(reference, typed, deletion_indices, substitution_indices):
    IF = 0
    typed_index = 0
    reference_index = 0
    for char in typed:
        if char == "<":
            # Backspace character, check if it's necessary
            if typed_index - 1 in deletion_indices or typed_index - 1 in substitution_indices:
                # The backspace was necessary to correct a mistake
                pass
            else:
                # The backspace was unnecessary, counting towards IF
                IF += 1
            # Adjust indices based on the mistake type and position
            if typed_index - 1 in deletion_indices:
                deletion_indices.remove(typed_index - 1)
            if typed_index - 1 in substitution_indices:
                substitution_indices.remove(typed_index - 1)
        else:
            typed_index += 1

    return IF


def simplify_typed_text(typed_text):
    """
    Simplify the typed text by applying backspaces ("<") and ignoring characters
    that are backspaced and then retyped at the same position.
    """
    delete_chars = []
    simplify_text = []
    bsp_count = 0
    for i in range(len(typed_text)):
        if typed_text[i] == "<":
            delete_chars.append(typed_text[i - 2 * bsp_count - 1])
            bsp_count += 1
        else:
            if bsp_count == 0:
                simplify_text.append(typed_text[i])
            else:
                # turn the order of  the delete_chars
                delete_chars = delete_chars[::-1]
                # remove the last bsp_count elements from the simplify_text
                simplify_text = simplify_text[:-bsp_count]
                correct_char = list(typed_text[i: i + bsp_count])
                correct_char_used = [False] * len(correct_char)
                for j in range(len(correct_char)):
                    if delete_chars[j] != correct_char[j]:
                        simplify_text.append(delete_chars[j])
                        simplify_text.append('<')
                        # simplify_text.append(correct_char[j])
                bsp_count = 0
                delete_chars.clear()
                simplify_text.append(typed_text[i])
    return ''.join(simplify_text)


def track_typing_errors(reference, typed):
    committed = rebuild_committed_sentence(typed)
    C, INF, deletion_indices, insertion_indices, substitution_indices = calculate_C_INF(reference, committed)
    # add 1 to every element to insertion_indices
    insertion_indices = [i + 1 for i in insertion_indices]
    # print("deletion_indices", deletion_indices)
    # print("substitution_indices", substitution_indices)
    # print("insertion_indices", insertion_indices)
    F = typed.count("<")
    simply_typed = simplify_typed_text(typed)
    # print("simply_typed", simply_typed)
    IF = compute_IF_from_indices(reference, simply_typed, deletion_indices, substitution_indices)
    return C, INF, IF, F


if __name__ == "__main__":
    pass
    reference_sentence = "the quick brown"
    user_input = "th quick brpown"
    input_stream = "th quxck<<<ick brpown"

    # reference_sentence = "the quick brown"
    # user_input = "the quikc brown"
    # input_stream = "the quikc brown"

    user_input = 'Was wondering if you and Natalie connected?'
    input_stream = 'Was c<wimedrting <<<<<<<<<<ondering if you and Natalie conce<<nected ?<<v< <?'
    reference_sentence = "Was wondering if you and Natalie connected?"
    # test_phrase(reference_sentence, user_input, input_stream)
    # reference_sentence = 'lähetä paperit minulle'
    # user_input = 'lähetä paperit minulm'
    # input_stream = 'lähet<tä paperit minulm'
    # test_phrase(reference_sentence, user_input, input_stream)

    # input_stream = 'I was planning on calling you after I swa revisionh<s <s<.eof'
    # user_input = 'I was planning on calling you after I swa revisions.eof'
    # reference_sentence = 'I was planning on calling you after I saw your revisions.eof'
    uncorrected_error_rate, corrected_error_rate = compute_error_rate(reference_sentence, user_input, input_stream)
    print("uncorrected error rate: ", uncorrected_error_rate)
    print("corrected error rate: ", corrected_error_rate)
