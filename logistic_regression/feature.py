import sys

import csv
import numpy as np

VECTOR_LEN = 300  # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(
        file, delimiter="\t", comments=None, encoding="utf-8", dtype="l,O"
    )
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(
        file, comments=None, encoding="utf-8", dtype=f"U{MAX_WORD_LEN},l"
    )
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter="\t")
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def get_bag_of_words(data, word_dict):
    data_bog = np.zeros((data.shape[0], len(word_dict)))
    data_label = np.zeros(data.shape[0])
    for i, (label, review) in enumerate(data):
        data_label[i] = label
        for word in review.split():
            if word in word_dict:
                data_bog[i][word_dict[word]] = 1
    return np.hstack((data_label[:, None], data_bog))


def get_word_embeddings(data, word_dict):
    data_bog = np.zeros((data.shape[0], VECTOR_LEN))
    data_label = np.zeros(data.shape[0])
    for i, (label, review) in enumerate(data):
        data_label[i] = label
        count = 0.0
        for word in review.split():
            if word in word_dict:
                count += 1
                data_bog[i] += word_dict[word]
        data_bog[i] = data_bog[i] / count
    return np.hstack((data_label[:, None], data_bog))


def save_formatted_data(data, out_file, feature_flag):
    with open(out_file, "w") as f:
        for row in data:
            if feature_flag == 1:
                f.write(f"{int(row[0])}")
                for feature in row[1:]:
                    f.write(f"\t{int(feature)}")
            elif feature_flag == 2:
                f.write(f"{row[0]:.06f}")
                for feature in row[1:]:
                    f.write(f"\t{feature:.06f}")
            f.write("\n")


def main(
    train_input,
    validation_input,
    test_input,
    dict_input,
    feature_dictionary_input,
    formatted_train_out,
    formatted_validation_out,
    formatted_test_out,
    feature_flag,
):
    train_data = load_tsv_dataset(train_input)
    validation_data = load_tsv_dataset(validation_input)
    test_data = load_tsv_dataset(test_input)

    if feature_flag == 1:
        word_dict = load_dictionary(dict_input)
        func = get_bag_of_words
    elif feature_flag == 2:
        word_dict = load_feature_dictionary(feature_dictionary_input)
        func = get_word_embeddings
    else:
        print("Provide correct feature flag!")

    formatted_train_data = func(train_data, word_dict)
    formatted_validation_data = func(validation_data, word_dict)
    formatted_test_data = func(test_data, word_dict)

    save_formatted_data(formatted_train_data, formatted_train_out, feature_flag)
    save_formatted_data(
        formatted_validation_data, formatted_validation_out, feature_flag
    )
    save_formatted_data(formatted_test_data, formatted_test_out, feature_flag)


if __name__ == "__main__":
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]

    dict_input = sys.argv[4]
    feature_dictionary_input = sys.argv[5]

    formatted_train_out = sys.argv[6]
    formatted_validation_out = sys.argv[7]
    formatted_test_out = sys.argv[8]

    feature_flag = int(sys.argv[9])
    main(
        train_input,
        validation_input,
        test_input,
        dict_input,
        feature_dictionary_input,
        formatted_train_out,
        formatted_validation_out,
        formatted_test_out,
        feature_flag,
    )
