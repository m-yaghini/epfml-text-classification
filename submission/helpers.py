import re
import numpy as np
import csv
# You need to download the stopwords using the nltk downloader, more information in the README file.
from nltk.corpus import stopwords


def clean_str_regex(string):
    """
    Tokenization/string cleaning for the data files
    The original is taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    We modified it to remove the expression such as "'m", "'s", instead of representing them as "am", "is",
    please see the report for full explanation.
    We also remove all digits and some punctuation marks and transform all characters to lower case.

    ---> Input string :  A single line of the text file passed as a string

    ---> Output RETURN : The same string filtered with the regular expressions
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"^\d+", "", string)
    string = re.sub(r"\'m", "", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"url", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\d*", "", string)
    return string.strip().lower()

def remove_stop_words(string, stop_words_list):
    """
    Remove all stopwords from the given string accordlingly to the list of stopwords passed as parameters.

    ---> Input string : A single line of the text file passed as a string
    ---> Input stop_words_list : The list of the stop words which must be removed from the text

    ---> Output RETURN : The same string as passed in input without the stopwords.
    """

    words = [str(w.lower()) for w in string.split() if w not in stop_words_list]
    return " ".join(words)

def clean_str(string, stop_words_list):
    """
    Perform both cleaning operations defined above :
    First, we filter the string using the regular expresssions,
    then we remove the stop words from the string

    ---> Input string : A single line of the text file passed as a string
    ---> Input stop_words_list : The list of the stop words which must be removed from the string

    ---> Output RETURN : The initial string cleaned and ready to be used for the embeddings.
    """

    string = clean_str_regex(string)
    string = remove_stop_words(string, stop_words_list)
    return string


def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    """
    Loads data from files, splits the data into single lines and clean each line with the cleaning functions
    Then generates the labels as a vector with shape 1x2 where the first column is the probability of being
    a negative tweet and the second, the probability of being a positive tweet. Since the labels are known
    for the training data, we simply put a 0 or 1 in the corresponding column.
    Returns the cleaned sentences and labels for the training set and returns  the split sentences for the testing set.

    ---> Input positive_data_files : The file containing the positive tweets typically train_pos.txt
    ---> Input negative_data_files : The file containing the negative tweets typically train_neg.txt
    ---> Input test_data_files     : The file containing the test data, it is test_data.txt

    ---> Output RETURN : A list which contains the following elements :
                         - train  : The training set which will be passed as input for the model.
                         - labels : The labels corresponding to the tweets of the training data.
                         - test   : The test set which must be passed to the trained model.
    """

    # Loads the data from the files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open(test_data_file, "r").readlines())
    test_examples = [s.strip() for s in test_examples]

    # Creates the set of stop words using the one from the nltk library, for convinience purpose, we have copied the
    # set of words here so you do not have to download this external library. We also made some slight modifications
    # to the original set by adding punctuation signs, some special characters and frequent terms specific to our
    # data set (user, rt, url). We also remove some words from the stop words list such as the negation and words
    # like "what", "which" since those words may be significant
    stop_words_set = { 'same', 'we', 'd', ',', 'if', 'that', 'but', 'very', 'other', 'doing', 'for', 'under',
                       'through', 'in', 'than', 'further', '.', 'does', 'the', 'his', 'too', 'again', '``', 'during',
                       'hers', 'were', 'them', 'was', 'him', 'yourself', ':', 'had', 'ourselves', 'himself', 'between',
                       'her', 'on', 'any', 'over', 'you', 'being', '(', 'all', 'i', 'once', 'by', 'about', 'then',
                        'up', 'some', 'themselves', 'their', '{', 'it', 'my', 'be', 'how', 'have', 'these', 'because',
                       'before', 'they', 've', 'do', '%', 'or', '--', 'an', '&', 'yourselves', 'so', 'am', 'she', '[',
                       'above', 'few', 'itself', 'ain', 'those', 'own', 'just', 'there', '@', 'off', 'to', 'should',
                       'll', 'this', 'who', 'will', 'me', '}', 'while', 'shan', 'its', '>', 'don', 'no', 're', "''",
                       'our', '—', '"', 'a', 'ours', 'has', 'where', 'as', '<', 'at', 'into', ';', 'of', ']', 'won',
                       'down', 'been', 'below', 'having', 'why', 'yours', 'out', 'only', 'ma', ')', 'he', 'until',
                       "'", 'url', 'theirs', 'here', 'such', 'y', 'now', 'did', 'both', 'o', 'and', 's', 'can', 'each',
                       'is', 'm', 'u', 'most', 'herself', 'with',  'rt', '#', 'myself', '-', 'more', 'your', 'whom',
                       'are', '*', 't', 'from', 'user', '•', 'after'}

    stop_words_set.remove('what')
    stop_words_set.remove('when')
    stop_words_set.remove('which')
    stop_words_set.remove('why')

    # Creates the training set and cleans both data sets
    train = positive_examples + negative_examples
    train = [clean_str(sent, stop_words_set) for sent in train]
    test = [clean_str(sent, stop_words_set) for sent in test_examples]

    # Generates the labels for the training set
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)

    with open('data/train_cleaned.txt', 'w') as f:
        lines_seen = set()
        for sent in train:
            if sent not in lines_seen:
                f.write(sent)
                lines_seen.add(sent)

    train = list(open('data/train_cleaned.txt', 'r').readlines())

    return [train, labels, test]

def create_csv_file(predictions):
    '''
    Creates the csv file for the submission on Kaggle from the predicitons produced by the model.
    The predictions must be passed as a list of results already formatted for Kaggle (1 and -1).

    ---> Input predictions : The predictions created by the model
    ---> Output : Produces a cvs file named submission.csv in the current directory
    '''

    with open('submission.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        sub_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        index = 0
        sub_writer.writeheader()
        for result in predictions:
            index += 1
            sub_writer.writerow({'Id': str(index), 'Prediction': str(result[1])})
    print("Submission file created")