import os
import ssl
import nltk
import math
import random
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from statsmodels.stats import weightstats
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"

def colored_text(text, color):
    return f"{color}{text}{Colors.RESET}"

def read_files_in_directory(directory_path):
    dic_term_frequency = Counter()
    dic_pairs_frequency = Counter()

    for file in os.listdir(directory_path):
        with open(os.path.join(directory_path, file), 'r') as rfile:
            for line in rfile:
                current_line = line.strip()
                tokens = word_tokenize(current_line)

                for token in tokens:
                    dic_term_frequency[token] += 1

                pairs = list(zip(tokens[:-1], tokens[1:]))
                dic_pairs_frequency.update(pairs)

    return dic_term_frequency, dic_pairs_frequency

def freq_to_prob(dic_term_frequency, dic_pairs_frequency, smoothing=1e-5):
    total_terms = sum(dic_term_frequency.values())
    total_pairs = sum(dic_pairs_frequency.values())

    dic_term_prob = {term: (count + smoothing) / (total_terms + len(dic_term_frequency) * smoothing) for term, count in dic_term_frequency.items()}
    dic_pairs_prob = {pair: (count + smoothing) / (total_pairs + len(dic_pairs_frequency) * smoothing) for pair, count in dic_pairs_frequency.items()}

    return dic_term_prob, dic_pairs_prob

def calculate_unigram_probability(dic_term_prob, input_text):
    prob = 0.0
    tokens = word_tokenize(input_text)

    for token in tokens:
        prob += dic_term_prob.get(token, 0)

    return prob

def calculate_bigram_probability(dic_pairs_prob, input_text):
    prob = 0.0
    tokens = word_tokenize(input_text)
    pairs = list(zip(tokens[:-1], tokens[1:]))

    for pair in pairs:
        prob += dic_pairs_prob.get(pair, 0)

    return prob

def calculate_combined_probability(dic_term_prob, dic_pairs_prob, input_text, lambda_value):
    unigram_prob = calculate_unigram_probability(dic_term_prob, input_text)
    bigram_prob = calculate_bigram_probability(dic_pairs_prob, input_text)

    combined_prob = lambda_value * unigram_prob + (1 - lambda_value) * bigram_prob
    return combined_prob

def calculate_max_genre(input_text, genres, directory_path, model_type, lambda_value):
    max_genre = None
    max_probability = float('-inf')

    for genre in genres:
        genre_directory = os.path.join(directory_path, genre)
        term_freq, pairs_freq = read_files_in_directory(genre_directory)
        term_prob, pairs_prob = freq_to_prob(term_freq, pairs_freq)

        if model_type == "unigram":
            result = calculate_unigram_probability(term_prob, input_text)
        elif model_type == "bigram":
            result = calculate_bigram_probability(pairs_prob, input_text)
        else:
            result_unigram = calculate_unigram_probability(term_prob, input_text)
            result_bigram = calculate_bigram_probability(pairs_prob, input_text)
            result = lambda_value * result_unigram + (1 - lambda_value) * result_bigram

        print(f"Genre: {genre}, Probability ({model_type}): {result}")

        if result > max_probability:
            max_probability = result
            max_genre = genre

    return max_genre, max_probability

def calculate_t_test(data1, data2):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    df = ((std1**2 / n1) + (std2**2 / n2))**2 / (((std1**2 / n1)**2 / (n1 - 1)) + ((std2**2 / n2)**2 / (n2 - 1)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=df))
    return t_statistic, p_value

def evaluate_models_on_test_set(test_data, genres, directory_path):
    true_labels = []
    predictions_unigram = []
    predictions_bigram = []
    predictions_combined = []

    label_mapping = {genre: i for i, genre in enumerate(genres)}

    for index, row in test_data.iterrows():
        text = row['Text']
        genre = row['Genre']
        true_labels.append(label_mapping[genre])

        # Unigram Model
        model_type = "unigram"
        _, probability_unigram = calculate_max_genre(text, genres, directory_path, model_type, 1)
        predictions_unigram.append(label_mapping[_])

        # Bigram Model
        model_type = "bigram"
        _, probability_bigram = calculate_max_genre(text, genres, directory_path, model_type, 1)
        predictions_bigram.append(label_mapping[_])

        # Combined Model
        model_type = "combined"
        lambda_value = 0.5
        _, probability_combined = calculate_max_genre(text, genres, directory_path, model_type, lambda_value)
        predictions_combined.append(label_mapping[_])

    # Evaluate F1-score
    f1_unigram = f1_score(true_labels, predictions_unigram, average='weighted')
    f1_bigram = f1_score(true_labels, predictions_bigram, average='weighted')
    f1_combined = f1_score(true_labels, predictions_combined, average='weighted')

    print(f"F1-score (Unigram): {f1_unigram}")
    print(f"F1-score (Bigram): {f1_bigram}")
    print(f"F1-score (Combined): {f1_combined}")

    # Print classification report
    print(colored_text("Classification Report (Unigram):", Colors.RED))
    print(colored_text(classification_report(true_labels, predictions_unigram, zero_division=1), Colors.RED))

    print(colored_text("Classification Report (Bigram):", Colors.YELLOW))
    print(colored_text(classification_report(true_labels, predictions_bigram, zero_division=1), Colors.YELLOW))

    print(colored_text("Classification Report (Combined):", Colors.BLUE))
    print(colored_text(classification_report(true_labels, predictions_combined, zero_division=1), Colors.BLUE))

    print(colored_text("\nT-test Results:------------------------------", Colors.RED))
    # Perform t-test
    t_stat_unigram_bigram, p_value_unigram_bigram = ttest_rel(predictions_unigram, predictions_bigram)
    t_stat_unigram_combined, p_value_unigram_combined = ttest_rel(predictions_unigram, predictions_combined)
    t_stat_bigram_combined, p_value_bigram_combined = ttest_rel(predictions_bigram, predictions_combined)

    print(f"T-Test (Unigram vs Bigram): T-statistic = {t_stat_unigram_bigram}, p-value = {p_value_unigram_bigram}")
    print(f"T-Test (Unigram vs Combined): T-statistic = {t_stat_unigram_combined}, p-value = {p_value_unigram_combined}")
    print(f"T-Test (Bigram vs Combined): T-statistic = {t_stat_bigram_combined}, p-value = {p_value_bigram_combined}")

    # Perform t-tests using scipy.stats library
    t_statistic_unigram_vs_bigram, p_value_unigram_vs_bigram, _ = weightstats.ttest_ind(predictions_unigram, predictions_bigram)
    t_statistic_unigram_vs_combined, p_value_unigram_vs_combined, _ = weightstats.ttest_ind(predictions_unigram, predictions_combined)
    t_statistic_bigram_vs_combined, p_value_bigram_vs_combined, _ = weightstats.ttest_ind(predictions_bigram, predictions_combined)

    print(colored_text("\nT-test Results with scipy.stats lib::--------------------------------", Colors.YELLOW))
    print(f"T-statistic (Unigram vs Bigram): {t_statistic_unigram_vs_bigram}, P-value: {p_value_unigram_vs_bigram}")
    print(f"T-statistic (Unigram vs Combined): {t_statistic_unigram_vs_combined}, P-value: {p_value_unigram_vs_combined}")
    print(f"T-statistic (Bigram vs Combined): {t_statistic_bigram_vs_combined}, P-value: {p_value_bigram_vs_combined}")

    # Perform t-tests manually
    t_statistic_unigram_vs_bigram, p_value_unigram_vs_bigram = calculate_t_test(predictions_unigram, predictions_bigram)
    t_statistic_unigram_vs_combined, p_value_unigram_vs_combined = calculate_t_test(predictions_unigram, predictions_combined)
    t_statistic_bigram_vs_combined, p_value_bigram_vs_combined = calculate_t_test(predictions_bigram, predictions_combined)

    print(colored_text("\nT-test Results manually:----------------------------", Colors.BLUE))
    print(f"T-statistic (Unigram vs Bigram): {t_statistic_unigram_vs_bigram}, P-value: {p_value_unigram_vs_bigram}")
    print(f"T-statistic (Unigram vs Combined): {t_statistic_unigram_vs_combined}, P-value: {p_value_unigram_vs_combined}")
    print(f"T-statistic (Bigram vs Combined): {t_statistic_bigram_vs_combined}, P-value: {p_value_bigram_vs_combined}")

def main():
    genres = ["Pop", "Rock", "Rap", "Metal", "Country", "Blues"]
    directory_path = "TM_CA1_Lyrics/"

    # Split the data into training and validation (90:10 split for each genre)
    training_data = {}
    validation_data = {}

    for genre in genres:
        genre_directory = os.path.join(directory_path, genre)
        files = os.listdir(genre_directory)
        split_index = int(len(files) * 0.9)
        training_data[genre] = files[:split_index]
        validation_data[genre] = files[split_index:]

    # Choose lambda values to try
    lambda_values = [i / 100 for i in range(50)]

    # Find the optimal lambda on the validation set
    best_lambda = None
    best_combined_prob = float('-inf')
    second_best_combined_prob = float('-inf')
    lambda_values_plot = []
    combined_probabilities_plot = []

    # Initialize these variables to be used later
    best_term_prob = None
    best_pairs_prob = None
    second_best_term_prob = None
    second_best_pairs_prob = None

    for lambda_value in lambda_values:
        total_combined_prob = 0.0

        for genre in genres:
            term_freq, pairs_freq = read_files_in_directory(os.path.join(directory_path, genre))
            term_prob, pairs_prob = freq_to_prob(term_freq, pairs_freq)

            for file in validation_data[genre]:
                file_path = os.path.join(directory_path, genre, file)
                with open(file_path, 'r') as rfile:
                    text = rfile.read()
                    combined_prob = calculate_combined_probability(term_prob, pairs_prob, text, lambda_value)
                    total_combined_prob += combined_prob

        if total_combined_prob > best_combined_prob:
            second_best_combined_prob = best_combined_prob
            best_combined_prob = total_combined_prob

            second_best_term_prob = best_term_prob
            second_best_pairs_prob = best_pairs_prob

            best_term_prob = term_prob
            best_pairs_prob = pairs_prob

            best_lambda = lambda_value
        elif total_combined_prob > second_best_combined_prob:
            second_best_combined_prob = total_combined_prob
            second_best_term_prob = term_prob
            second_best_pairs_prob = pairs_prob

        lambda_values_plot.append(lambda_value)
        combined_probabilities_plot.append(total_combined_prob)

    # Plot the results
    plt.plot(lambda_values_plot, combined_probabilities_plot, marker='o')
    plt.title('Combined Probabilities vs. Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Combined Probabilities')
    plt.show()

    print(f"The optimal lambda is: {best_lambda}")
    print(f"The best combined probability on the validation set with optimal lambda is: {best_combined_prob}")
    print(f"The second best combined probability on the validation set is: {second_best_combined_prob}")

    # Example input text - Update this with the actual input text
    """input_text = How they dance in the courtyard
                    Sweet summer sweat
                    Some dance to remember
                    Some dance to forget
                    So I called up the Captain
                    "Please bring me my wine"
                    He said, "We haven't had that spirit here
                    Since 1969"""
    #input_text="A singer in a smokey roomA smell of wine and cheap perfumeFor a smile they can share the nightIt goes on and on and on and on."
    #input_text="I hate to turn up out of the blue, uninvitedBut I couldn't stay away, I couldn't fight itI had hoped you'd see my faceAnd that you'd be reminded that for me, it isn't over	Pop"
    input_text="This ain't a song for the broken-hearted No silent prayer for the faith-departed I ain't gonna be just a face in the crowd You're gonna hear my voice when I shout it out loud It's my life, it's now or never I ain't gonna live forever I just want to live while I'm alive It's my life My heart is like the open highway Like Frankie said I did it my way I just wanna live while I'm alive It's my life."

    # Unigram Model
    model_type = "unigram"
    max_genre_unigram, max_probability_unigram = calculate_max_genre(input_text, genres, directory_path, model_type, 1)
    print(colored_text(f"The genre with the maximum probability (Unigram) is: {max_genre_unigram} \n with a probability of {max_probability_unigram}",Colors.YELLOW))

    # Bigram Model
    model_type = "bigram"
    max_genre_bigram, max_probability_bigram = calculate_max_genre(input_text, genres, directory_path, model_type, 1)
    print(colored_text(f"The genre with the maximum probability (Bigram) is: {max_genre_bigram} \n with a probability of {max_probability_bigram}",Colors.BLUE))

    # Combined Model
    model_type = "combined"
    lambda_value = 0.5
    max_genre_combined, max_probability_combined = calculate_max_genre(input_text, genres, directory_path, model_type, lambda_value)
    print(colored_text(f"The genre with the maximum probability (Combined) is: {max_genre_combined} \n with a probability of {max_probability_combined}",Colors.RED))

    # Load test data
    test_data_path = "test.tsv"
    test_data = pd.read_csv(test_data_path, delimiter='\t')

    # Evaluate models on the test set
    evaluate_models_on_test_set(test_data, genres, directory_path)

if __name__ == "__main__":
    main()
