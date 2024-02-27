import os
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')


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


def freq_to_prob(dic_term_frequency, dic_pairs_frequency):
    total_terms = sum(dic_term_frequency.values())
    total_pairs = sum(dic_pairs_frequency.values())

    dic_term_prob = {term: count / total_terms for term, count in dic_term_frequency.items()}
    dic_pairs_prob = {pair: count / total_pairs for pair, count in dic_pairs_frequency.items()}

    return dic_term_prob, dic_pairs_prob


def calculate_probability(dic_term_prob, input_text):
    prob = 0.0
    tokens = word_tokenize(input_text)

    for token in tokens:
        prob += dic_term_prob.get(token, 0)

    return prob


def main():
    genres = ["Pop", "Rock", "Rap", "Metal", "Country", "Blues"]
    directory_path = "TM_CA1_Lyrics/"
    input_text = """
                How they dance in the courtyard
                Sweet summer sweat
                Some dance to remember
                Some dance to forget
                So I called up the Captain
                "Please bring me my wine"
                He said, "We haven't had that spirit here
                Since 1969
                """
    max_genre = None
    max_probability = float('-inf')
    for genre in genres:
        genre_directory = os.path.join(directory_path, genre)
        term_freq, pairs_freq = read_files_in_directory(genre_directory)
        term_prob, pairs_prob = freq_to_prob(term_freq, pairs_freq)

        result = calculate_probability(term_prob, input_text)
        print(f"Genre: {genre}, Probability: {result}")

        if result > max_probability:
                    max_probability = result
                    max_genre = genre

    print(f"The genre with the maximum probability is: {max_genre} \n with a probability of {max_probability}")


if __name__ == "__main__":
    main()
