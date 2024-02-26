# Text Mining - Course Assignment 1: Lyrics Genre Classification

## Overview

This Python script is part of the Text Mining course's Assignment 1, focusing on Lyrics Genre Classification. The script performs sentiment analysis and calculates term and pairs frequency probabilities to classify lyrics into genres using unigram, bigram, and combined models.

## Dependencies

Make sure you have the necessary libraries installed:

- os
- nltk
- pandas
- sklearn
- matplotlib
- scipy
- statsmodels
- numpy

You can install these libraries using:

```bash
pip install nltk pandas scikit-learn matplotlib scipy statsmodels numpy
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script using the following command:

```bash
python your_script.py
```

Make sure to replace `your_script.py` with the actual name of your script.

## Directory Structure

```plaintext
/
|-- TM_CA1_Lyrics/
|   |-- Pop/
|   |   |-- lyrics_file1.txt
|   |   |-- lyrics_file2.txt
|   |   |-- ...
|   |-- Rock/
|   |   |-- lyrics_file1.txt
|   |   |-- lyrics_file2.txt
|   |   |-- ...
|   |-- Rap/
|   |   |-- ...
|-- test.tsv
|-- your_script.py
|-- README.md
```

Ensure your lyrics files are organized in the `TM_CA1_Lyrics` directory, and the test data is in `test.tsv`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- This script uses the Natural Language Toolkit (NLTK) library for tokenization and sentiment analysis.
- Special thanks to the course instructor for the assignment.

Feel free to customize this template to fit your project's specific details and requirements.
