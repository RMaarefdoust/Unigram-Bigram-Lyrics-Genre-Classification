# Text Mining - Course Assignment : Lyrics Genre Classification

## Overview

This Python script is part of the Text Mining course's Assignment 2, focusing on Lyrics Genre Classification. The script performs sentiment analysis and calculates term and pairs frequency probabilities to classify lyrics into genres using unigram, bigram, and combined models.

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
git clone https://github.com/RMaarefdoust/Unigram-Bigram-Lyrics-Genre-Classification
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script using the following command:

```bash
python MixedModel.py
```

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
|-- MixedModel.py
|-- README.md
```

Ensure your lyrics files are organized in the `TM_CA1_Lyrics` directory, and the test data is in `test.tsv`.

## Acknowledgments

- This script uses the Natural Language Toolkit (NLTK) library for tokenization and sentiment analysis.
- Special thanks to the course instructor for the assignment.
```

