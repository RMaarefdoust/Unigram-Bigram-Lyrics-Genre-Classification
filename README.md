
```markdown
# Text Genre Classification

This project focuses on text genre classification using unigram, bigram, and combined models. The genres include Pop, Rock, Rap, Metal, Country, and Blues. The classification is performed on a dataset of song lyrics.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- nltk
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- statsmodels

Install the required dependencies using the following command:

```bash
pip install nltk pandas numpy matplotlib scikit-learn scipy statsmodels
```

Additionally, download the NLTK punkt data:

```bash
python -m nltk.downloader punkt
```

## File Descriptions

- **MixedModel.py:** Main script for text genre classification.
- **test.tsv:** Test dataset in tab-separated values format.
- **TM_CA1_Lyrics/:** Directory containing song lyrics categorized by genre.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

2. Run the main script:

```bash
python 123.py
```

3. Follow the instructions and provide input when prompted.

## Results

The script will output the genre classification results based on the unigram, bigram, and combined models. It will also display F1-scores and classification reports. T-tests are performed to compare the models' performances.

## Example Input

The script provides an example input for testing the models. Replace it with your actual input when evaluating the models.

