# Importing required libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords




# Loading the Input Excel sheet into a DataFrame
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)


def extract_article_info(url):
    try:
        # Fetch webpage content
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find and extract article title
            title_element = soup.find('h1',class_="entry-title")
            article_title = title_element.get_text() if title_element else None
            
            # Find article elements based on HTML structure using class tags
            articles = soup.find_all(class_="td-post-content tagdiv-type")  # Adjust this based on the structure of the webpage
            
            # Extract text from articles
            article_text = ''
            for article in articles:
                # The article text is contained within <p> tags to avoid header and footer
                paragraphs = article.find_all('p')
                article_paragraphs = [p.get_text() for p in paragraphs]
                article_text += '\n'.join(article_paragraphs)
                # article_text += ' '.join([p.get_text() for p in paragraphs])
            
            return article_title, article_text
        else:
            print(f"Failed to fetch the webpage: {url}")
            return None, None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None, None

# Extract article info for each URL
extracted_data = []
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    title, text = extract_article_info(url)
    
    # Check if title and text are None
    if title is None or text is None:
        title, text = np.NaN, np.NaN  # Filling with 'NaN'
    
    # Concatenating title and text before appending to extracted_data
    full_text = f"{title} {text}" if title and text else ''  # Concatenate title and text
    extracted_data.append({'URL_ID': url_id, 'URL': url, 'Text': full_text})

# Create a DataFrame from the extracted data
sample_df = pd.DataFrame(extracted_data)

# Loading stopwords from text files
directory = 'StopWords-20231226T073510Z-001/StopWords'
STOPWORDS = set()
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as file:
        words = file.read().split()
        STOPWORDS.update(words)


# Function to clean text using stopwords
def clean_text(text):
    stop_words = STOPWORDS
    words = word_tokenize(text.lower())
    cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned_words)

# Loading positive and negative words from text files
with open('MasterDictionary-20231226T073510Z-001/MasterDictionary/positive-words.txt', 'r') as file:
    positive_words = set(file.read().split())

with open('MasterDictionary-20231226T073510Z-001/MasterDictionary/negative-words.txt', 'r') as file:
    negative_words = set(file.read().split())

# Function to calculate derived variables for sentiment analysis
def calculate_sentimental_variables(text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    total_words = len(words)
    polarity_score = (positive_score - negative_score) / max((positive_score + negative_score), 0.000001)
    subjectivity_score = (positive_score + negative_score) / max(total_words, 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to count syllables in a word with exceptions for "es" or "ed" endings
def syllable_count(word):
    word = word.lower()
    if word.endswith(('es', 'ed')):
        return 0
    count = max(1, sum(1 for char in word if char.lower() in "aeiou"))
    if word.endswith("e") and not word.endswith(("es", "ed")):
        count -= 1
    return count

# Function to calculate readability analysis variables
def calculate_readability_variables(text):
    sentences = sent_tokenize(text)
    words = clean_text(text).split()
    complex_words_count = sum(1 for word in words if syllable_count(word) > 2)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    percentage_complex_words = complex_words_count / max(len(words), 1)
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(words) / max(len(sentences), 1)
    word_count = len(words)
    syll_per_word = sum(syllable_count(word) for word in words) / max(len(words), 1)
    personal_pronouns_count = len(re.findall(r'\b(i|we|my|ours|us)\b', text.lower()))
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    return avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_words_count, word_count, syll_per_word, personal_pronouns_count, avg_word_length

output_data = []

# Assuming sample_df is your DataFrame containing 'URL' and 'Text' columns
for index, row in sample_df.iterrows():
    text = row['Text']
    
    # Calculate sentimental analysis variables
    positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentimental_variables(text)
    
    # Calculate readability analysis variables
    avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_words_count, word_count, syll_per_word, personal_pronouns_count, avg_word_length = calculate_readability_variables(text)
    
    output_data.append({
        'URL': row['URL'],
        'URL_ID': row['URL_ID'],
        'POSITIVE_SCORE': positive_score,
        'NEGATIVE_SCORE': negative_score,
        'POLARITY_SCORE': polarity_score,
        'SUBJECTIVITY_SCORE': subjectivity_score,
        'AVG_SENTENCE_LENGTH': avg_sentence_length,
        'PERCENTAGE_OF_COMPLEX_WORDS': percentage_complex_words,
        'FOG_INDEX': fog_index,
        'AVG_NUMBER_OF_WORDS_PER_SENTENCE': avg_words_per_sentence,
        'COMPLEX_WORD_COUNT': complex_words_count,
        'WORD_COUNT': word_count,
        'SYLLABLE_PER_WORD': syll_per_word,
        'PERSONAL_PRONOUNS': personal_pronouns_count,
        'AV_WORD_LENGTH': avg_word_length
    })

output_df = pd.DataFrame(output_data)
output_df.to_csv('Output_dataframe.csv')
output_df.head()


# Depedencies: Pycharm
