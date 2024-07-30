import re
import math
import numpy as np
from cleantext import clean
from bs4 import BeautifulSoup
from nltk import ngrams
from nltk.corpus import stopwords

MALE_WORDS, FEMALE_WORDS = [], []

with open("../gender list/de male.txt", "r") as file:
    MALE_WORDS = [word.strip() for word in file.readlines()]

with open("../gender list/de female.txt", "r") as file:
    FEMALE_WORDS = [word.strip() for word in file.readlines()]

GENDER_SET = set(MALE_WORDS) | set(FEMALE_WORDS)

def filter_female_pronouns(sentences):
    # filter out ambiguous pronouns that have multiple meanings more than genders
    results = []
    for sentence in sentences.split('.'):
        # matches a word followed by a space, followed by one of the specified German pronouns, where both the word and the pronoun are whole, distinct words
        # those words appear in the middle of the sentence meaning a polite form rather than gender
        pattern = r"\b(\w+)\s(?:Sie|Ihr|Ihrem|Ihren|Ihrer|Ihres)\b"
        matches = re.findall(pattern, sentence)
        if matches:
            sentence = re.sub(pattern, matches[0].split(' ')[0], sentence)
        results.append(sentence)
    return '.'.join(results)

def stop_word_removal(text, language): 
    token = text.split()
    if language == 'en':
        stop_words = stopwords.words('english')
    elif language == 'de':
        stop_words = stopwords.words('german')
    # keep gendered words
    stop_words = set(stop_words) - GENDER_SET
    cleaned = ' '.join([w for w in token if not w in list(stop_words)])
    return cleaned

def clean_text(text, language):
    # remove html tags
    sentences = BeautifulSoup(text).get_text(" ")
    if language == 'de':
        # filter ambuiguous pronouns
        sentences = filter_female_pronouns(sentences)
    # use package
    sentences = clean(sentences, no_emoji=True, lower=True,
        no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True,
        no_digits=True, no_currency_symbols=True, no_punct=True,
        replace_with_url='', replace_with_email='', replace_with_phone_number='',
        replace_with_number='', replace_with_digit='', replace_with_currency_symbol='', lang=language)
    # remove stop words
    sentences = stop_word_removal(sentences, language)
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', sentences).strip()
    return text_no_doublespace

def get_cooccurrences(review, window, eps=1e-6):
    # add a small epsilon to avoid division by zero
    data = {}
    n_grams = ngrams(review.split(), window)
    for grams in n_grams:
        pos = 1
        m = 0
        f = 0
        for w in grams:
            pos += 1
            if w not in data:
                data[w] = {"m": eps, "f": eps, 'female': [], 'male': []}

            if pos == (window+1)//2:
                if w in MALE_WORDS:
                    m = 1
                if w in FEMALE_WORDS:
                    f = 1
                if m > 0:
                    for t in grams:
                        if t != w:
                            if t not in data:
                                data[t] = {"m": eps, "f": eps, 'female': [], 'male': []}
                            data[t]['m'] += 1
                            data[t]['male'].append(w)
                if f > 0:
                    for t in grams:
                        if t != w:
                            if t not in data:
                                data[t] = {"m": eps, "f": eps, 'female': [], 'male': []}
                            data[t]['f'] += 1
                            data[t]['female'].append(w)
    return data

def get_cooccurrences_exp(review, window, Beta, eps=1e-6):
    n_grams = ngrams(review.split(), window)
    data = {}
    for grams in n_grams:
        pos = 0
        m = 0
        f = 0
        center = (window+1)//2
        center_word = grams[center]
        if center_word not in data:
            data[center_word] = {"m": eps, "f": eps}
        for w in grams:
            distance = abs(center - pos)

            pos += 1
            if distance == 0:
                continue

            if w not in data:
                data[w] = {"m": eps, "f": eps}

            if w in MALE_WORDS:
                m = m + pow(Beta, distance)

            if w in FEMALE_WORDS:
                f = f + pow(Beta, distance)

        data[center_word]["m"] = data[center_word]["m"]+m
        data[center_word]["f"] = data[center_word]["f"]+f

    return data

def word_dict(review):
    '''construct a dictionary of words and their counts'''
    words = review.split() 
    data_dict = {}
    for word in words:
        if word not in data_dict:
            data_dict[word] = 1
        else:
            data_dict[word] += 1
    return data_dict

def calculate_mean(data, key):
    '''Calculate mean score of a given key in a dictionary'''
    total = 0
    count = 0
    if data:
        for _, val in data.items():
            if key in val:
                total += val[key]
                count += 1

        if count > 0:
            return round(total/count, 4)
        else:
            return 0
    else:
        return 0

def cnt_words(data_dict):
    return sum(data_dict.values())

def cnt_unique_words(data_dict):
    cnt = 0
    for val in data_dict.values():
        if val > 0:
            cnt += 1
    return cnt

def cnt_cooc(word_gender, gender):
    sum_cooc = 0
    for k, v in word_gender.items():
        sum_cooc += v[gender]
    return sum_cooc

def gender_word_counts(review, gender, male_words='', female_words=''):
    gender_dict = {}
    words = review.split()
    if gender == 'male':
        gender_dict = {male: 0 for male in male_words}
        for word in words:
            for male in male_words:
                if male == word:
                    gender_dict[word] += 1
    elif gender == 'female':
        gender_dict = {female: 0 for female in female_words}
        for word in words:
            for female in female_words:
                if female == word:
                    gender_dict[word] += 1
    return {k: v for k, v in gender_dict.items() if v > 0}

def bias_records(word_gender, total_m_cooc, total_f_cooc):
    bias_record = {}
    for word, record in word_gender.items():
        m_cooc = record['m']
        f_cooc = record['f']
        word_m_prob = m_cooc / total_m_cooc
        word_f_prob = f_cooc / total_f_cooc
# 
        p_wg = word_m_prob / word_f_prob
        score = math.log(p_wg)
        rec = {'male cooc': m_cooc, 'female cooc': f_cooc, "bias_score": round(score, 3)}
        word_gender[word].update(rec)
        bias_record[word] = word_gender[word]

    return bias_record
    
def genbit_results(df, col, window_size=20, context_type='fixed', beta=0.95):
    '''construct a dataframe with all the bias features for a given column of text'''
    if context_type == 'fixed':
        df['word_gender'] = df[col].apply(lambda x: get_cooccurrences(x, window_size))
    elif context_type == 'infinite':
        df['word_gender'] = df[col].apply(lambda x: get_cooccurrences_exp(x, window_size, beta))
    df['word_dict'] = df[col].apply(lambda x: word_dict(x))
    df['total_words'] = df['word_dict'].apply(lambda x: cnt_words(x))
    df['total_unique_words'] = df['word_dict'].apply(lambda x: cnt_unique_words(x))

    df['total_m_cooc'] = df['word_gender'].apply(lambda x: cnt_cooc(x, 'm'))
    df['total_f_cooc'] = df['word_gender'].apply(lambda x: cnt_cooc(x, 'f'))

    bias_records_col = []
    for _, row in df.iterrows():
        bias = bias_records(row['word_gender'], row['total_m_cooc'], row['total_f_cooc'])
        bias_records_col.append(bias)
    df['bias_records'] = bias_records_col
    df['bias_score'] = df['bias_records'].apply(lambda x: calculate_mean(x, 'bias_score'))
    return df

def clean_calc_bias(df, col):
    col_new = col + '_clean'
    df[col_new] = df[col].apply(lambda x: clean_text(x, 'de'))
    return genbit_results(df, col_new)