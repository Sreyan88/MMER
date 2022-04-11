import textgrid
import torch
from torch import nn
import re

from transformers import (Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor)

def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""

    ind = 0
    str_interval = str(IntervalObject)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name

    return (pn, st, et)


def parse_textgrid(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList("words")
    words_list = list_words[0]
    
    result = []
    for ele in words_list:
        d = parse_Interval(ele)
        result.append(d)
    return result

def create_processor(model_name_or_path,vocab_file= None):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)

    if vocab_file:
        tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name_or_path,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    return Wav2Vec2Processor(feature_extractor, tokenizer)


def prepare_example(text,vocabulary_text_cleaner):

    # Normalize and clean up text; order matters!
    try:
        text = " ".join(text.split())  # clean up whitespaces
    except:
        text = "NULL"
    updated_text = text
    updated_text = vocabulary_text_cleaner.sub("", updated_text)
    if updated_text != text:
        return re.sub(' +', ' ', updated_text).strip()
    else:
        return re.sub(' +', ' ', text).strip()

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +",' ', text).strip()

    return text