import re 

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

def label2idx(label):
    label2idx = {
        "hap":0,
        "ang":1,
        "neu":2,
        "sad":3,
        "exc":0}

    return label2idx[label]