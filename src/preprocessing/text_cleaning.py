import re
import emoji

def clean_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    return text
