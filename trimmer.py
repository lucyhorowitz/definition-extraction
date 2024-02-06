import re
import os
import spacy
import sys
from spacy_conll import init_parser
from spacy.language import Language

def remove_tags(text):
    text = text.replace('#todo', '')
    text = text.replace('#write_proof', '')
    text = text.replace('#', '')
    words = [word for word in text.split() if not word.startswith('https')]
    return ' '.join(words)

def remove_aliases(text=''):
    start_indices = [match.start() for match in re.finditer('---', text)]
    i = 0
    while i < len(start_indices):
        next_line = text.find('---', start_indices[i] + 4)
        text = text.replace(text[start_indices[i]:next_line +3], (next_line +3 - start_indices[i])* '#')
        i = i + 2
    text = text.replace('#', '')
    return text

def clean_links(text):
    start_indices = [match.start() for match in re.finditer('\[\[', text)]
    
    for index in start_indices:
        if text[index - 1] == '!':
            close_index = text.find(']', index)
            text = text.replace(text[index - 1:close_index + 3], (close_index + 3 - index)*'#')
        else:
            next_pipe = text.find('|', index)
            close_index = text.find(']', index)
            if next_pipe != -1:
                if next_pipe - close_index < 0:
                    text = text.replace(text[index:next_pipe + 1], (next_pipe - index + 1)*'#')
    text = text.replace('#', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    return text

# cleaning latex

def remove_simples(text):
    text = text.replace('$$', '$')
    text = text.replace('.$', '$.')
    text = text.replace('$', ' $ ')
    text = text.replace('**', ' ')
    text = text.replace('-', ' - ') #this one was key!
    return text

def remove_environments(text):
    start_indices = [match.start() for match in re.finditer('begin', text)]
    i = 0
    while i < len(start_indices):
        end_index = text.find('\end', start_indices[i] + 5)
        end_end = text.find('}', end_index)
        text = text.replace(text[start_indices[i]:end_end], (end_end - start_indices[i])* '#')
        i = i + 1
    text = text.replace('#', '')
    return text

def list_fix(text):
    text = text.replace('1.', '(i)')
    text = text.replace('2.', '(ii)')
    text = text.replace('3.', '(iii)')
    text = text.replace('4.', '(iv)')
    text = text.replace('5.', '(v)')
    text = text.replace('6.', '(vi)')
    text = text.replace('7.', '(vii)')
    text = text.replace('8.', '(viii)')
    text = text.replace('9.', '(ix)')
    text = text.replace('10.', '(x)')
    return text

def final_sweep(text):
    words = [word.replace('\\', '') for word in text.split() if not word.startswith('Helvetica')]
    text = ' '.join(words)
    return text

def clean_everything(text):
    text = remove_tags(text) #done first
    text = clean_links(text)
    text = remove_aliases(text)
    text = remove_environments(text)
    text = remove_simples(text)
    text = list_fix(text)
    text = final_sweep(text)
    return text


nlp = init_parser("en_core_web_sm", 'spacy')

# defining the pipeline component detextor
@Language.component('detextor')
def detextor(doc):
    dollar_indices = [index for index, token in enumerate(doc) if token.text == '$']
    while len(dollar_indices) > 1:
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[dollar_indices[0]:dollar_indices[1] + 1])
        dollar_indices = [index for index, token in enumerate(doc) if token.text == '$']
    return doc
#nlp.remove_pipe('detextor') #might need to add this back in if you run this code more than once
nlp.add_pipe('detextor', after='tagger')

file = sys.argv[1] # file to read
text = open(file).read()

with open('post_spacy.conllu', 'a') as f:
    clean_text = clean_everything(text)
    doc = nlp(clean_text)
    sents = [sent for sent in doc.sents] #i think i did this bc access often

    j = 1
    for sent in doc.sents:
        doc2 = nlp(sent.text)
        conll = doc2._.conll_str
        f.write('# sent_id = ' + str(j) + '\n')
        f.write('# sent_len = ' + str(len(doc2)) + '\n')
        f.write('# text = ' + sent.text + '\n')
        f.write(conll + '\n')
        j = j + 1