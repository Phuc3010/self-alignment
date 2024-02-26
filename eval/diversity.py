from evaluate import load
import torch
from fast_bleu import BLEU, SelfBLEU


# Perplexity
model_id = "gpt2"
predictions = None
# perplexity = load("perplexity", module_type="metric")
# results = perplexity.compute(predictions=predictions, model_id=model_id)

from itertools import chain

__all__ = ["ngrams"]

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def ttr(texts, tokenizer):
    encodings_dict = tokenizer(texts)
    input_ids = encodings_dict['input_ids']
    num_unique = [len(set(ele)) for ele in input_ids]
    num_tokens = [len(ele) for ele in input_ids]
    ttr = sum([unique_len/total_len for unique_len, total_len in zip(num_unique, num_tokens)])/len(num_unique)
    return ttr

def distinct_n(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

# Semantic Diversity
def per_input_diversity():
    pass

def cross_input_diversity():
    pass

# Syntactic Diversity

if __name__=="__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
    # tokenizer.batch_encode_plus()
    texts = ["I'm I'm", "hello there"]
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
    # print(ttr(texts, tokenizer))
    # sentence = "hello there"
    self_bleu = SelfBLEU(, weights)
    print(distinct_n(texts, n=2))