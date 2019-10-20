# .ipynb 파일 코드를 붙여넣기한 .py 파일

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from scipy.special import expit as sigmoid

sentences = [[13, 100, 100, 100, 100, 74, 100, 47, 100, 16, 100, 100, 100, 100, 100, 26, 71, 100, 27, 21, 97, 100, 100, 100, 15], [13, 100, 100, 74, 20, 100, 100, 21, 13, 100, 100, 100, 14, 49, 39, 100, 100, 16, 13, 100, 14, 26, 100, 13, 100, 17, 100, 16, 13, 100, 16, 100, 27, 25, 13, 100, 20, 49, 13, 100, 23, 100, 15], [13, 100, 100, 100, 39, 62, 100, 36, 100, 100, 100, 100, 100, 100, 18, 100, 100, 16, 100, 26, 100, 27, 20, 13, 100, 100, 49, 23, 100, 36, 100, 100, 100, 100, 15], [26, 84, 19, 100, 100, 16, 100, 100, 23, 100, 27, 14, 13, 100, 74, 14, 26, 100, 13, 100, 100, 20, 13, 100, 14, 13, 100, 16, 100, 17, 13, 100, 16, 38, 100, 27, 15], [13, 100, 74, 28, 100, 100, 21, 100, 16, 100, 100, 17, 100, 100, 26, 42, 100, 45, 100, 17, 100, 100, 27, 15], [28, 100, 21, 100, 100, 100, 26, 18, 46, 91, 100, 100, 17, 100, 18, 13, 100, 16, 100, 17, 100, 82, 27, 15], [13, 100, 100, 100, 32, 19, 100, 16, 85, 100, 14, 100, 82, 13, 100, 17, 100, 100, 100, 100, 49, 28, 74, 26, 42, 100, 100, 17, 100, 100, 100, 100, 49, 100, 18, 13, 100, 100, 16, 100, 100, 27, 15], [100, 100], [100, 14, 13, 100, 74, 28, 100, 26, 91, 92, 100, 100, 33, 100, 18, 100, 100, 100, 17, 100, 13, 100, 16, 100, 27, 15], [13, 100, 100, 100, 14, 13, 100, 74, 14, 26, 22, 100, 20, 100, 100, 100, 30, 19, 100, 16, 100, 100, 100, 27, 15]]

word2idx={'START': 0, 'END': 1, 'man': 2, 'paris': 3, 'britain': 4, 'england': 5, 'king': 6, 'woman': 7, 'rome': 8, 'london': 9, 'queen': 10, 'italy': 11, 'france': 12, 'the': 13, ',': 14, '.': 15, 'of': 16, 'and': 17, 'to': 18, 'a': 19, 'in': 20, 'that': 21, 'is': 22, 'was': 23, 'he': 24, 'for': 25, '``': 26, "''": 27, 'it': 28, 'with': 29, 'as': 30, 'his': 31, 'on': 32, 'be': 33, ';': 34, 'at': 35, 'by': 36, 'i': 37, 'this': 38, 'had': 39, '?': 40, 'not': 41, 'are': 42, 'but': 43, 'from': 44, 'or': 45, 'have': 46, 'an': 47, 'they': 48, 'which': 49, '--': 50, 'one': 51, 'you': 52, 'were': 53, 'her': 54, 'all': 55, 'she': 56, 'there': 57, 'would': 58, 'their': 59, 'we': 60, 'him': 61, 'been': 62, ')': 63, 'has': 64, '(': 65, 'when': 66, 'who': 67, 'will': 68, 'more': 69, 'if': 70, 'no': 71, 'out': 72, 'so': 73, 'said': 74, 'what': 75, 'up': 76, 'its': 77, 'about': 78, ':': 79, 'into': 80, 'than': 81, 'them': 82, 'can': 83, 'only': 84, 'other': 85, 'new': 86, 'some': 87, 'could': 88, 'time': 89, '!': 90, 'these': 91, 'two': 92, 'may': 93, 'then': 94, 'do': 95, 'first': 96, 'any': 97, 'my': 98, 'now': 99, 'UNKNOWN': 100}

vocab_size = len(word2idx)
window_size = 2
learning_rate = 0.025
final_learning_rate = 0.0001
num_negatives = 5 # number of negative samples to draw per input word
epochs = 20
D = 50 # word embedding size
learning_rate_delta = (learning_rate - final_learning_rate) / epochs

def get_negative_sampling_distribution(sentences, vocab_size):
    word_freq = np.ones(vocab_size) * 1e-5
    word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
            
    # smoothing: 너무 높은 freq, 너무 낮은 freq의 갭을 줄여줌
    # eg. 0.71 ** (3/4) = 0.77 vs 0.07 ** (3/4) = 0.14
    p_neg = word_freq**0.75 

    # normalize it
    p_neg = p_neg / p_neg.sum()

    assert(np.all(p_neg > 0))
    return p_neg

###################################### params
W = np.random.randn(vocab_size, D) # input-to-hidden
W2 = np.random.randn(D, vocab_size) # hidden-to-output
# sampling 될 확률
p_neg = get_negative_sampling_distribution(sentences, vocab_size)

threshold = 1e-5
p_drop = 1 - np.sqrt(threshold / p_neg) # p_neg가 thr보다 크면: 
                                        # 1-작은값 = p_drop커짐
                                        # 즉, p_neg에 비례하여 p_drop커짐
        
###################################### train
def get_context(pos, sen, window_size):
    start = max(0, pos-window_size)
    end_ = min(len(sen), pos + window_size)
    context = []
    for ctx_pos, ctx_word_idx in enumerate(sen[start:end_], start=start):
        if ctx_pos == pos:
            continue
        context.append(ctx_word_idx)
    return context

def sgd(input_, targets, label, learning_rate, W, W2):
    # W[input_] shape: D <- word의 one hot encoding으로 
                            # lookup=dense vector 
    # W2[:,targets] shape: D x N 
    # activation shape: N
    print("input_:", input_, "targets:", targets)
    activation = W[input_].dot(W2[:, targets])
    prob = sigmoid(activation) 
    
    # return cost (binary cross entropy)
    cost = label * np.log(prob + 1e-10) + (1 - label) * \
    np.log(1 - prob + 1e-10)

    # gradients
    gW2 = np.outer(W[input_], prob - label)  # D x N
    gW = np.sum((prob - label)*W2[:, targets], axis=1)  # D
    
    # update weight
    W2[:, targets] -= learning_rate*gW2  # D x N
    W[input_] -= learning_rate*gW  # D
    return cost.sum()

###################################### 1st epoch
cost = 0
for sen in sentences:
    # 1 - p_drop = drop되지 않을 확률, 즉 p_drop 크면 w는 살아남지 못함
    # negative sample 갯수도 랜덤하게 추출
    sen = [w for w in sen if np.random.random() < (1-p_drop[w])]
    print('sentence', sen)
    print()
    if len(sen) < 2:
        continue
    # center words를 랜덤하게 선택 -> neg
    shuffle_words = np.random.choice(len(sen), size=len(sen), replace=False)
    for pos in shuffle_words:
        word = sen[pos]
        print('input', word)
        context_words = get_context(pos, sen, window_size)
        print('label data', context_words)
        neg_word = np.random.choice(vocab_size, p=p_neg)
        print('negative sample', neg_word)
        targets = np.array(context_words)
        
        # update W: input_=word, target=target,
        c = sgd(input_=word, targets=targets, label=1, learning_rate=learning_rate, W=W, W2=W2)
        cost += c
        c = sgd(neg_word, targets, 0, learning_rate, W, W2)
        cost += c
        print()
# decay learning rate
learning_rate -= learning_rate_delta