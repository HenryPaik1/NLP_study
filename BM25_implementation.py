# 참조: http://lixinzhang.github.io/implementation-of-okapi-bm25-on-python.html

from itertools import chain
import math
from collections import defaultdict
import gensim
from gensim import corpora

# dataset: product titles
tokens = ['i9s Tws Headphone Wireless Bluetooth 5.0 Earphone Mini Earbuds With Mic Charging Box Sport Headset For Smart Phone',
 'Roreta AUX 3.5mm Jack Bluetooth Receiver Car Wireless Adapter Handsfree Call Bluetooth Adapter Transmitter Auto Music Receiver',
 'kebidu Bluetooth 5.0 Receiver Car Kit MP3 Player Decoder Board Color Screen FM Radio TF USB 3.5 Mm AUX Audio For Iphone XS',
 'VIKEFON Bluetooth 5.0 Audio Receiver Transmitter Mini Stereo Bluetooth AUX RCA USB 3.5mm Jack For TV PC Car Kit Wireless Adapter',
 'New BLON BL-03 10mm Carbon Diaphragm Dynamic Driver In Ear Earphone HIFI DJ Running Sport Earphone Earbuds Detachable 2PIN Cable',
 'i7s TWS Mini Wireless Bluetooth 5.0 Earphones In-Ear Stereo Earbuds Sports Handsfree Headset Mic Binaural call For Xiaomi iPhone',
 'Original For Apple Airpods 1 2 Wireless Bluetooth Earphone Case Colorful Candy For Apple AirPods New PC Hard Cute Cover Box Case',
 'Magnetic Bluetooth Wireless Stereo Earphone Sport Headset For iPhone X 7 8 Samsung S8 Xiaomi Huawei Waterproof Earbuds With Mic',
 'Hot Sale I7s TWS Bluetooth Earphone Stereo Earbud Wireless Bluetooth Earphones In-ear Headsets For All Smart Phone',
 'Roreta Dual Drive Stereo Wired earphone In-Ear Sport Headset With Mic mini Earbuds Earphones For iPhone Samsung Huawei Xiaomi']

DocLen = [len(sen) for sen in tokens_ls]
docTotalLen = sum([len(sen) for sen in tokens_ls])
n_docs = len(tokens_ls)
docAvgLen = docTotalLen/n_docs
DF = defaultdict(int)
DocIDF = {}
DocTF = []

# 각 doc별 단어의 nomalized freq 계산
for doc in tokens_ls:
    bow = dict([(term, freq*0.1/len(doc))for term, freq in dictionary.doc2bow(doc)])
    DocTF.append(bow)
    for term, tf in bow.items():
        DF[term] += 1
# >> bow
# {26: 0.005,
#  27: 0.005,
# ...
#  306: 0.005}

# IDF 게산
## IDF는 단어마다 정해짐: 상수        
for term in DF:
    DocIDF[term] = math.log((n_docs-DF[term]+0.5) / (DF[term]+0.5))

# query에 대한 각 Doc의 BM25 scoring 계산
query = tokens_ls[0]
query_bow = dictionary.doc2bow(query)
scores = []

k1 = 1.5; b = 0.75;
for doc_idx, doc_tf in enumerate(DocTF): # 모든 Doc 순회, doc_tf는 documents 내 단어들의 frequency
    commonTerms = set(dict(query_bow).keys()) & set(doc_tf.keys())
    tmp_score = []
    doc_terms_len = DocLen[doc_idx]
    
    for term in commonTerms:
        upper = ((k1 + 1) * doc_tf[term])
        L = doc_terms_len/docAvgLen
        below = (k1 * (1 - b + b * L))
        tmp_score.append(DocIDF[term] * upper / below)

# 유사도 top k 확인
tok_k = 5
print(' '.join(query))
for idx in np.argsort(scores)[::-1][1:top_k]: # idx0은 자기 자신이므로 제외
    print(idx)
    print(' '.join(tokens_ls[idx]))        

