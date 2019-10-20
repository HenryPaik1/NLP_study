# .ipynb 코드를 붙여넣기 한 .py 파일
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Embedding, dot, Reshape
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

corpus_raw = ['He is the king', 'The king is royal', 
              'she is the queen', 'the queen is royal']
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(corpus_raw)
tokenizer.word_index

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}
vocab_size = len(word2id)

wids = tokenizer.texts_to_sequences(corpus_raw)

############################# negative sampling by keras
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size,\
                        window_size=4) for wid in wids]

# pair: core words(input)와 context words(output) 쌍
# label: 정답1 오답0, 오답은 negative sampling된 단어와 내적하여 계산
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i, elem in enumerate(skip_grams):
    target = np.array(list(zip(*elem[0]))[0], dtype='int32')
    #print(pair_first_elem)
    context = np.array(list(zip(*elem[0]))[1], dtype='int32')
    labels = np.array(elem[1], dtype='int32')
    Y = labels
    
############################# modeling w2v by keras functional API
# 임베딩 차원(dense matrix 차원)
embed_size=4

target_input = Input(shape=(1,), name='target_input')
context_input = Input(shape=(1,), name='context_input')

# 핵심: 임베딩 레이어는 하나만 사용. 논문 구현의 여러가지 practical한 구현 방법 중 하나.
embedding_layer = Embedding(vocab_size, embed_size, input_length=1, name='embedding_layer')

#encode target word
target_encoded = embedding_layer(target_input)
target_encoded = Reshape((embed_size, 1))(target_encoded)

#encode context word    
context_encoded = embedding_layer(context_input)
context_encoded = Reshape((embed_size, 1))(context_encoded)

#dot product two words: 내적 후 미니배치 size 24를 모두 flatten시킴
dot_product = dot([target_encoded, context_encoded], axes=1)
dot_product = Reshape((1,))(dot_product)

#normalize
output_layer = Dense(1, activation='sigmoid', name='output_layer')(dot_product)

# compile model
main_model = Model(inputs=[target_input, context_input], outputs=[output_layer])
main_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
main_model.summary()

# visualize graph
SVG(model_to_dot(main_model).create(prog='dot', format='svg'))

############################# train
main_model.fit(x = [target, context], y=labels, batch_size=8, shuffle=True, verbose=2, epochs=10000)

############################# embedding layer
W = main_model.layers[2].get_weights()[0]

############################# visualize
pca = PCA(n_components=2)
Wp = pca.fit_transform(W)
p = Wp[6,:] - Wp[3,:] + Wp[5,:] # similarity 
x = Wp[:,0]
y = Wp[:,1]

%matplotlib inline
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(p[0],p[1])

n = word2id.keys()
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))