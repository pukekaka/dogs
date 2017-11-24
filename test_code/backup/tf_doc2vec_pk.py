import tensorflow as tf
import os
import string
import collections
import numpy as np
import pickle
from tensorflow.python.framework import ops
import zipfile
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# data_folder_name = 'temp'
data_folder_name = 'E:/Works/Data/samples/output_c'
# if not os.path.exists(data_folder_name):
#     os.makedirs(data_folder_name)

'''
Step0 : Variable Init
'''

window_size = 3
vocabulary_size = 7500
stops = []

# valid_words = ['incesi', 'movecxesi', 'pusheax', 'movediedi', 'bad', 'happy']
valid_words = ['incesi', 'movecxesi', 'pusheax', 'movediedi', 'testedxedx', 'call_memset']

embedding_size = 200
doc_embedding_size = 100
concatenated_size = embedding_size + doc_embedding_size

batch_size = 500
num_sampled = int(batch_size/2)
model_learning_rate = 0.001

generations = 100000
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

'''
Step1 : Loading Data
'''
# def load_data():
#     save_folder_name = 'temp'
#     pos_file = os.path.join(save_folder_name, 'train-pos.txt')
#     neg_file = os.path.join(save_folder_name, 'train-neg.txt')
#
#     pos_data = []
#     with open(pos_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             pos_data.append(line.encode('ascii', errors='ignore').decode())
#     f.close()
#     pos_data = [x.rstrip() for x in pos_data]
#
#     neg_data = []
#     with open(neg_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             neg_data.append(line.encode('ascii', errors='ignore').decode())
#     f.close()
#     neg_data = [x.rstrip() for x in neg_data]
#
#     texts = pos_data + neg_data
#     target = [1] * len(pos_data) + [0] * len(neg_data)
#
#     return (texts, target)
#
# texts, target = load_data()

# print('texts size', len(texts))
# print('target size', len(target))
# print('texts type', type(texts))
# print('target type', type(target))
# print(texts[:5])
# print(target[:5])

def load_data():
    save_folder_name = 'E:/Works/Data/samples/output_c'
    bbl_file = os.path.join(save_folder_name, 'basicblock_by_line')

    bbl_data = []
    with open(bbl_file, 'r', encoding='utf-8') as f:
        for line in f:
            bbl_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    bbl_data = [x.rstrip() for x in bbl_data]

    texts = bbl_data

    return texts

texts = load_data()


'''
Step2 : Normalization 
'''
# def normalize_text(texts, stops):
#     texts = [x.lower() for x in texts]
#     texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
#     texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
#     texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
#     texts = [' '.join(x.split()) for x in texts]
#
#     return (texts)
#
# texts = normalize_text(texts, stops)
#
# target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
# texts = [x for x in texts if len(x.split()) > window_size]
# assert(len(target) == len(texts))

# print('texts size', len(texts))
# print('target size', len(target))
# print(texts[:5])
# print(target[:5])


'''
Step3 : Build Dictionary
'''
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]

    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    return (word_dict)

word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

# print(word_dictionary['characters'])
# print(word_dictionary_rev[100])


'''
Step3-1 : Text to Numbers
'''
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)

text_data = text_to_numbers(texts, word_dictionary)

print(text_data[:5])
print(text_data[379])

'''
Step3-2 : valid example setting
'''
valid_examples = [word_dictionary[x] for x in valid_words]


'''
Step4 : Create Model
'''

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size], stddev=1.0 / np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1])
y_target = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)

final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])

loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = y_target,
                                     inputs = final_embed,
                                     num_sampled = num_sampled,
                                     num_classes = vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


'''
Step4-1 : Save Model
'''
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})


'''
Step5 : Run doc2vec Model
'''
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        # Pull out center word of interest for each window and create a tuple for each window
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i + window_size], rand_sentence[i + window_size]) for i in
                                range(0, len(rand_sentence) - window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)




sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size, method='doc2vec')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(train_step, feed_dict=feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))

    # Validation: Print some random words and top 5 related words
    if (i + 1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5  # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str)

    # Save dictionary + embeddings
    if (i + 1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'instruction_line.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'doc2vec_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))


