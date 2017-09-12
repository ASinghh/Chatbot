import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import os



# Load the data
lines = open('cornell movie-dialogs corpus\movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('cornell movie-dialogs corpus\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# The sentences' ids, which will be processed to become our input and target data.
conv_lines[:10]

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

print(len(questions))
print(len(answers))

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))
    
min_line_length = 2
max_line_length = 20


short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1
    



vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
            
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
threshold = 10
count = 0           
vocab_to_int = {}

word_num = 2
for word, count in vocab.items():
    if count >= threshold:
        vocab_to_int[word] = word_num
        word_num += 1
        
codes = ['<PAD>','<EOS>','<UNK>']
k = 0
for code in codes:
    vocab_to_int[code] = k
    k +=1
    
int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in vocab_to_int:
            ints.append(vocab_to_int['<UNK>'])
        else:
            ints.append(vocab_to_int[word])
    ints.append(1)
    for i in range(21-len(ints)):
        ints.append(0)
    questions_int.append(ints)

        
questions_int = np.transpose(np.array(questions_int))


answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in vocab_to_int:
            ints.append(vocab_to_int['<UNK>'])
        else:
            ints.append(vocab_to_int[word])
    ints.append(1)
    for i in range(21-len(ints)):
        ints.append(0)
    answers_int.append(ints)
    
    


cursor = 0      
answers_int = np.transpose(np.array(answers_int))

def batch_generator(batch_length,questions,answers):
    global cursor
    input_q = questions_int[:,cursor:cursor+batch_length]
    input_a = answers_int[:,cursor:cursor+batch_length]
    
    cursor += batch_length
    le = []
    for i in range(batch_length):
        le.append(21)
    return {
        encoder_inputs: input_q,
        encoder_inputs_length: le,
        decoder_targets: input_a
    }
import numpy as np
import tensorflow as tf


tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = len(vocab_to_int)
input_embedding_size = 512

encoder_hidden_units = 512
decoder_hidden_units = 1024

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
encoder_cell = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )
    
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
z  = []
for i in range(128):
    z.append(21)
num_units = 1024
decoder_cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units)
attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units, attention_states)#,
    #memory_sequence_length= z)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell1, attention_mechanism,
    attention_layer_size=num_units)




##this small helper function replaces all those huge loops in the last model
helper = tf.contrib.seq2seq.TrainingHelper(
    encoder_inputs_embedded, z, time_major=True)
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_final_state)
outputs, _ , _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
decoder_outputs = outputs.rnn_output

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
decoder_prediction = tf.argmax(decoder_logits, 2)




stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())
loss_track = []
max_batches = 3001
batches_in_epoch = 100

max_batches = 3001
batches_in_epoch = 100

try:
    for batch in range(max_batches):
        fd = batch_generator(128, questions_int,answers_int)
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            #for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
            #    print('  sample {}:'.format(i + 1))
            #    print('    input     > {}'.format(inp))
              #  print('    predicted > {}'.format(pred))
              #  if i >= 2:
                 #   break
           # print()

except KeyboardInterrupt:
    print('training interrupted')