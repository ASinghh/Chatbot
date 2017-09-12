import pandas as pd
import numpy as np
import time
import re
##The data is seperated in various tables, we need to utilize the matadata in
##in the conversation file and the actual  sentences in the linefile
text = open('cornell movie-dialogs corpus\movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
metadata = open('cornell movie-dialogs corpus\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
##Each entry is represented by a new line, so we split on new lines
##Every text has an ID, we need to develop a dictionary

text_id_dictionary = {}
for i in text :
    split = i.split(' +++$+++ ')##Mind that there are spaces between and after the plus signs
    if len(split) == 5:
        text_id_dictionary[split[0]] = split[4]
        
##the matadata for each conversation, gives a list of ids, these should be 
## colleceted
convid_list = []
for i in metadata :
     split= i.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
     convid_list.append(split.split(','))
    
    
##So what does this mean?? each list in the list is a conversation
##Each sublist has ids of the sentences that were said, and they are in sequence
##To feed into the sequence to sequence model, we need to break down the dataset
##into initiations and replies.We put initiation in and try to predict the replies


initials = []
replies = []
for i in convid_list :
    for j in range(len(i)-1):
        initials.append(text_id_dictionary[i[j]])
        replies.append(text_id_dictionary[i[j+1]])
        
##take a good look at the two list, what do you see??replies is the initial 
##list shifted by one element.Confused?? We are actually interested in the 
##sequence,SEQ--2--SEQ model it is !


##Now we are left with data with a lot of unnecessary variations
##these variaotions dont necessary add any meaning and are mostly conventional


def format(text):
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

##for help on the replacement code, see http://lzone.de/blog/Python-re.sub-Examples
##lets use the function

format_initials = []
for i in initials:
    format_initials.append(format(i))
    

format_replies = []
for i in replies:
    format_replies.append(format(i))
    
##Take one element from the two lists to see what we did

##very large question or very short answers create problems
##for the time being we will keep the questions between 2 -20
## we can change this later
## we will sort 1st with respect to questions and later for answers

sort_initials1 = []
sort_replies1 = []

j = 0
for i in format_initials:
    if len(i.split()) >= 2 and len(i.split()) <= 20:
        sort_initials1.append(i)
        sort_replies1.append(format_initials[j])
        j = j+1
        
## removed all iniitals beyond the range and the respective answers

## now we do the same with the replies and their respective initials
sort_initials = []
sort_replies = []

j = 0
for i in sort_replies1:
    if len(i.split()) >= 2 and len(i.split()) <= 20:
        sort_replies.append(i)
        sort_initials.append(sort_initials1[j])
    j += 1
    
##make disctionary of most common words

vocab = {}
for i in sort_initials:
    for k in i.split():
        if k not in vocab:
            vocab[k] = 1
        else:
            vocab[k] += 1
            
for j in sort_replies:
    for l in j.split():
        if l not in vocab:
            vocab[l] = 1
        else:
            vocab[l] += 1
                 
##Vocab will have some very rare words, we can replace them by unknown.
##I have taken some initial guesses and tried to keep the vocan size near
## 10000(just a random number)
lim = 8
word_id = {}

counter = 0
for i, j in vocab.items():
    if j >= lim:
        word_id[i] = counter
        counter += 1
        
markers = ['<PAD>','<EOS>','<UNK>','<GO>']
for i in markers:
    word_id[i] = len(word_id)+1
        
id_word = {i: j for j, i in word_id.items()}

##At every end of replies, we will mark the end by EOS
for i in range(len(sort_replies)):
    sort_replies[i] += ' <EOS>'

##we need to convert all text to integer to feeed in the embedding lookup
initials_int = []
for i in sort_initials:
    list = []
    for k in i.split():
        if k not in word_id:
            list.append(word_id['<UNK>'])
        else:
            list.append(word_id[k])
    initials_int.append(list)
#doing the same for replies

replies_int = []
for i in sort_replies:
    list = []
    for k in i.split():
        if k not in word_id:
            list.append(word_id['<UNK>'])
        else:
            list.append(word_id[k])
    replies_int.append(list)
    
##We will sort once more for increasing lenghts and then would be done with
##data preprocessing

Initials = []
Replies = []

for k in range(1, 21):
    for i in enumerate(initials_int):
        if len(i[1]) == k:
            Initials.append(initials_int[i[0]])
            Replies.append(replies_int[i[0]])

###########################################################################
############################# THE MODEL ###################################
############################ EL MODELO ####################################
########################### DAS MODEL #####################################
########################## LE MODELE ######################################
#########################   该模型   #######################################
########################    モデル   ########################################
#######################   आदर्श    #########################################
###########################################################################


# We can take the straight approach of defining everything on th graph #
# But modular approach has been proved to be better in all kinds of    #
# integrated systems, from space crafts to phones                      #

### Input definations, you need to decide over these#############
import tensorflow as tf
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    ##intials are fed to this
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    ## replies are fed here
    lr = tf.placeholder(tf.float32, name='learning_rate')
    ## you can fix this, but what about little flexibility mate
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    ## just in case we want to play arround with dropout layer
    return inputs, targets, lr, keep_prob

############################ Architecture #################################
# We will feed the input with EOS tag at end of each statement
#All statements would be padded to mke them 20 in length
#Thebiderictional lstm cell will result in a thought vector.
# We feed the thougth vector to the decoder 
#the decoder is having attention mechanism
# we also feed the target, by laging it by 1 position and add a GO tag 
# at the start
#Attention mechanism is a bit complicated, if you dont get it, you can treat
# it as a black box for now
# just focus on the shapes, input and  output

def dec_input_f(targets, word_id, batch_size):
    i = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], word_id['<GO>']), i], 1)
    return dec_input

#Removed the last word and added GO tag at the begning, what about the last 
#word?, well input is laggged by 1 step, so it is not required
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
def encoder(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    

##so, to construct a biderictional RNN, we 1st create a basic LSTMcell
## We use that cell as forward and backward layer with  dropout
## we only take the encoder sate, it is a tuple
def decoding_layer_train(enc_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    '''Decode the training data'''
    
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(enc_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                              train_decoder_fn, 
                                                              dec_embed_input, 
                                                              sequence_length, 
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)

def decoding_layer_infer(enc_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    '''Decode the prediction data'''
    
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn, 
                                                                         enc_state[0], 
                                                                         att_keys, 
                                                                         att_vals, 
                                                                         att_score_fn, 
                                                                         att_construct_fn, 
                                                                         dec_embeddings,
                                                                         start_of_sequence_id, 
                                                                         end_of_sequence_id, 
                                                                         maximum_length, 
                                                                         vocab_size, 
                                                                         name = "attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                infer_decoder_fn, 
                                                                scope=decoding_scope)
    
    return infer_logits###check why
    
def decoding_layer(dec_embed_input, dec_embeddings, enc_state, vocab_size, sequence_length, rnn_size,
                   num_layers, word_id, keep_prob, batch_size):
    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''
    
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                vocab_size, 
                                                                None, 
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)

        train_logits = decoding_layer_train(enc_state, 
                                            dec_cell, 
                                            dec_embed_input, 
                                            sequence_length, 
                                            decoding_scope, 
                                            output_fn, 
                                            keep_prob, 
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(enc_state, 
                                            dec_cell, 
                                            dec_embeddings, 
                                            word_id['<GO>'],
                                            word_id['<EOS>'], 
                                            sequence_length - 1, 
                                            vocab_size,
                                            decoding_scope, 
                                            output_fn, keep_prob, 
                                            batch_size)

    return train_logits, infer_logits

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, vocab_size 
                  , enc_embedding_size, dec_embedding_size, rnn_size, num_layers, 
                  questions_vocab_to_int):
   
    
    '''Use the previous functions to create the training and inference logits'''
    
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                       vocab_size+1, 
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    enc_state = encoder(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = dec_input_f(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    train_logits, infer_logits = decoding_layer(dec_embed_input, 
                                                dec_embeddings, 
                                                enc_state, 
                                                vocab_size, 
                                                sequence_length, 
                                                rnn_size, 
                                                num_layers, 
                                                questions_vocab_to_int, 
                                                keep_prob, 
                                                batch_size)
    return train_logits, infer_logits
    
# Set the Hyperparameters
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()
    
# Load the model inputs    
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(20, None, name='sequence_length')
# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create the training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(word_id), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, 
    word_id)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, word_id))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, word_id))
        yield pad_questions_batch, pad_answers_batch
    

train_valid_split = int(len(Initials)*0.15)

# Split the questions and answers into training and validating data
train_questions = Initials[train_valid_split:]
train_answers = Replies[train_valid_split:]

valid_questions = Initials[:train_valid_split]
valid_answers = Replies[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))
display_step = 100 # Check training loss after every 100 batches
stop_early = 0 
stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_questions))//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss = 0 # Record the training loss for each display step
summary_valid_loss = [] # Record the validation loss for saving improvements in the model

checkpoint = "best_model.ckpt" 

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs+1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(train_questions, train_answers, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs, 
                          batch_i, 
                          len(train_questions) // batch_size, 
                          total_train_loss / display_step, 
                          batch_time*display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                cost, {input_data: questions_batch,
                       targets: answers_batch,
                       lr: learning_rate,
                       sequence_length: answers_batch.shape[1],
                       keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))
            
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!') 
                stop_early = 0
                saver = tf.train.Saver() 
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break
    
    if stop_early == stop:
        print("Stopping Training.")
        break











