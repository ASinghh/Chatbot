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












