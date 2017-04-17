#just give the path of .json file and .h5 file in the last lines while loading the model.

'''This Code will train Network on bAbi Dataset.
Papers on which this code is based on are:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
It gains 98.6% accuracy on task 'single_supporting_fact_10k' at 700 epochs, using RELU as a activation function, and when we used Sigmoid as a Activation function we need more Epoch as compared to RELU i.e at Epoch 600 we got only 60% accuracy.

'''
from __future__ import print_function
import tarfile
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file


'''
This function will parse the stories of the bAbi dataset, Only the sentences that support the answer are kept and rest are discared.

'''

def pruning(Totalline, only_supporting=False):
    
    TotalDataSet = []
    StoryDataSet = []
    for line in Totalline:
       
        line = line.decode('utf-8').strip()
        
        
        nid, line = line.split(' ', 1)    #Where is Daniel?  garden 11 i.e All lines will come here. including question and answer
                                          #line.split(' ',1), split line on spaces, keep 1st element in  nid and after space in line.

        nid = int(nid)                    #15 i.e line number
        if nid == 1:                      #only 1st line is kept in story dataset.
            StoryDataSet = []
        if '\t' in line:                  #It willl take only Question line, because only question line will contain tab.
            
            q, a, supporting = line.split('\t')
            #print('line' ,line)                   #line Where is John? 	bedroom	8
            #print('question' ,q)                   #question Where is John? 
            #print('answer' ,a)                     #answer bedroom
            #print('supporting' ,supporting)        #supporting 8

            q = tokenize(q)
            #print('Question', q)#Question [u'Where', u'is', u'John', u'?']

            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [StoryDataSet[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in StoryDataSet if x]
            TotalDataSet.append((substory, q, a))
            StoryDataSet.append('')
        else:
            Sentence = tokenize(line)
            #print('sentence' , Sentence) #sentence [u'Daniel', u'journeyed', u'to', u'the', u'garden', u'.']

            StoryDataSet.append(Sentence)#append each sentence in StoryDataSet.
    return TotalDataSet




'''
convert stories , Questions , Answers in a vector form, AnswerArray will contain an array of size equal to the vocabulary size , initially we fill all entries of this array as Zero and then we will find place 1 at those indexes that contain answer index... i.e  here DataArray array will contain
Stories i.e in vectorize form, QuestionArray will contain Questions, and About AnswerArray I explained Earlier....At the end of this function we have different arrays of Stories, Question, Answer that we will feed to neural Network. 

'''
def vectorize_stories(TotalDataSet, vocabularyId, story_maxlen, query_maxlen):
    DataArray = []
    QuestionArray = []
    AnswerArray = []
    for StoryDataSet, query, answer in TotalDataSet:
       
        dataarray = [vocabularyId[w] for w in StoryDataSet]
      
        queryarray = [VocabularyId[w] for w in query]
        # let's not forget that index 0 is reserved
        answerarray = np.zeros(len(VocabularyId) + 1)
        
        answerarray[VocabularyId[answer]] = 1
        

        DataArray.append(dataarray)
        QuestionArray.append(queryarray)
        AnswerArray.append(answerarray)
       # print('DataArray>>>' , DataArray)
       # print('QuestionArray>>>' ,QuestionArray)
       # print('AnswerArray>>>' , AnswerArray)
        
      
    return (pad_sequences(DataArray, maxlen=story_maxlength),
            pad_sequences(QuestionArray, maxlen=query_maxlength), np.array(AnswerArray))



''' This function will return tokenize the sentence  E.g.    tokenize('Jaya went to the temple. Where does Jaya gone?')
['Jaya' , 'went' , 'to' , 'the' , 'temple' , '.' ,'Where' , 'does' , 'Jaya' , 'gone' , '?' ]

'''


def tokenize(Sentence):
    
    return [x.strip() for x in re.split('(\W+)?', Sentence) if x.strip()] 

'''
This function will convert all the sentences of a given file into a single story , also stories longer than maximum length are pruned.
'''
def get_stories(f, only_supporting=False, max_length=None):
    
    TotalDataSet = pruning(f.readlines(), only_supporting=only_supporting)
    
    flatten = lambda TotalDataSet: reduce(lambda x, y: x + y, TotalDataSet)
    TotalDataSet = [(flatten(StoryDataSet), q, answer) for StoryDataSet, q, answer in TotalDataSet if not max_length or len(flatten(StoryDataSet)) < max_length]
    return TotalDataSet



'''
Download the babi Dataset and take the path where it is being stored and do rest work on it
'''
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)


challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))


vocabulary= set()
for StoryDataSet, q, answer in train_stories + test_stories:
    vocabulary |= set(StoryDataSet + q + [answer])
vocabulary = sorted(vocabulary)


# Reserve 0 for masking via pad_sequences
vocabulary_size = len(vocabulary) + 1
story_maxlength = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlength = max(map(len, (x for _, x, _ in train_stories + test_stories)))


'''
print('---->>')
print('Vocabulary size>>>', vocabulary_size, 'unique words')
print('maximum length of storY>>>', story_maxlength, 'words')
print('maximum length of story>>>', query_maxlength, 'words')
print('Number of training stories>>>', len(train_stories))
print('Number of test stories>>>', len(test_stories))
print('total vocabulary>>>' , vocabulary)
print('<<---')

print(train_stories[0])
print('<<<----')
print('***************On Vectorizing Word Sequences*************')

'''

VocabularyId = dict((c, i + 1) for i, c in enumerate(vocabulary))
# id for the words of story,id for the words of query, array Y contain a value of 1 at the position of answer index
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               VocabularyId,
                                                               story_maxlength,
                                                               query_maxlength)
# id for the test story, id for the query, array Y contain a value 1 at the position index of answer
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            VocabularyId,
                                                            story_maxlength,
                                                            query_maxlength)


'''
print('--->>>')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape--->>>', inputs_train.shape)
print('inputs_test shape--->>>', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape--->>>', queries_train.shape)
print('queries_test shape--->>>', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocabulary_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('<<<---')
print('*************************Compiling*************************')
'''
  # Create Tensor Object That will be used in making Model and training on it.
input_sequence = Input((story_maxlength,))
question = Input((query_maxlength,))


#  sequential encoder is used for encoding.
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocabulary_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.7))

# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocabulary_size,
                              output_dim=query_maxlength))
input_encoder_c.add(Dropout(0.7))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocabulary_size,
                               output_dim=64,
                               input_length=query_maxlength))
question_encoder.add(Dropout(0.7))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('sigmoid')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])



# we have chosen  RNN for reduction.
answer = LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = Dropout(0.3)(answer)
answer = Dense(vocabulary_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('sigmoid')(answer)

# build the final model
model = Model([input_sequence, question], answer)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#             metrics=['accuracy'])

# train
#history = model.fit([inputs_train, queries_train], answers_train,
#          batch_size=32,
#          epochs=1500,
         
#          validation_data=([inputs_test, queries_test], answers_test))


print('loading model')
import json   # importing json
from keras.models import model_from_json 
# open the .json file
with open('/home/iitp/Desktop/QA SYSTEM/QAPY/QASYSTEM5000.json', 'r') as arch_file:   # edit the file path as your own file path  

   model_arch = json.load(arch_file)
 
Re_model = model_from_json(model_arch)   #loading model to Re_model
Re_model.load_weights('/home/iitp/Desktop/QA SYSTEM/QAPY/QASYSTEM5000_weights.h5')   # loading weights to Re_model
Re_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])   # compiling model before use

scores = Re_model.evaluate([inputs_test, queries_test], answers_test,batch_size=32)         # evaluating the answer for queries in test file

print("Accuracy of the loaded model: %.4f%%" % (scores[1]*100))            #printing scores


