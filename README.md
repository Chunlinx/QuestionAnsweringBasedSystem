# QuestionAnsweringBasedSystem
This is an implementation Of Question Answering System on bAbi DataSet
This Code will train Network on bAbi Dataset.
Papers on which this code is based on are:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
It gains 98.6% accuracy on task 'single_supporting_fact_10k' at 700 epochs, using RELU as a activation function, and when we used Sigmoid as a Activation function we need more Epoch as compared to RELU i.e at Epoch 600 we got only 60% accuracy,But at Epoch 1000 we get 93% accuracy

Now the Methods that we used are:

Pruning--->>This function will parse the stories of the bAbi dataset, Only the sentences that support the answer are kept and rest are discared.

vectorize_stories--->>convert stories , Questions , Answers in a vector form, AnswerArray will contain an array of size equal to the vocabulary size , initially we fill all entries of this array as Zero and then we will find place 1 at those indexes that contain answer index... i.e  here DataArray array will contain
Stories i.e in vectorize form, QuestionArray will contain Questions, and About AnswerArray I explained Earlier....At the end of this function we have different arrays of Stories, Question, Answer that we will feed to neural Network. 

tokenize--->>This function will return tokenize the sentence  E.g.    tokenize('Jaya went to the temple. Where does Jaya gone?')
['Jaya' , 'went' , 'to' , 'the' , 'temple' , '.' ,'Where' , 'does' , 'Jaya' , 'gone' , '?' ]


get_stories--->>This function will convert all the sentences of a given file into a single story , also stories longer than maximum length are pruned.

How We Actually Implemeted this??

1> Download the bAbi Dataset, from the given link in the code.
Sample bAbi DataSet>>

1 John Went to the temple.
2  Denial followed John.
3  Enna finally went to kitchen.
4  Where does John went?   temple 1

So, 1 in the neighbourhood of temple indicate that which statment will support this answer.

2> Tokenize the sentence using the utf format.


3> prune the dataset.

4> create the three arrays storyArray,QuestionArray,AnswerArray.

5>Vectorize these arrays.

6>Give these Arrays to the LSTM neural network.

7>Training is done on 10000 data sample and validation on 1000dataset.

8>Save the model.

9>plot the graphs for Accuracy and Loss.





