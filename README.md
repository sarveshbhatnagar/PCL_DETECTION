# PCL_DETECTION Task 1

Please ignore the plagiarism found with https://github.com/sarveshbhatnagar/PCL_DETECTION

This is our own code repository. You can verify that from its create date being not too old as we used this to combine our work together.
I don't know if we can give private access to more than 1 people.

Detecting whether a text contains patronizing or condesending language

# WORD2VEC From Data Binary

dataword.wordvectors
word2vec.wordvectors

# Scores

Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned).

# Below are the results for Binary PCL or NOT PCL.

## Over sampling and Deep Learning

Accuracy 0.8529130850047756
Precision [0.92915089 0.27755102]
Recall [0.90659631 0.34170854]

## Undersampling and Deep Learning

Accuracy 0.7110792741165234
Precision [0.95231417 0.19610778]
Recall [0.71662269 0.65829146]

## Naive Bayes

## (RandomOverSampling)

Accuracy: 0.7636103151862464
Precision: [0.96052632 0.24216028]
Recall: [0.77044855 0.69849246]

## (RandomUnderSampling)

Accuracy: 0.5377268385864374
Precision: [0.98534031 0.16242318]
Recall: [0.49656992 0.92964824]

NOTE : Accuracy, Precision and Recall for deep learning might change for different runs. we chose the best of 2 runs. You might find similar results if it is run atleast 3 times.

# How to run the PCL or NOT PCL?

for Naive Bayes...
python main_nb.py

for Deep Learning...
python main_dl.py for Under Sampling
python main_dl_ros.py for Over Sampling

# Code Organization for task1:

dataset : Data Original and Clean
feature : Basic Features, WordVector
models : Deep Learning Model (NOTE Naive Bayes implemented directly in main_nb.py)
preprocessing : basicPreProcessing and data cleaning
dont_patronize_me.py : Modified by dhia...
main_dl.py DEEP LEARNING WITH UNDERSAMPLING
main_dl_ros.py DEEP LEARNING WITH OVERSAMPLING
main_nb.py NAIVE BAYES

# For Task 2, please check task2 folder.

Additionally, data insights were collaborative efforts of Sarvesh and Pavan and preprocessing insights were individual efforts of Dhia(Task 1) and Krishna(Task2). Everyone contributed towards their respective tasks.

# Contributors
Bhatnagar Sarvesh

Naga Pavan Nukala

Rzig Dhia Elhaq

Krishna Koundinya Burgula


