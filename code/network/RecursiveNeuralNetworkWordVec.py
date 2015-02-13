"""
This file is a network trial for a Recursive NN using word2vec as an absolute word vector generator
No learning is done on the word vectors. Gradients are used to learn the softmax weight matrix and the
NN weight matrix.
"""

import math
import pickle
import os
import numpy as np
import scipy.optimize
import scipy.sparse as sp
import gensim
from gensim.models import Word2Vec
import theano
import theano.tensor as T

import json
import random
import sys
from stanfordcorenlp.jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint
from nltk.tree import Tree
from StanfordSentimentUtils import buildStanfordSentimentDataSet


class CrossEntropyCost:
    
    @staticmethod
    def fn(a, y):
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))
        
    @staticmethod
    def delta(z, a, y):
        return (a-y)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


class RecursiveNeuralNetwork(object):
    
    def __init__(self, word_vector_size, cost=CrossEntropyCost):
        self.cost = cost
        self.word_vec_size = word_vector_size
        self.default_weight_initializer()
        
        # initialize some random vectors to symbolically show how the activation
        # looks
        self.y = T.drow('y')
        self.output = T.drow("output")
        
        # train the word2vec that is used by this network
        self.trainWord2Vec()
    
    def trainWord2Vec(self):
        base = '/Users/abhinavkhanna/Documents/Princeton/Independent_Work/python-scraper/lib/word2vec-training/'
        sentences = MySentences(base)
        self.model = gensim.models.Word2Vec(sentences)

    def default_weight_initializer(self):
        self.biases = theano.shared(
            np.asarray(
                np.random.randn(self.word_vec_size)
            ), name="biases", borrow=True
        )
        self.weight_matrix = theano.shared(
            np.asarray(
                np.random.randn(self.word_vec_size, self.word_vec_size * 2) # d x 2d
            ), name="weight_matrix", borrow=True
        )
        self.softmax_matrix = theano.shared(
            np.asarray(
                np.random.randn(2, self.word_vec_size) # 2 x d
            ), name="softmax_matrix", borrow=True
        )
        
        # set the parameters into one central array here
        self.params = [self.biases, self.weight_matrix, self.softmax_matrix]
    
    def feedforward(self, sentence):
        """
        Takes in a sentence, converts it to a sentence tree, and produces the vector form of all the words
        Then proceeds to input the sentence into the initial layer of the network.
        """
        # get the sentence tree structure.
        tree = getparsetree(sentence)
        
        # calculate the score, based on Socher et al 2012 paper on RNNs
        output_vector = self.recurse(tree) # length d output vector
        
        # use the output_vector as input into a softmax function classifier
        label = self.classifyOutput(output_vector)

        return label
    
    def recurse(self, tree):
        """
        This function works by recursing the tree down to nodes that do not have any children
        It grabs these children's word2Vec values and utilizes the Socher functions to calculate
        the vector to feed forward.
        """
        if not isinstance(tree, Tree):
            # this means its a leaf
            # symbolically tie this to a vector
            try:
                return self.model[tree][:self.word_vec_size]
            except:
                return np.random.randn(self.word_vec_size)
        elif len(tree) == 1:
            # this is a transfer node, not a true parent / child node, linear transfer okay
            return self.recurse(tree[0])
        else:
            # has exactly two children! This should be the case all the time
            # time to apply the Socher algorithm to it.
            v1 = self.recurse(tree[0])
            v2 = self.recurse(tree[1])
            return self.applyactivation(v1, v2)
    
    def applyactivation(self, v1, v2):
        # given v1 and v2, apply the socher activation function
        vec1 = T.dvector("vec1")
        vec2 = T.dvector("vec2")
        children_concat_vector = T.concatenate((vec1, vec2))
        dot_product_vector = T.dot( self.weight_matrix , children_concat_vector ) + self.biases
        activated_vector = T.tanh(dot_product_vector)
        f = theano.function([vec1, vec2], activated_vector)
        return f(v1, v2)
    
    def classifyOutput(self, output_vector):
        """
        Uses the softmax layer described in the Socher paper 2012 for RNNs to classify
        the d-dimensional vector into one of two bins, positive or negative
        Only the output layer has a softmax, because the other layers are truly
        incomplete forms
        """
        output = T.dvector("output")
        preclassification_dist = T.dot( self.softmax_matrix , output )
        classification_dist = T.nnet.softmax(preclassification_dist)
        f = theano.function([output], classification_dist)
        classification = f(output_vector)
        return classification

    def SGD(self, training_data, epochs, eta):
        # We are going to use theano to do SGD not using MiniBatches. Lack of MiniBatch use
        # is due to a lack of understanding of MB rather than a lack of appreciation for it.
        cost = self.cross_entropy_error()
        grads = theano.gradient.grad(cost, self.params)
        
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - eta * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]
        # updates = [
        #     (param_i, param_i)
        #     for param_i, grad_i in zip(self.params, grads)
        # ]
        
        a = T.drow("a")
        y1 = T.drow("y1")
        train_model = theano.function(
            [a, y1],
            cost,
            updates=updates,
            givens={
                self.output: a,
                self.y: y1
            }
        )
        np.random.shuffle(training_data)
        # we want to loop over the training data
        for epoch in xrange(epochs):
            for (sentence , sentiment) in training_data:
                # first feedforward on sentence
                #old_weight_matrix = np.sum(self.weight_matrix.get_value())
                sentiment = float(sentiment)
                if sentiment > 0.5:
                    sentiment = np.array([[0,1]])
                else:
                    sentiment = np.array([[1, 0]])

                #import pdb; pdb.set_trace()
                print("Sentence: {0}".format(sentence))
                try:
                    #import pdb; pdb.set_trace()
                    predicted_label_dist = self.feedforward(sentence)
                    actual_dist = sentiment
                    cost_ij = train_model(predicted_label_dist, actual_dist)
        
                    print("Cost of Training Data: {0}".format(cost_ij))
                    #print("Weight Matrix Diff: {0}".format(old_weight_matrix - np.sum(self.weight_matrix.get_value())))
                except:
                    continue
    
    def cross_entropy_error(self):
        return self.cost.fn(self.output, self.y)

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))

def getparsetree(sentence):
    nlp = StanfordNLP()
    result = nlp.parse(sentence)
    #pprint(result)

    tree = Tree.fromstring(result['sentences'][0]['parsetree'])
    #pprint(tree)
    return tree

def test_network():
    # this function is designed to test the network as best as possible
    training_set, test_set, dev_set = buildStanfordSentimentDataSet()
    network = RecursiveNeuralNetwork(100)
    network.SGD(training_set, 1, 0.05)