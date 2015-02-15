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
        self.skipped = 0
        self.word_vec_size = word_vector_size
        self.secondary_word_dict = {}
        self.tree_to_vector = {} # empty dict
        self.rand_word_vec_count = 0 # count how many times we use a random word vector
        self.word_count = 0 # counts words
        self.default_weight_initializer()
        
        # train the word2vec that is used by this network
        self.trainWord2Vec()
    
    def trainWord2Vec(self):
        base = '/Users/abhinavkhanna/Documents/Princeton/Independent_Work/python-scraper/lib/word2vec-training/'
        sentences = MySentences(base)
        self.model = gensim.models.Word2Vec(sentences)

    def default_weight_initializer(self):
        self.biases = np.asarray(np.random.randn(self.word_vec_size))
        self.biases = self.biases.reshape(self.biases.shape[0], 1)

        self.weight_matrix = np.asarray(np.random.randn(self.word_vec_size, self.word_vec_size * 2))
        self.softmax_matrix = np.asarray(np.random.randn(2, self.word_vec_size))
        
        # set the parameters into one central array here
        self.params = [self.biases, self.weight_matrix, self.softmax_matrix]
    
    def softmax(self, w):
        w = w.reshape((2, 1))
        w = np.array(w)

        # maxes = np.amax(w, axis=1)
        # maxes = maxes.reshape(maxes.shape[0], 1)
        # e = np.exp(w - maxes)
        # dist = e / np.sum(e)
        
        
        e = np.exp(w)
        fullexp = np.sum(np.exp(w))
        dist = e / fullexp
        
 
        return dist
    
    def feedforward(self, sentence):
        """
        Takes in a sentence, converts it to a sentence tree, and produces the vector form of all the words
        Then proceeds to input the sentence into the initial layer of the network.
        """
        # get the sentence tree structure.
        # convert sentence to lower case first
        sentence = sentence.lower()
        self.tree = getparsetree(sentence)
        
        # calculate the score, based on Socher et al 2012 paper on RNNs
        output_vector = self.recurse(self.tree) # length d output vector
        
        # use the output_vector as input into a softmax function classifier
        label = self.classifyOutput(output_vector)

        return label
    
    def inSecondaryDict(self, label):
        for key, value in self.secondary_word_dict.iteritems():
            if (key == label):
                return True;
        return False
    
    def recurse(self, tree):
        """
        This function works by recursing the tree down to nodes that do not have any children
        It grabs these children's word2Vec values and utilizes the Socher functions to calculate
        the vector to feed forward.
        """
        if not isinstance(tree, Tree):
            # this means its a leaf
            # symbolically tie this to a vector
            self.word_count = self.word_count + 1
            try:
                word_vec = self.model[tree][:self.word_vec_size]
                word_vec = word_vec.reshape(word_vec.shape[0], 1)
                
                assert word_vec.shape[0] == self.word_vec_size
                assert word_vec.shape[1] == 1
                
                return word_vec
            except:
                
                # check the secondary dictionary first
                if (self.inSecondaryDict(tree)):
                    word_vec = self.secondary_word_dict[tree] # only stores the word_vecs we add to it, which all have length d
                    word_vec = word_vec.reshape(word_vec.shape[0], 1)
                else:
                    self.rand_word_vec_count = self.rand_word_vec_count + 1 # count how many times we use random word vectors
                    word_vec = np.random.randn(self.word_vec_size)
                    # add the word_vec to the secondary dictionary in the flattened form
                    self.secondary_word_dict[tree] = word_vec
                    word_vec = word_vec.reshape(word_vec.shape[0], 1)
                
                # assert that the size is maintained of the vector properly
                assert word_vec.shape[0] == self.word_vec_size
                assert word_vec.shape[1] == 1
                
                return word_vec
        elif len(tree) == 1:
            # this is a transfer node, not a true parent / child node, linear transfer okay
            activation = self.recurse(tree[0])
            self.tree_to_vector[tree.pprint()] = activation
            return activation
        else:
            # has exactly two children! This should be the case all the time
            # time to apply the Socher algorithm to it.
            v1 = self.recurse(tree[0])
            v2 = self.recurse(tree[1])
            
            if v1.shape[0] != v2.shape[0] or v1.shape[1] != v2.shape[1]:
                print "{0} + {1} shapes".format(v1.shape, v2.shape)

            assert v1.shape[0] == v2.shape[0] and v1.shape[1] == v2.shape[1]
            
            activation = self.applyactivation(v1, v2)
            self.tree_to_vector[tree.pprint()] = activation
            return activation
    
    def applyactivation(self, v1, v2):
        # given v1 and v2, apply the socher activation function
        children_concat_vector = np.concatenate((v1, v2))
        dot_product_vector = np.dot( self.weight_matrix , children_concat_vector ) + self.biases
        activated_vector = np.tanh(dot_product_vector)
        return activated_vector
    
    def classifyOutput(self, output_vector):
        """
        Uses the softmax layer described in the Socher paper 2012 for RNNs to classify
        the d-dimensional vector into one of two bins, positive or negative
        Only the output layer has a softmax, because the other layers are truly
        incomplete forms
        """
        preclassification_dist = np.dot( self.softmax_matrix , output_vector )
        classification_dist = self.softmax(preclassification_dist)
        classification = classification_dist
        return classification

    def SGD(self, training_data, epochs, eta, mini_batch_size,
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False, 
            monitor_training_accuracy=False):
        # We are going to use theano to do SGD not using MiniBatches. Lack of MiniBatch use
        # is due to a lack of understanding of MB rather than a lack of appreciation for it.
        self.word_count = 0
        self.rand_word_vec_count = 0
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            count = 0
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
                count = count + 1
                print "BP: {0} percent complete".format(float(count) / float(len(mini_batches)) * 100)
            print "Epoch %s training complete" % j
            print "Random word vec used {0} of the time".format(float(self.rand_word_vec_count / self.word_count))
            
            # import pdb; pdb.set_trace()
            if monitor_training_cost:
                cost = self.total_cost(training_data)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data)

        # need to save the model because I am tired of retraining over and over again
        np.savetxt("weight_matrix.txt", self.weight_matrix)
        np.savetxt("softmax_matrix.txt", self.softmax_matrix)
        np.savetxt("biases.txt", self.biases)
        np.savetxt("model.txt", self.model)
        np.savetxt("secondary_word_dict.txt", self.secondary_word_dict)

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        batch_delta_b = np.zeros(self.biases.shape)
        batch_delta_ws = np.zeros(self.softmax_matrix.shape)
        batch_delta_w = np.zeros(self.weight_matrix.shape)
        len_delta_w = 0
        try:
            # import pdb; pdb.set_trace()
            for x, y in mini_batch:
                x = clean_input(x)
                delta_nabla_b, delta_nabla_w, delta_nabla_ws = self.backprop(x, y)
                len_delta_w = len_delta_w + len(delta_nabla_w)
                # add in the delta from this round into the respective vectors
                # in the end we will subtract from the utilized parameters based on these net deltas.
                for dnb in delta_nabla_b:
                    batch_delta_b = batch_delta_b + dnb

                for dnw in delta_nabla_w:
                    batch_delta_w = batch_delta_w + dnw
                
                for dnws in delta_nabla_ws:
                    batch_delta_ws = batch_delta_ws + dnws

            # import pdb; pdb.set_trace()
            batch_delta_w = batch_delta_w / len_delta_w # average delta is the one delta that will be applied
            
            if np.isnan(np.sum(batch_delta_w)):
                self.skipped = self.skipped + 1
                print "Skipped: {0}".format(self.skipped)
                return; # just don't update it count it as a skip
            
            self.softmax_matrix = np.nan_to_num((1 - eta * (lmbda/n)) * self.softmax_matrix - (eta/len(mini_batch)) * batch_delta_ws)
            self.weight_matrix = np.nan_to_num((1 - eta * (lmbda/n)) * self.weight_matrix - (eta/len(mini_batch)) * batch_delta_w)
            self.biases = np.nan_to_num(self.biases - (eta/len(mini_batch)) * batch_delta_b)
        
            # print "Weight Matrix: {0}\n".format(self.weight_matrix)
            # print "Weight Matrix Error: {0}\n".format(batch_delta_w)
            # print "Softmax: {0}\n".format(self.softmax_matrix)
            # print "Softmax Error: {0}\n".format(batch_delta_ws)
        except:
            self.skipped = self.skipped + 1
            print "Skipped: {0}".format(self.skipped)
            return
    
    def traverse(self, tree, delta_down, nabla_b, nabla_w):
        """
        define this function to operate recursively, otherwise will not work
        """
        if not isinstance(tree, Tree):
            # this means its a leaf
            # we need to give it the delta
            # it doesn't really make sense for there to be error in this condition
            # delta_down needs to get absorbed by something eventually
            # otherwise it will continue to grow erratically
            # since we have reached a leaf there must be some error with the word vector
            # we can utilize the delta
            try:
                word_vec = self.model[tree][:self.word_vec_size]
                word_vec = word_vec.reshape(word_vec.shape[0], 1)
                word_vec = np.nan_to_num(word_vec + 0.005 * delta_down)
                self.model[tree][:self.word_vec_size] = word_vec.flatten()
            except:
                # not in self.model, must be in secondary dictionary
                word_vec = self.secondary_word_dict[tree]
                word_vec = word_vec.reshape(word_vec.shape[0], 1)
                word_vec = np.nan_to_num(word_vec + 0.005 * delta_down)
                self.secondary_word_dict[tree] = word_vec.flatten()
        elif len(tree) == 1:
            # this is a transfer node, not a true parent / child node, linear transfer okay
            self.traverse(tree[0], delta_down[:self.word_vec_size], nabla_b, nabla_w)
        else:
            # has exactly two children! This should be the case all the time
            # time to apply the Socher algorithm to it.
            # we need to calculate delta down
            
            # you are receiving your own layer's error through the delta, need to calculate change
            LCV = self.tree_to_vector[tree[0].pprint()]
            RCV = self.tree_to_vector[tree[1].pprint()]
            layered_child_vec = np.concatenate((LCV, RCV))
            layered_child_vec = layered_child_vec.reshape(layered_child_vec.shape[0], 1)
            
            # import pdb; pdb.set_trace()
            # Has to be the activation times the delta to produce the derivative of change
            activated_delta = np.nan_to_num(np.dot(delta_down, layered_child_vec.T))
            
            # append this layers derivative change quantities
            nabla_b.append(delta_down)
            nabla_w.append(activated_delta)
            
            # calculate the new delta to propogate downwards
            new_delta_down = np.nan_to_num(0.05 * np.dot(self.weight_matrix.T, delta_down) * tanh_prime(layered_child_vec))
            self.traverse(tree[0], new_delta_down[self.word_vec_size:], nabla_b, nabla_w)
            self.traverse(tree[1], new_delta_down[:self.word_vec_size], nabla_b, nabla_w)
    
    def backprop(self, sentence, y):
        nabla_b = []
        nabla_w = []
        nabla_ws = []
        # feedforward
        predicted = self.feedforward(sentence)
        # backward pass
        # convert y first
        if (float(y) < 0.5):
            y = np.array([[1, 0]])
        else:
            y = np.array([[0, 1]])
        
        
        # y has shape 1, 2 when it should have 2, 1, just transpose it.
        delta_outputLayer = np.dot(self.softmax_matrix.T, (predicted - y.T)) * tanh_prime(self.tree_to_vector[self.tree.pprint()])
        nabla_ws.append(np.dot(predicted, delta_outputLayer.T))
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        if (len(self.tree) == 1):
            layered_child_vec = np.concatenate((self.tree_to_vector[self.tree[0].pprint()], self.tree_to_vector[self.tree[0].pprint()]))
            delta_down = np.dot(self.weight_matrix.T, delta_outputLayer) * tanh_prime(layered_child_vec)
        
            # this will propogate the errors recursively downwards
            self.traverse(self.tree[0], delta_down[:self.word_vec_size], nabla_b, nabla_w)
        elif len(self.tree) == 2:
            layered_child_vec = np.concatenate((self.tree_to_vector[self.tree[0].pprint()], self.tree_to_vector[self.tree[1].pprint()]))
            delta_down = np.dot(self.weight_matrix.T, delta_outputLayer) * tanh_prime(layered_child_vec)
        
            # this will propogate the errors recursively downwards
            self.traverse(self.tree[0], delta_down[:self.word_vec_size], nabla_b, nabla_w)
            self.traverse(self.tree[1], delta_down[self.word_vec_size:], nabla_b, nabla_w)
            
        return (nabla_b, nabla_w, nabla_ws)
    
    def total_cost(self, data):
        cost = 0.0
        count = 0
        for x, y in data:
            count = count + 1
            print "TC: {0} percent complete".format(float(count) / float(len(data)))
            try:
                if (float(y) < 0.5):
                    y = np.array([[1, 0]])
                else:
                    y = np.array([[0, 1]])
            
                a = self.feedforward(x)
                cost += self.cost.fn(a, y)/len(data)
            except:
                continue
        return cost
    
    def accuracy(self, data):
        results = []
        count = 0
        for x, y in data:
            count = count + 1
            print "A: {0} percent complete".format(float(count) / float(len(data)))
            try:
                if (float(y) < 0.5):
                    y = np.array([[1, 0]])
                else:
                    y = np.array([[0, 1]])
                results.append((np.argmax(self.feedforward(x)), np.argmax(y)))
            except:
                continue
        return sum(int(x == y) for (x, y) in results) #TODO: DOUBLE CHECK THIS LINE, NOT SURE WHAT ARGMAX DOES IN PREV LINE

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

def tanh_prime(x):
    retVec = 4 * np.square(np.cosh(x)) / (np.square(np.cosh(2 * x) + 1))
    retVec = retVec.reshape(retVec.shape[0], 1)
    return np.nan_to_num(retVec)

def clean_input(sentence):
    """
    cleans the input sentence of various garbage
    """
    sentence = sentence.replace("-", "")
    sentence = sentence.replace("\\", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace(",","")
    sentence = sentence.replace("LRB", "")
    sentence = sentence.replace("RRB", "")
    return sentence

def test_network():
    # this function is designed to test the network as best as possible
    training_set, test_set, dev_set = buildStanfordSentimentDataSet()
    network = RecursiveNeuralNetwork(100)
    network.SGD(training_set, 10, 0.005, 10, monitor_training_accuracy=True, monitor_training_cost=True)
    
test_network()