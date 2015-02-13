from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

def create_network():
    
    n = RecurrentNetwork()
    inputLayer1 = LinearLayer(