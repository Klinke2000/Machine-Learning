import sys
import math

class NeuroNet:

    def __init__(self, filename):
        self.data : list = self.loadFile(filename)
                             #bias(1)       #a          #b        
        self.w_h : list = [[ 0.20000,   -0.30000,    0.40000],
                            [-0.50000,   -0.10000,   -0.40000],
                            [ 0.30000,    0.20000,    0.10000]]
                             #bias(1)       #h1         #h2        #h3
        self.w_o  : list = [-0.10000,    0.10000,    0.30000,  -0.40000]

        self.input : float = []
        self.h : list = [0,0,0]
        self.o : float = 0
        self.t : int = 0
        self.delta_h : float = [0,0,0]
        self.delta_o : float = 0

    def computeActivation(self, activation : list):
        self.input.insert(0,1) #insert Bias
        for i in range(len(self.w_h)):
            net : float = 0
            for j in range(len(self.w_h[i])):
                net += self.input[j] * self.w_h[i][j]
            activation[i] = 1 / (1 + math.exp(-net))
        return activation
    
    def computeOutputActivation(self):
        net : float = 1 * self.w_o[0]
        for i in range(1,len(self.w_o)):
            net += self.h[i-1] * self.w_o[i]
        activation = 1 / (1 + math.exp(-net))   
        return activation

    def adjustWeight(self, eta : float):
        for i in range(len(self.w_h)):
            for j in range(len(self.input)):
                self.w_h[i][j] += eta * self.input[j] * self.delta_h[i]
    
    def adjustOutputWeight(self,eta : float):
        #insert Bias
        self.w_o[0] += eta * 1 * self.delta_o
        for i in range(1,len(self.w_o)):
                self.w_o[i] += eta * self.h[i-1] * self.delta_o
 
    def computeError(self):
        for i in range(len(self.delta_h)):
            error : float = self.h[i] * (1 - self.h[i])
            self.delta_h[i] = (self.delta_o * self.w_o[i+1]) * error 
    
    def loadFile(self, fileName : str) -> list:
        file = open(filename)
        data = [[float(element.split(",")[i]) for i in range(3)] for element in file]
        return data
    
    def printStep(self):
        print(str(self.input[1]),",",str(self.input[2]),",",str(self.h[0]),",",str(self.h[1]),",",str(self.h[2]),",",str(self.o),",",str(self.t),",",str(self.delta_h[0]),",",str(self.delta_h[1]),",",str(self.delta_h[2]),","+str(self.delta_o),",",str(self.w_h[0][0]),",",str(self.w_h[0][1]),",",str(self.w_h[0][2]),",",str(self.w_h[1][0]),",",str(self.w_h[1][1]),",",str(self.w_h[1][2]),",",str(self.w_h[2][0]),",",str(self.w_h[2][1]),",",str(self.w_h[2][2]),",",str(self.w_o[0]),",",str(self.w_o[1]),",",str(self.w_o[2]),",",str(self.w_o[3]))
        


if __name__ == "__main__":
    args = sys.argv
    fileName : str = ""
    eta : float = 0.2
    iterations : int = 0
    
    for i in range(1, len(args)):
        if args[i] == "--data":
            filename = str(args[i+1])
        elif args[i] == "--eta":
            eta = float(args[i+1])
        elif args[i] == "--iterations":
            iterations = int(args[i+1])

    neuroNet = NeuroNet(filename)
    
    print("a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o")
    print("-,-,-,-,-,-,-,-,-,-,-,   0.20000,  -0.30000,   0.40000,  -0.50000,  -0.10000,  -0.40000,   0.30000,   0.20000,   0.10000,  -0.10000,   0.10000,   0.30000,  -0.40000")

    for iter in range(iterations):
        for i in range(len(neuroNet.data)):
            neuroNet.input = neuroNet.data[i][0:2]
            neuroNet.t = neuroNet.data[i][2]
            neuroNet.h = neuroNet.computeActivation(neuroNet.h)
            neuroNet.o = neuroNet.computeOutputActivation()
            neuroNet.delta_o = neuroNet.o * (1 - neuroNet.o) * (neuroNet.t - neuroNet.o)
            neuroNet.computeError()
            neuroNet.adjustWeight(eta)
            neuroNet.adjustOutputWeight(eta)
            neuroNet.printStep()
            

            

    