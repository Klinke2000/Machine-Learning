import math
import sys   

def setTargets(data : list):
    targets : list = []
    for target in data:
        if target[-1] not in targets: #target[-1] classification at the end
            targets.append(target[-1])
    return targets

class DecisionTree():
      
    def __init__(self, depth : int, data : list, targets : list, logbase : int, attrIndex : list):
        self.node : str = "root"
        self.leaf : str = "no_leaf"
        self.depth : int = depth
        self.data : list = data
        self.targets : list = targets #want to classificate
        self.logbase : int = logbase
        self.attrIndex : list = attrIndex
        self.entropy : float = 0.0

    def printNode(self):

        print(str(self.depth)+","+str(self.node)+","+str(self.entropy)+","+str(self.leaf))

    def computeOverallEntropy(self, idx):
        entropy : float = 0.0
        entropyValues : list = []
        
        #get all different Values with their amount
        for i in range(len(self.data)): 
            if self.data[i][idx] not in entropyValues:
                entropyValues.extend([self.data[i][-1], 1]) #start at 1,because it occurs if it append on list
            else:
                entropyValues[entropyValues.index(self.data[i][-1]) + 1] += 1

        sumOfTargets : int = sum([element for element in entropyValues if not isinstance(element, str)])

        #compute Entropy with the formula
        for i in range(1, len(entropyValues), 2): 
            p_i = (entropyValues[i] / sumOfTargets) 
            entropy -= p_i * math.log(p_i, self.logbase)

        return entropy

    def computePartialEntropy(self, values : list):
        entropy : float = 0.0
        attValues : list = []

        #get different values with their amount for the targets
        for value in values:
            if value not in attValues:
                attValues.append(value)

        amountValues : list = [[0 for i in range(len(self.targets))] for j in range(len(attValues))]

        for i in range(len(values)):
            amountValues[attValues.index(values[i])][self.targets.index(self.data[i][-1])] += 1 #Increase the amount
        
        #compute Partial Entropy for every value
        for i in range(len(attValues)): 
            valueEntropy : float = 0.0

            for j in range(len(self.targets)):
                p_i : float = amountValues[i][j] / sum(amountValues[i])
                if p_i != 0.0:
                    valueEntropy -= p_i * math.log(p_i, self.logbase)
            entropy += (sum(amountValues[i]) / len(self.data)) * valueEntropy

        return entropy
    
    def algorithm(self):
        #compute Entropy
        self.entropy = self.computeOverallEntropy(-1)#compute Entropy
        
        if self.entropy == 0.0: #stops if entropy == 0.0
            self.leaf = self.data[0][-1]
            self.printNode()
            return
        
        #compute partial Entropys
        partialEntropy : list = []
        for i in range(len(self.data[0]) - 1): #last element are the targets
            partialEntropy.append(self.computePartialEntropy([self.data[k][i] for k in range(len(self.data))]))
        
        #compute Information Gain for next Node
        informationGain : list = []
        for i in range(len(self.data[0]) - 1):
            informationGain.append(self.entropy - partialEntropy[i])
        maxIdx : int = informationGain.index(max(informationGain)) #information Gain with max value is next node

        self.printNode()
        
        attValues : list = []
        for i in range(len(self.data)):
            if self.data[i][maxIdx] not in attValues:
                attValues.append(self.data[i][maxIdx])

        attValues.reverse()
        
        for i in range(len(attValues)):
            oldData : list = [line for line in self.data if line[maxIdx] == attValues[i]] #get all remaining data 
            newData : list = []
            
            #remove the max Attribute
            for j in range(len(oldData)):
                tempData : list = []
                for k in range(len(oldData[j])):
                    if k != maxIdx:
                        tempData.append(oldData[j][k])
                newData.append(tempData)

            #remove max Index
            newAttIndex : list = []
            for j in range(len(self.attrIndex)): #remove the maxIdx from index
                if j != maxIdx:
                    newAttIndex.append(self.attrIndex[j])
            
            childNode = DecisionTree(self.depth + 1, newData, self.targets, self.logbase, newAttIndex) #create Next Node with the new Data
            childNode.node = "att"+str(self.attrIndex[maxIdx])+"="+str(attValues[i])
            childNode.algorithm()

if __name__ == "__main__":
    args = sys.argv
    fileName : str = args[2]
 
    treeData : list = []
    file = open(fileName)

    for line in file:
            treeData.append(list(line.split(",")))

    for i in range(len(treeData)):
            treeData[i][-1] = treeData[i][-1].replace("\n", "")

    for i in range(len(treeData)):
            if len(treeData[i]) != len(treeData[0]):
                treeData.remove(treeData[i])

    decisionTree = DecisionTree(0, treeData, setTargets(treeData), len(setTargets(treeData)), [i for i in range(len(treeData[0]))])
    decisionTree.algorithm() #start algorithm