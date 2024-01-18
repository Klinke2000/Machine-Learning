import sys
import math

class Clustering:
    
    def __init__(self, data):
        self.centroids = [[0,5], [0,4], [0,3]]
        self.criteria = 0
        self.optCriteria = [1]
        self.data = data

    def calculateError(self, centroid, point):
        return ((centroid[0] - point[0])**2) + ((centroid[1] - point[1])**2)
        
    def calculateCriteria(self):
        criteria = 0
        for element in self.data:
            criteria += min([self.calculateError(self.centroids[i], element)for i in range(len(self.centroids))])
        
        self.criteria = criteria
        self.optCriteria.append(self.criteria)
            
    def calculateDistance(self, centroid, point):
        return math.sqrt(((centroid[0] - point[0])**2) + ((centroid[1] - point[1])**2))

    def calculateCentroids(self):
        
            
def loadFile(filename : str) -> list:
        file = open(filename)
        data = [[float(element.replace("\n","").split(",")[i]) for i in range(1,3)] for element in file]
        return data

if __name__ == "__main__":

    #args = sys.argv
    #filename : str = args[1]

    cluster = Clustering(loadFile("Example.csv"))
    
    while cluster.criteria != cluster.optCriteria[-1]:
        cluster.calculateCriteria()
        cluster.calculateCentroids()
    
    for i in range(1, len(cluster.optCriteria)):
        print(cluster.optCriteria[i])
    
    