import sys
import csv

if __name__ == "__main__":

    args = sys.argv
    
    eta : float = 0
    threshold : float = 0
    filename : str = ""

    for i in range(len(args)):
        if args[i] == "--data":
            filename = str(args[i + 1])
        if args[i] == "--eta":
            eta = float(args[i + 1])
        if args[i] == "--threshold":
            threshold = float(args[i + 1])
        
    file = open(str(filename))
    data = []
    csv_reader_object = csv.reader(file)

    for row in csv_reader_object:
       data.append(row)

    weights : list = [0.0 for _ in range(len(data[0])) ]
    oldSumOfSq = 0
    iteration = 0
    
    while True:
        gradient : list = [0.0 for _ in range(len(data[0])) ]
        sumOfSq = 0
        string = ""

        for i in range(len(data)):
            vectorX = [data[i][k] for k in range(len(data[i])-1)]
            vectorX.insert(0, 1.0)

            fx = 0
            for j in range(len(data[i])):
                fx += weights[j] * float(vectorX[j])    #calculate f(x)

            t = float(data[i][len(data[i])-1]) - fx     #(yi - f(x))

            sumOfSq += t**2
            
            for l in range(len(vectorX)):
                vectorX[l] = float(vectorX[l]) * t      #xi * (yi - f(x))

            for j in range(len(gradient)):              #sum(xi * (yi - f(x)))    
                gradient[j] += vectorX[j]
        
        string += str(iteration) + ","
        for i in range(len(weights)):
            string += str(weights[i]) + ","
        string += str(sumOfSq)
        print(string)

        for j in range(len(gradient)):
            weights[j] = weights[j] + (eta * gradient[j]) #w -> w + (µ * gradient)

        if iteration > 1 and (oldSumOfSq - sumOfSq) < threshold:
            break

        oldSumOfSq = sumOfSq
        
        iteration += 1
        
