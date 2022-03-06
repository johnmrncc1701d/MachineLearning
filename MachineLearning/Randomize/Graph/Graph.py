
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
import numpy as np
import csv

def doGraph(folder, method, problem, maxIters):
    data = pd.read_csv("../" + folder + "/" + method + "-" + problem + ".csv")
    data = data[:maxIters]
    sns.set_theme(style="whitegrid")

    #fitness v. iters
    plt.clf()
    sns_plot=sns.lineplot(x="iters", y="fitness", data=data,)
    fig=sns_plot.get_figure()
    fig.savefig("../" + folder + "/" + problem + "_.png")

    #fitness v. fevals
    #plt.clf()
    #sns_plot2=sns.lineplot(x="fevals", y="fitness", data=data,)
    #fig2=sns_plot2.get_figure()
    #fig.savefig("../" + folder + "/" + problem + "2.png")

    #fevals/iters v. fitness
    #plt.clf()
    #sns_plot3=sns.lineplot(x="fitness", y="value", hue="variable", data=pd.melt(data, ['fitness']), legend="full")
    #fig3=sns_plot3.get_figure()
    #fig.savefig("../" + folder + "/" + problem + "3.png")

#doGraph("RandHillClimb", "RandomizedHillClimbing", "CountOnes")
#doGraph("RandHillClimb", "RandomizedHillClimbing", "FlipFlop_", 3000)
#doGraph("RandHillClimb", "RandomizedHillClimbing", "FourPeaks_100", 2000)
#doGraph("RandHillClimb", "RandomizedHillClimbing", "Knapsack_", 300)

#doGraph("SimAnneal", "SimulatedAnnealing", "CountOnes")
#doGraph("SimAnneal", "SimulatedAnnealing", "FlipFlop_", 3000)
#doGraph("SimAnneal", "SimulatedAnnealing", "FourPeaks_100", 2000)
#doGraph("SimAnneal", "SimulatedAnnealing", "Knapsack_", 300)

#doGraph("Genetic", "GeneticAlgo", "CountOnes")
#doGraph("Genetic", "GeneticAlgo", "FlipFlop_", 12000)
#doGraph("Genetic", "GeneticAlgo", "FourPeaks_100", 2000)
#doGraph("Genetic", "GeneticAlgo", "Knapsack_", 300)

#doGraph("MIMIC", "MIMIC", "CountOnes", 50)
#doGraph("MIMIC", "MIMIC", "FlipFlop_", 3000)
#doGraph("MIMIC", "MIMIC", "FourPeaks_100", 2000)
#doGraph("MIMIC", "MIMIC", "Knapsack_", 300)

def Load(ff, maxlen=1000):
    fullData = []
    with open(ff, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        count = 0
        for row in reader:
            count += 1
            if count >= maxlen:
                break

            fullData.append(row)

    return fullData

# Back Propigation
if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm0_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm0_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[2], myData[1])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm0_Result_1_vsIter.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

# Rnadom Hill Climb
if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm1_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm1_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[2], myData[1])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if True:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm1_Result_1_vsIter.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

# Simulated Annealing
if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm2_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm2_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[2], myData[1])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if True:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm2_Result_1_vsIter.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

# Genetic
if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm3_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm3_Result_1_vsSize.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[2], myData[1])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    d = Load('D:\\School\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm3_Result_1_vsIter.csv')
    myData = np.asfarray(d)
    myData = np.transpose(myData)

    fig, ax = plt.subplots()

    ax.plot(myData[0], myData[1])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()