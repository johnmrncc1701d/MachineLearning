
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import csv

#ff = '../Sarcasm/Sarcasm_result_grid.csv'
#ff = '../Sarcasm/Sarcasm_KNN_result_grid.csv'
ff = '../Sarcasm/Sarcasm_ELB_result_grid.csv'

fullData = []
headerLine = []
sideLine = []
with open(ff,'r', newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for row in reader:
        lineData = []
        first = True
        for i in row:
            if first:
                sideLine.append(i)
                first = False
            else:
                lineData.append(i)

        if len(headerLine) == 0:
            headerLine = lineData
        else:
            fullData.append(lineData)
del sideLine[0]

myData = np.asfarray(fullData)
npHeader = np.asfarray(headerLine)
npSide = np.asfarray(sideLine)

# Decision Tree - Full Grid Results
if False:
    X, Y = np.meshgrid(npHeader, npSide)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, myData, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlim(50,350)
    ax.set_ylim(0.0,0.0020)
    plt.xlabel("Max Depth")
    plt.ylabel("Prune %")
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.4f}'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
    
# Decision Tree - Results @ Prune 0.0007
if False:
    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[6])
    
    plt.xlabel("Max Depth")
    plt.ylabel("% Match")

    plt.show()

# kNN
if False:
    fig, ax = plt.subplots()

    ax.plot(npSide, myData[:,0])
    
    plt.xlabel("k")
    plt.ylabel("% Match")

    plt.show()


# Boosting
if False:
    X, Y = np.meshgrid(npHeader, npSide)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, myData, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlim(0.0,0.0020)
    ax.set_ylim(0,5)
    plt.xlabel("Prune %")
    plt.ylabel("Max Depth")
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.4f}'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if True:
    tests = ["D-Tree", "Neural", "k-Near", "Boosting", "SVC", "NuSVC", "LinearSVC"]

    sarc_values = [76, 81, 71, 76, 80, 80, 79]
    sent_values = [65, 70, 60, 67, 70, 70, 67]
    sent2_values = [65, 67, 64, 64, 63, 0, 63]

    x_pos = [i for i, _ in enumerate(tests)]
    
    fig, ax = plt.subplots()
    ax.set_ylim(58,82)
    plt.bar(x_pos, sarc_values, color='green')
    #plt.bar(x_pos, sent_values, color='green')
    #plt.bar(x_pos, sent2_values, color='green')
    plt.xlabel("Learning Algorithm")
    plt.ylabel("% Match")
    plt.title("Percent match for each learning algorithm with the sarcasm data set")
    #plt.title("Percent match with the sentiment data set in regard to sentences")
    #plt.title("Percent match with the sentiment data set in regard to authors")
    plt.xticks(x_pos, tests)

    plt.show()