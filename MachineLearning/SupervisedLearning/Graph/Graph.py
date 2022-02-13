
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import csv

#ff = '../Sarcasm/Sarcasm_result_grid.csv'
#ff = '../Sarcasm/Sarcasm_KNN_result_grid.csv'
#ff = '../Sarcasm/Sarcasm_ELB_result_grid.csv'
#ff = '../Sarcasm/Sarcasm_result_vsSize.csv'

def Load(ff, maxlen=1000):
    fullData = []
    headerLine = []
    #sideLine = []
    with open(ff,'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            lineData = []
            count = 0
            for i in row:
                #if first:
                #    sideLine.append(i)
                #    first = False
                #else:
                lineData.append(i)
                count += 1
                if count >= maxlen:
                    break

            if len(headerLine) == 0:
                headerLine = lineData
            else:
                fullData.append(lineData)
    #del sideLine[0]
    return headerLine, fullData

#myData = np.asfarray(fullData)
#npHeader = np.asfarray(headerLine)
#npSide = np.asfarray(sideLine)

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

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_prune.csv', 10)
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Prune Value")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_depth.csv', 10)
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Depth")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_prune.csv', 10)
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Prune Value")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_depth.csv', 10)
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Depth")
    plt.ylabel("Performance (%)")

    plt.show()


# NN
if False:
    h, d = Load('../Sarcasm/Sarcasm_result_NN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_NN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_NN_iter.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_NN_hls.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_NN_alpha.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Alpha")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_NN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_NN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_NN_iter.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_NN_hls.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_NN_alpha.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Alpha")
    plt.ylabel("Performance (%)")

    plt.show()


# kNN
if False:
    fig, ax = plt.subplots()

    ax.plot(npSide, myData[:,0])
    
    plt.xlabel("k")
    plt.ylabel("% Match")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_kNN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_kNN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_kNN_K.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("# Neighbors")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_kNN_W.csv')
    myData = np.asfarray(d)
    npHeader = np.array(["uniform", "distance"])

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Weights")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_kNN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_kNN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_kNN_K.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("# Neighbors")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_kNN_W.csv')
    myData = np.asfarray(d)
    npHeader = np.array(["uniform", "distance"])

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Weights")
    plt.ylabel("Performance (%)")

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

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_ELB_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_ELB_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_ELB_est.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Estimators")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_ELB_prune.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Prune Value")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_ELB_depth.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Depth")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_ELB_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_ELB_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_ELB_est.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Estimators")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_ELB_prune.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Prune Value")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_ELB_depth.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Depth")
    plt.ylabel("Performance (%)")

    plt.show()


# SVN

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_SVN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_SVN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if True:
    h, d = Load('../Sarcasm/Sarcasm_result_SVN_iter.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_SVN_kernel.csv')
    myData = np.asfarray(d)
    npHeader = np.array(['linear', 'poly', 'rbf', 'sigmoid'])

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Kernal Function")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sarcasm/Sarcasm_result_SVN_tolerance.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Tolerance")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_SVN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Data Size (samples)")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_SVN_vsSize.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(myData[1], myData[0])
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Performance (%)")

    plt.show()

if True:
    h, d = Load('../Sentiment/Sentiment_result_SVN_iter.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Max Iterations")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_SVN_kernel.csv')
    myData = np.asfarray(d)
    npHeader = np.array(['linear', 'poly', 'rbf', 'sigmoid'])

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Kernal Function")
    plt.ylabel("Performance (%)")

    plt.show()

if False:
    h, d = Load('../Sentiment/Sentiment_result_SVN_tolerance.csv')
    myData = np.asfarray(d)
    npHeader = np.asfarray(h)

    fig, ax = plt.subplots()

    ax.plot(npHeader, myData[0])
    
    plt.xlabel("Tolerance")
    plt.ylabel("Performance (%)")

    plt.show()

###

if False:
    tests = ["D-Tree", "Neural", "k-Near", "Boosting", "SVC"] #, "NuSVC", "LinearSVC"

    sarc_values = [76, 80, 71, 74, 80] #, 80, 79
    sent_values = [66, 70, 60, 65, 70] #, 70, 67
    #sent2_values = [65, 67, 64, 64, 63, 0, 63]

    x_pos = [i for i, _ in enumerate(tests)]
    
    fig, ax = plt.subplots()
    ax.set_ylim(58,82)
    #plt.bar(x_pos, sarc_values, color='green')
    plt.bar(x_pos, sent_values, color='green')
    #plt.bar(x_pos, sent2_values, color='green')
    plt.xlabel("Learning Algorithm")
    plt.ylabel("% Match")
    #plt.title("Percent match for each learning algorithm with the sarcasm data set")
    plt.title("Percent match for each learning algorithm with the sentiment data set")
    #plt.title("Percent match with the sentiment data set in regard to sentences")
    #plt.title("Percent match with the sentiment data set in regard to authors")
    plt.xticks(x_pos, tests)

    plt.show()