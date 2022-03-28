
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def MakeClusterFile(folder, red, clstr, trial, outfile):
    col_label = []
    col_collection = []
    for i in range(1, 2001):
        data = pd.read_csv(folder + "\\Sarcasm_" + red + trial + "_" + clstr + "_result_" + str(i) +".csv", skipfooter=5, header=None)
        col_data = []
        for j in range(450):
            lst = data.iloc[j]
            if i == 1:
                col_label.append(lst[0])
            col_data.append(lst[1])
        col_collection.append(col_data)
        print(".", end="", flush=True)

    with open(outfile + "\\Sarcasm_train_" + red + trial + "_" + clstr + "_result.csv", "w") as f:
        for j in range(450):
            for i in range(2000):
                f.write(str(col_collection[i][j]) + ",")
            f.write(str(col_label[j]) + "\n")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "pca", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "ica", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#for i in range(1,50):
#    MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "rca", "kmeans", str(i), \
#                    "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "lca", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "pca", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "ica", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#for i in range(1,50):
#    MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "rca", "rlink", str(i), \
#                    "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeClusterFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "lca", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

def GetDist(row1, row2):
    total = 0;
    a = []
    for i in range(np.size(row2)):
        d = row1[i] - row2[i]
        a.append(d*d)
    h = len(a)//2
    e = len(a)%2
    while True:
        for i in range(h):
            a[i] += a[(i+h)]
        if e == 1:
            a[0] += a[(h+h)]
        e = h%2
        h = h//2
        if h == 0:
            total = a[0]
            break
    return total

def MakeTestFile(folder, red, clstr, trial, outfile):
    col_label = []
    data = pd.read_csv(folder + "\\Sarcasm_" + red + trial + "_" + clstr + "_result_" + str(1) +".csv", skipfooter=5, header=None)
    for j in range(450):
        lst = data.iloc[j]
        col_label.append(lst[0])

    centerpairs = []
    for i in range(1, 2001):
        data = pd.read_csv(folder + "\\Sarcasm_" + red + trial + "_" + clstr + "_result_" + str(i) +".csv", skiprows=453, header=None)
        ndata = data.to_numpy(dtype=np.double)
        centerpairs.append([ndata[0],ndata[1]])
        print(".", end="", flush=True)
        
    strike = pd.read_csv("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sarcasm_" + red + "_strike_" + trial + ".csv", header=None)
    nstrike = strike.to_numpy(dtype=np.int)
    nstrike = nstrike.flatten()
    nstrike = np.delete(nstrike, nstrike.size-1)
    data = pd.read_csv("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sarcasm_test_list_1.csv", header=None)
    ndata = data.to_numpy(dtype=np.double)
    ndata = np.delete(ndata, nstrike, axis=1)
    with open(outfile + "\\Sarcasm_test_" + red + trial + "_" + clstr + "_result.csv", "w") as f:
        ccc = 0
        for j in range(50):
            for i in range(2000):
                dist1 = GetDist(ndata[j], centerpairs[i][0])
                dist2 = GetDist(ndata[j], centerpairs[i][1])
                if dist1 < dist2:
                    value = 0
                else:
                    value = 1
                f.write(str(value) + ",")
                ccc += 1
                if ccc == 20:
                    print(",", end="", flush=True)
                    ccc = 0
            f.write(str(col_label[j]) + "\n")

#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "pca", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "ica", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#for i in range(1,2):
#    MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "rca", "kmeans", str(i), \
#                    "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\kmeans", "lca", "kmeans", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")


#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "pca", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "ica", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#for i in range(1,2):
#    MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "rca", "rlink", str(i), \
#                    "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

#MakeTestFile("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\rlink", "lca", "rlink", str(1), \
#                "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2")

def MakeClusterPie(folder, dataset, clstr, trial, skip):
    rows = 10;
    data = pd.read_csv(folder + "\\" + dataset + "_" + clstr + "_result__group.csv", nrows=rows, skiprows=skip, header=None)
    ndata = data.to_numpy(dtype=np.float)
    labels = []
    for i in range(rows):
        labels.append("" + str(int(ndata[i][0])) + "-" + str(int(ndata[i][1])))
    ndata = np.delete(ndata, 0, axis=1)
    ndata = np.delete(ndata, 0, axis=1)
    ndata = ndata.flatten()

    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, ndata, color='green')
    plt.xlabel("Cluster Sizes")
    plt.ylabel("Frequency of Cluster Sizes")
    plt.xticks(x_pos, labels)
    plt.show()

#MakeClusterPie("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sarcasm", "kmeans", str(1), 1)
#MakeClusterPie("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sarcasm", "rlink", str(1), 2)
#MakeClusterPie("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sentiment", "kmeans", str(1), 1)
#MakeClusterPie("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sentiment", "rlink", str(1), 2)

def FindBestAndWorst(folder, dataset, clstr):
    best = 0.0
    best_i = 0
    worst = 1.0
    worst_i = 0
    for i in range(1, 2001):
        data = pd.read_csv(folder + "\\" + dataset + "_" + clstr + "_result_" + str(i) + ".csv", nrows=1, skiprows=452, header=None)
        lst = data.iloc[0]
        if lst[5] > best:
            best = lst[5]
            best_i = i
        if lst[5] < worst:
            worst = lst[5]
            worst_i = i
        print(",", end="", flush=True)
    print(best)
    print(best_i)
    print(worst)
    print(worst_i)

#FindBestAndWorst("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sarcasm", "kmeans")
#FindBestAndWorst("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\", "Sentiment", "kmeans")


if True:
    fullData = []
    with open('D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sentiment_lca_dist_1.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData = np.array(row, dtype=np.float)

    fullData = fullData[:-1]
    fig, ax = plt.subplots()
    ax.plot(fullData)
    #plt.ylabel("Eigenvalues")
    plt.yticks(np.arange(fullData.min(), fullData.max()+0.03, 0.1))
    plt.show()

if False:
    fullData = []
    with open('D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sarcasm_train_group_1.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData.append(row)

    fullData = np.array(fullData, dtype=np.int)
    
    strike = pd.read_csv("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sarcasm_pca_strike_1.csv", header=None)
    nstrike = strike.to_numpy(dtype=np.int)
    nstrike = nstrike.flatten()
    nstrike = np.delete(nstrike, nstrike.size-1)
    
    with open('D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sarcasm_pca_result_1_.csv', 'w', newline='') as f:
        for y in range(fullData.shape[0]):
            for x in range(fullData.shape[1]):
                if x in nstrike:
                    f.write('"[' + str(fullData[y][x]) + ']"')
                else:
                    f.write(str(fullData[y][x]))
                if x == fullData.shape[1] - 1:
                    f.write("\n")
                else:
                    f.write(",")

if False:
    strikeList = []
    for i in range(1,50):
        strike = pd.read_csv("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sentiment_rca_strike_" + str(i) + ".csv", header=None)
        nstrike = strike.to_numpy(dtype=np.int)
        nstrike = nstrike.flatten()
        nstrike = np.delete(nstrike, nstrike.size-1)
        for s in nstrike:
            for l in strikeList:
                if l[1] == s:
                    l[0] += 1
                    break
            else:
                strikeList.append([1,s])
        print(".", end="", flush=True)
    strikeList.sort(reverse=True)
    with open('D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\Sentiment_rca_strike_common.csv', 'w', newline='') as f:
        for l in strikeList:
            f.write("" + str(l[1]) + "," + str(l[0]) + "\n")

if False:
    best_i = 0
    worst_i = 0
    best_a = 0.0
    worst_a = 0.0
    best_t = 0.0
    worst_t = 0.0
    for i in range(1,50):
        itt = pd.read_csv("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn\\Sarcasm_rca_nn_iterations_" + str(i) + ".csv", header=None)
        row = itt.iloc[0]
        if i == 1:
            best_i = worst_i = row[0]
            best_a = worst_a = row[1]
            best_t = worst_t = row[1]
        else:
            if best_i > row[0]:
                best_i = row[0]
            if worst_i < row[0]:
                worst_i = row[0]
            if best_a < row[1]:
                best_a = row[1]
            if worst_a > row[1]:
                worst_a = row[1]
            if best_t > row[2]:
                best_t = row[2]
            if worst_t < row[2]:
                worst_t = row[2]
    print("iterations: " + str(worst_i) + "-" + str(best_i))
    print("accuracy: " + str(worst_a) + "-" + str(best_a))
    print("time: " + str(worst_t) + "-" + str(best_t))