#
# Use different values of k
#

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import csv
import time

#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, Y)

#result = clf.predict([[2., 2.]])
#print(result)


_printCtr = 0
def PrintDot():
    global _printCtr
    _printCtr += 1
    if _printCtr >= 10:
        _printCtr = 0
        print(".", end="", flush=True)

class KNN:
    def __init__(self, file, k, weights, maxLines, multipleText, removeLowVariance):
        # Find all the words
        def AddWords(self, line):
            if line == "":
                return
            text = line
            words = text.split(' ')
            for word in words:
                if word == '':
                    continue
                if word[0] == '@':
                    word = '@name'
                if word not in self.dct:
                    self.dct[word] = len(self.dct.keys())
                    PrintDot()

        print("Computing Words", end="", flush=True)
        start = time.perf_counter()
        self.dct = {}
        for ff in file:
            with open(ff,'r', newline='') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                count = 0
                for row in reader:
                    if multipleText:
                        ln = len(row) - 2
                        for j in range(ln):
                            AddWords(self, row[2+j])
                    else:
                        AddWords(self, row[0])
                    count += 1
                    if count >= maxLines:
                        break
        print("Time: " + str(time.perf_counter() - start))
        
        # Craft the table
        def AddToSample(self, line, value, features, sampleList, valueList):
            if line == "":
                return
            if value == 1:
                return
            sample = [-1 for i in range(features)]
            words = line.split(' ')
            for word in words:
                if word == '':
                    continue
                if word[0] == '@':
                    word = '@name'
                sample[self.dct[word]] = 1
            sampleList.append(sample)
            valueList.append(-1 if value <= 1 else 1)
            PrintDot()

        print("\nBuilding Table", end="", flush=True)
        start = time.perf_counter()
        features = len(self.dct.keys())
        sampleList = []
        valueList = []
        for ff in file:
            with open(ff,'r', newline='') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                count = 0
                for row in reader:
                    if multipleText:
                        ln = len(row) - 2
                        for j in range(ln):
                            AddToSample(self, row[2+j], int(row[1]), features, sampleList, valueList)
                    else:
                        AddToSample(self, row[0], int(row[1]), features, sampleList, valueList)
                    count += 1
                    if count >= maxLines:
                        break
        print("Time: " + str(time.perf_counter() - start))

        if removeLowVariance:
            print("\nRemoving Low Variance", end="", flush=True)
            start = time.perf_counter()
            # Removing features with low variance
            sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
            sel = sel.fit(sampleList)
            retainList = sel.get_support()
            kys = list(self.dct.keys())
            pl = [None for j in kys]
            for i in kys:
                if retainList[self.dct[i]] == False:
                    del self.dct[i]
                else:
                    pl[self.dct[i]] = i
            i = 0
            while i < len(pl):
                if pl[i] is None:
                    del pl[i]
                    PrintDot()
                else:
                    i += 1
            for i in range(len(pl)):
                self.dct[pl[i]] = i
            sampleList = sel.transform(sampleList)
            print("Time: " + str(time.perf_counter() - start))

        # Create the tree
        print("\nCreating the KNN Grid", flush=True)
        print("K: " + str(k))
        print("Num Features: " + str(len(self.dct.keys())))
        print("Num Samples: " + str(len(valueList)))
        start = time.perf_counter()
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights, p=1)
        self.clf = clf.fit(sampleList, valueList)
        print("Time: " + str(time.perf_counter() - start))

    def Predict(self, file, maxLines, multipleText, outFile):
        def PredictValue(self, line, features):
            if line == "":
                return 0, False
            sample = [-1 for i in range(features)]
            words = line.split(' ')
            for word in words:
                if word in self.dct:
                    sample[self.dct[word]] = 1
            return self.clf.predict([sample])[0], True

        percent = 0
        with open(outFile,'w') as o:
            o.write('File:,"' + file + '"\nTEXT,VALUE,PRED,CORRECT\n')
            count = 0
            matches = 0
            features = len(self.dct.keys())
            print("Predicting", end="", flush=True)
            start = time.perf_counter()
            with open(file,'r', newline='') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                count2 = 0
                for row in reader:
                    value = int(row[1])
                    if value == 1:
                        continue
                    text = ""
                    result = 0
                    if multipleText:
                        ln = len(row) - 2
                        n = 0
                        p = 0
                        for j in range(ln):
                            result, valid = PredictValue(self, row[2+j], features)
                            if valid == False:
                                continue
                            text += row[2+j] + " "
                            if result == 1:
                                p += 1
                            else:
                                n += 1
                        if p > n:
                            result = 1
                        elif n > p:
                            result = -1
                        else:
                            if p == 0 and n == 0:
                                continue
                            result, valid = PredictValue(self, text, features)
                            if valid == False:
                                continue
                    else:
                        result, valid = PredictValue(self, row[0], features)
                        if valid == False:
                            continue
                        text = row[0]

                    v = -1 if value <= 1 else 1
                    o.write('"' + text + '",' + str(v) + ',' + str(result) + ',' + ('MATCH' if v == result else 'FAIL') + '\n')
                    count += 1
                    if v == result: matches += 1
                    PrintDot()
                    count2 += 1
                    if count2 >= maxLines:
                        break
            print("Time: " + str(time.perf_counter() - start))
            percent = 100*matches/count
            o.write('Tests:,' + str(count) + ',Matches:,' + str(matches) + ',Percent:,' + str(percent) + '\n')
            print("", flush=True)
        return percent

num = 6000

if False:
    k = 3
    d = KNN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], k, 'uniform', num, False, True)
    percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num, False, "../Sarcasm/Sarcasm_KNN_result_0.csv")

if False:
    k_min = 1
    k_max = 21
    k_step = 1
    
    kList = [[1, 2]]
    kList[0].insert(0,0)
    for k in np.arange(k_min, k_max, k_step):
        numList = [k]
        for w in ['uniform', 'distance']:
            d = KNN(["../Sarcasm/Sarcasm_train_set_1.csv"], k, w, num, False, True)
            print("Weight: " + w)
            percent = d.Predict("../Sarcasm/Sarcasm_train_set_2.csv", num, False, "../Sarcasm/Sarcasm_KNN_result_0.csv")
            numList.append(percent)
        kList.append(numList)
    result = np.asfarray(kList)
    np.savetxt("../Sarcasm/Sarcasm_KNN_result_grid0.csv", result, delimiter=",")

if False:
    k = 3
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sarcasm/Sarcasm_train_set_{j}.csv")
            
        d = KNN(fileList, k, 'distance', num, False, True)
        percent = d.Predict(fileList[0], num, False, f"../Sarcasm/Sarcasm_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sarcasm/Sarcasm_train_set_{i}.csv", num*3, False, f"../Sarcasm/Sarcasm_KNN_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sarcasm/Sarcasm_test_set.csv", num*3, False, "../Sarcasm/Sarcasm_KNN_final.csv")
    print("Percent Match: " + str(percent))

if False:
    k = 3
    w = 'distance'
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = KNN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], k, w, numb, False, True)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_kNN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_kNN_vsSize.csv", result, delimiter=",")
    
num = 5000

if False:
    k_min = 1
    k_max = 21
    k_step = 1
    w = 'distance'

    numList = [[d for d in range(k_min, k_max, k_step)], [0 for d in range(k_min, k_max, k_step)]]
    i = 0
    for k in np.arange(k_min, k_max, k_step):
        d = KNN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], k, w, num, False, True)
        print("K: " + str(k))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_kNN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_kNN_K.csv", result, delimiter=",")

if False:
    k = 3

    numList = [[d for d in range(0, 2, 1)], [0 for d in range(0, 2, 1)]]
    i = 0
    for w in ['uniform', 'distance']:
        d = KNN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], k, w, num, False, True)
        print("W: " + w)
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_kNN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_kNN_W.csv", result, delimiter=",")

# @ 6000 samples, k = 1, ~69% Euclidian distance
# grid 3->2->1; distance seems to be slightly better than uniform; Over 1 - 20 k, there seems to be multiple hills
# @ 6000 samples, k = 1, ~71% Manhatten distance

################

num = 6000

if False:
    k_min = 21
    k_max = 41
    k_step = 1
    
    kList = [[1, 2]]
    kList[0].insert(0,0)
    for k in np.arange(k_min, k_max, k_step):
        numList = [k]
        for w in ['uniform', 'distance']:
            d = KNN(["../Sentiment/Sentiment_train_set_1.csv"], k, w, num, False, True)
            print("Weight: " + w)
            percent = d.Predict("../Sentiment/Sentiment_train_set_2.csv", num, False, "../Sentiment/Sentiment_KNN_result_0.csv")
            numList.append(percent)
        kList.append(numList)
    result = np.asfarray(kList)
    np.savetxt("../Sentiment/Sentiment_KNN_result_grid1.csv", result, delimiter=",")

if True:
    k = 20
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment/Sentiment_train_set_{j}.csv")
            
        d = KNN(fileList, k, 'distance', num, False, True)
        percent = d.Predict(fileList[0], num, False, f"../Sentiment/Sentiment_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sentiment/Sentiment_train_set_{i}.csv", num*3, False, f"../Sentiment/Sentiment_KNN_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment/Sentiment_test_set.csv", num*3, False, "../Sentiment/Sentiment_KNN_final.csv")
    print("Percent Match: " + str(percent))

if False:
    k = 3
    w = 'distance'
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = KNN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], k, w, numb, False, True)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_kNN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_kNN_vsSize.csv", result, delimiter=",")
    
num = 5000

if False:
    k_min = 1
    k_max = 21
    k_step = 1
    w = 'distance'

    numList = [[d for d in range(k_min, k_max, k_step)], [0 for d in range(k_min, k_max, k_step)]]
    i = 0
    for k in np.arange(k_min, k_max, k_step):
        d = KNN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], k, w, num, False, True)
        print("K: " + str(k))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_kNN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_kNN_K.csv", result, delimiter=",")

if False:
    k = 3

    numList = [[d for d in range(0, 2, 1)], [0 for d in range(0, 2, 1)]]
    i = 0
    for w in ['uniform', 'distance']:
        d = KNN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], k, w, num, False, True)
        print("W: " + w)
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_kNN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_kNN_W.csv", result, delimiter=",")
    
# grid looking like a gradual increase

################

num = 2500
if False:
    k = 17
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment2/Sentiment_train_set_{j}.csv")
            
        d = KNN(fileList, k, 'distance', num, True, True)
        percent = d.Predict(fileList[0], num, True, f"../Sentiment2/Sentiment_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sentiment2/Sentiment_train_set_{i}.csv", num*3, True, f"../Sentiment2/Sentiment_KNN_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment2/Sentiment_test_set.csv", num*3, True, "../Sentiment2/Sentiment_KNN_final.csv")
    print("Percent Match: " + str(percent))