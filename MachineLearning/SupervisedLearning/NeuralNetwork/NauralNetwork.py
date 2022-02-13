
from sklearn.neural_network import MLPClassifier
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

class NN:
    def __init__(self, file, alpha, num_hidden, maxLines, multipleText, removeLowVariance, hiddenLayerSizes=None, random_state=None, max_iter=500):
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
            print("\nRemoving Low Variance: maxLines="+str(maxLines), end="", flush=True)
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
        print("\nCreating the Network", flush=True)
        start = time.perf_counter()
        if hiddenLayerSizes is not None:
            hidden_layer_sizes = hiddenLayerSizes
        else:
            #hidden_layer_sizes = tuple([features*2 for i in range(num_hidden)])
            #hidden_layer_sizes = tuple([features for i in range(num_hidden)])
            #hidden_layer_sizes = tuple([features//(i+1) for i in range(num_hidden)])
            #hidden_layer_sizes = tuple([i*(features//(num_hidden+1)) for i in range(num_hidden, 0, -1)])
            #hidden_layer_sizes = tuple([(features//(i+2)) for i in range(num_hidden)])
            #hidden_layer_sizes = tuple([(features//(1+(i+1)**2)) for i in range(num_hidden)])
            #hidden_layer_sizes = tuple([(features//((i+2)**2)) for i in range(num_hidden)])
            #hidden_layer_sizes = [(features//((i+2)**3)) for i in range(num_hidden)]
            hidden_layer_sizes = (5,)
        for i in range(len(hidden_layer_sizes)-1,-1,-1):
            if hidden_layer_sizes[i] <= 0:
                del hidden_layer_sizes[i]
        hidden_layer_sizes = tuple(hidden_layer_sizes)
        print("Num Features: " + str(len(self.dct.keys())))
        print("Num Samples: " + str(len(valueList)))
        print("Hidden Layers: " + str(hidden_layer_sizes))
        clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, activation='logistic', random_state=random_state, verbose=True, max_iter=max_iter)
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

num = 6000 #332 #6000

if False:
    alpha = 0.000001 #0.00001
    num_hidden = 1
    percent = 0
    random_state = 0
    while percent < 65:
        random_state += 1
        d = NN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], alpha, num_hidden, num, False, True, (5,), random_state)
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_1.csv", num, False, "../Sarcasm/Sarcasm_NN_result_0a.csv")
        #if percent >= 65:
            #d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_NN_result_0b.csv")
        #d = NN(["../Sarcasm/Sarcasm_train_set_1.csv"], alpha, num_hidden, num*3, False, (5,), random_state)
        #percent = d.Predict("../Sarcasm/Sarcasm_train_set_1.csv", num*3, False, "../Sarcasm/Sarcasm_NN_result_0.csv")
    print(random_state)
    
if False:
    alpha = 0.000001 #0.00001
    num_hidden = 1
    hls_min = 3
    hls_max = 11
    hls_step = 1
    num_min = 1000
    num_max = 2000
    num_step = 1
    
    hlsList = [[d for d in range(num_min, num_max, num_step)]]
    hlsList[0].insert(0,0)
    for hls in np.arange(hls_min, hls_max, hls_step):
        numList = [hls]
        for n in np.arange(num_min, num_max, num_step):
            d = NN(["../Sarcasm/Sarcasm_train_set_1.csv"], alpha, num_hidden, n, False, True, (hls,))
            print("Count: " + str(n))
            print("Hidden Layers: " + str((hls,)))
            percent = d.Predict("../Sarcasm/Sarcasm_train_set_1.csv", n, False, "../Sarcasm/Sarcasm_NN_result_0.csv")
            numList.append(percent)
        hlsList.append(numList)
    result = np.asfarray(hlsList)
    np.savetxt("../Sarcasm/Sarcasm_NN_result_grid3.csv", result, delimiter=",")

if True:
    alpha = 0.0018
    num_hidden = 1
    percentList = [[0, 0, None, 1],[0, 1, None, 1],[0, 2, None, 1],[0, 3, None, 1]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sarcasm/Sarcasm_train_set_{j}.csv")
            
        percent = 0
        random_state = 0
        while percent < 65:
            random_state += 1
            d = NN(fileList, alpha, num_hidden, num, False, True, (5,), random_state, 80)
            percent = d.Predict(fileList[0], num, False, f"../Sarcasm/Sarcasm_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sarcasm/Sarcasm_train_set_{i}.csv", num*3, False, f"../Sarcasm/Sarcasm_NN_result_{i}.csv")
        percentList[i-1][2] = d
        percentList[i-1][3] = random_state

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sarcasm/Sarcasm_test_set.csv", num*3, False, "../Sarcasm/Sarcasm_NN_final.csv")
    print("Random Seed: " + str(percentList[0][3]))
    print("Percent Match: " + str(percent))

if False:
    alpha = 0.000001
    num_hidden = 1
    hls = 5
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = NN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], alpha, num_hidden, numb, False, True, (hls,), 0)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_NN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_NN_vsSize.csv", result, delimiter=",")

num = 3500

if False:
    alpha = 0.000001
    num_hidden = 1
    hls_min = 3
    hls_max = 11
    hls_step = 1

    hlsList = [[d for d in range(hls_min, hls_max, hls_step)], [0 for d in range(hls_min, hls_max, hls_step)]]
    i = 0
    for hls in np.arange(hls_min, hls_max, hls_step):
        d = NN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0)
        print("HLS: " + str(hls))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_NN_0.csv")
        hlsList[1][i] = percent
        i += 1
    result = np.asfarray(hlsList)
    np.savetxt("../Sarcasm/Sarcasm_result_NN_hls.csv", result, delimiter=",")

if False:
    alpha_min = 0.0001
    alpha_max = 0.0021
    alpha_step = 0.0001
    num_hidden = 1
    hls = 5

    alphaList = [[d for d in np.arange(alpha_min, alpha_max, alpha_step)], [0.0 for d in np.arange(alpha_min, alpha_max, alpha_step)]]
    i = 0
    for alpha in np.arange(alpha_min, alpha_max, alpha_step):
        d = NN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0)
        print("alpha: " + str(alpha))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_NN_0.csv")
        alphaList[1][i] = percent
        i += 1
    result = np.asfarray(alphaList)
    np.savetxt("../Sarcasm/Sarcasm_result_NN_alpha.csv", result, delimiter=",")

if False:
    iter_min = 5
    iter_max = 156
    iter_step = 10
    num_hidden = 1
    hls = 5
    alpha = 0.000001

    alphaList = [[d for d in np.arange(iter_min, iter_max, iter_step)], [0.0 for d in np.arange(iter_min, iter_max, iter_step)]]
    i = 0
    for iter in np.arange(iter_min, iter_max, iter_step):
        d = NN(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0, iter)
        print("iter: " + str(iter))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_NN_0.csv")
        print("result: " + str(percent))
        alphaList[1][i] = percent
        i += 1
    result = np.asfarray(alphaList)
    np.savetxt("../Sarcasm/Sarcasm_result_NN_iter.csv", result, delimiter=",")

# hidden=2
# 53.1%   #hidden_layer_sizes = tuple([i*(features//(num_hidden+1)) for i in range(num_hidden, 0, -1)])
# 53.1%   #hidden_layer_sizes = tuple([(features//(i+2)) for i in range(num_hidden)])
# 53.1%   #hidden_layer_sizes = tuple([(features//(1+(i+1)**2)) for i in range(num_hidden)])
# 53.1%   #hidden_layer_sizes = tuple([(features//((i+2)**2)) for i in range(num_hidden)])

# hidden=1
# 53.1%   #hidden_layer_sizes = tuple([(features//((i+2)**2)) for i in range(num_hidden)])

# hidden=3
# 53.1%   #hidden_layer_sizes = tuple([(features//((i+2)**2)) for i in range(num_hidden)])

# num 6000
# hidden=3
# 56.5%   #hidden_layer_sizes = tuple([(features//((i+2)**10)) for i in range(num_hidden)])

# alpha = 1
# num = 250, hidden_layer_sizes = (76,) == ~97
# num = 251, hidden_layer_sizes = (25,) == ~97
# num = 252, hidden_layer_sizes = (25,) == ~93
# num = 253, hidden_layer_sizes = (10,) == ~83
# num = 254, hidden_layer_sizes = (5,) == ~82
# num = 255, hidden_layer_sizes = (4,) == ~80
# num = 256, hidden_layer_sizes = (4,) == ~70
# num = 257, hidden_layer_sizes = (5,) == ~60
# alpha = 0.1
# num = 257, hidden_layer_sizes = (5,) == ~83
# num = 258, hidden_layer_sizes = (5,) == ~96
# num = 259, hidden_layer_sizes = (4,) == ~98
# num = 260, hidden_layer_sizes = (5,) == ~88

# num = 332
# alpha = 0.000001
# hidden = (5,)
# percent = 72.39
# random_state = 346

# removing low variance (0.99) helps a lot - 75.2% @ 332 samples
# after this, increasing max iterations to 500 (from 200) helps bacause it seems that is the new typical iteration count it gets to before it doesn't see improvement
# now adding samples improves the result a little - 76.9% @ 1000 samples
# now adding even more samples improves the result a little - 80.7% @ 6000 samples

num = 6000 #332 #6000

if False:
    alpha = 0.000001 #0.00001
    num_hidden = 1
    percent = 0
    random_state = 0
    while percent < 65:
        random_state += 1
        d = NN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], alpha, num_hidden, num, False, True, (5,), random_state)
        percent = d.Predict("../Sentiment/Sentiment_train_set_1.csv", num, False, "../Sentiment/Sentiment_NN_result_0a.csv")
        if percent >= 65:
            d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_NN_result_0b.csv")
        #d = NN(["../Sentiment/Sentiment_train_set_1.csv"], alpha, num_hidden, num*3, False, (5,), random_state)
        #percent = d.Predict("../Sentiment/Sentiment_train_set_1.csv", num*3, False, "../Sentiment/Sentiment_NN_result_0.csv")
    print(random_state)

if True:
    alpha = 0.0012
    num_hidden = 1
    percentList = [[0, 0, None, 1],[0, 1, None, 1],[0, 2, None, 1],[0, 3, None, 1]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment/Sentiment_train_set_{j}.csv")
            
        percent = 0
        random_state = 0
        while percent < 65:
            random_state += 1
            d = NN(fileList, alpha, num_hidden, num, False, True, (5,), random_state, 80)
            percent = d.Predict(fileList[0], num, False, f"../Sentiment/Sentiment_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sentiment/Sentiment_train_set_{i}.csv", num*3, False, f"../Sentiment/Sentiment_NN_result_{i}.csv")
        percentList[i-1][2] = d
        percentList[i-1][3] = random_state

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment/Sentiment_test_set.csv", num*3, False, "../Sentiment/Sentiment_NN_final.csv")
    print("Random Seed: " + str(percentList[0][3]))
    print("Percent Match: " + str(percent))

if False:
    alpha = 0.000001
    num_hidden = 1
    hls = 5
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = NN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], alpha, num_hidden, numb, False, True, (hls,), 0)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_NN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_NN_vsSize.csv", result, delimiter=",")

num = 2500

if False:
    alpha = 0.000001
    num_hidden = 1
    hls_min = 3
    hls_max = 11
    hls_step = 1

    hlsList = [[d for d in range(hls_min, hls_max, hls_step)], [0 for d in range(hls_min, hls_max, hls_step)]]
    i = 0
    for hls in np.arange(hls_min, hls_max, hls_step):
        d = NN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0)
        print("HLS: " + str(hls))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_NN_0.csv")
        hlsList[1][i] = percent
        i += 1
    result = np.asfarray(hlsList)
    np.savetxt("../Sentiment/Sentiment_result_NN_hls.csv", result, delimiter=",")

if False:
    alpha_min = 0.0001
    alpha_max = 0.0021
    alpha_step = 0.0001
    num_hidden = 1
    hls = 5

    alphaList = [[d for d in np.arange(alpha_min, alpha_max, alpha_step)], [0.0 for d in np.arange(alpha_min, alpha_max, alpha_step)]]
    i = 0
    for alpha in np.arange(alpha_min, alpha_max, alpha_step):
        d = NN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0)
        print("alpha: " + str(alpha))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_NN_0.csv")
        alphaList[1][i] = percent
        i += 1
    result = np.asfarray(alphaList)
    np.savetxt("../Sentiment/Sentiment_result_NN_alpha.csv", result, delimiter=",")

if False:
    iter_min = 5
    iter_max = 156
    iter_step = 10
    num_hidden = 1
    hls = 5
    alpha = 0.000001

    alphaList = [[d for d in np.arange(iter_min, iter_max, iter_step)], [0.0 for d in np.arange(iter_min, iter_max, iter_step)]]
    i = 0
    for iter in np.arange(iter_min, iter_max, iter_step):
        d = NN(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], alpha, num_hidden, num, False, True, (hls,), 0, iter)
        print("iter: " + str(iter))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_NN_0.csv")
        print("result: " + str(percent))
        alphaList[1][i] = percent
        i += 1
    result = np.asfarray(alphaList)
    np.savetxt("../Sentiment/Sentiment_result_NN_iter.csv", result, delimiter=",")

# Final 70%

################

num = 2500

if False:
    alpha = 0.000001
    num_hidden = 1
    percentList = [[0, 0, None, 1],[0, 1, None, 1],[0, 2, None, 1],[0, 3, None, 1]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment2/Sentiment_train_set_{j}.csv")
            
        percent = 0
        random_state = 0
        while percent < 65:
            random_state += 1
            d = NN(fileList, alpha, num_hidden, num, True, True, (5,), random_state)
            percent = d.Predict(fileList[0], num, True, f"../Sentiment2/Sentiment_NN_temp_{i}.csv")

        percentList[i-1][0] = d.Predict(f"../Sentiment2/Sentiment_train_set_{i}.csv", num*3, True, f"../Sentiment2/Sentiment_NN_result_{i}.csv")
        percentList[i-1][2] = d
        percentList[i-1][3] = random_state

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment2/Sentiment_test_set.csv", num*3, True, "../Sentiment2/Sentiment_NN_final.csv")
    print("Random Seed: " + str(percentList[0][3]))
    print("Percent Match: " + str(percent))