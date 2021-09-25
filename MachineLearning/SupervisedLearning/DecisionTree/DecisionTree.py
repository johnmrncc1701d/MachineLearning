
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import csv
import time

_printCtr = 0
def PrintDot():
    global _printCtr
    _printCtr += 1
    if _printCtr >= 10:
        _printCtr = 0
        print(".", end="", flush=True)

class DTree:
    def __init__(self, file, criterion, max_features, pruneValue, maxLines, maxDepth, min_leaf_samples, removeLowVariance, multipleText):
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
        print("\nCreating the Tree", flush=True)
        start = time.perf_counter()
        clf = tree.DecisionTreeClassifier(criterion=criterion, ccp_alpha=pruneValue, max_features=max_features, max_depth=maxDepth, min_samples_leaf=min_leaf_samples)
        self.clf = clf.fit(sampleList, valueList)
        print("Time: " + str(time.perf_counter() - start))
        print("Num Features: " + str(len(self.dct.keys())))
        print("Num Samples: " + str(len(valueList)))
        print("Tree Depth: " + str(self.clf.tree_.max_depth), flush=True)

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
    
num = 6000 #6000

# Single test run to try different values for sarcasm set
if False:
    depth = None #250
    prune = 0.0005 #0.0007
    #d = DTree(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, False)
    d = DTree(["../Sarcasm/Sarcasm_train_set_1.csv"], "entropy", None, prune, num, depth, 1, False, False)
    percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num, False, "../Sarcasm/Sarcasm_result_0.csv")
    
# Testing multiple combinations of prune and max depth for sarcasm set
if False:
    testMinPrune = 0.0001
    testMaxPrune = 0.0020
    testPruneStep = 0.0001
    testMinDepth = 50
    testMaxDepth = 350
    testDepthStep = 5

    pruneList = [[d for d in range(testMinDepth, testMaxDepth, testDepthStep)]]
    for prune in np.arange(testMinPrune, testMaxPrune, testPruneStep):
        depthList = [prune]
        for depth in np.arange(testMinDepth, testMaxDepth, testDepthStep):
            d = DTree(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, False)
            print("Max Depth: " + str(depth))
            print("Prune: " + str(prune))
            percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num, False, "../Sarcasm/Sarcasm_result_0.csv")
            depthList.append(percent)
        pruneList.append(depthList)
    pruneList[0].insert(0,0)
    result = np.asfarray(pruneList)
    np.savetxt("../Sarcasm/Sarcasm_result_grid.csv", result, delimiter=",")
    
# The final test run for sarcasm set
if False:
    depth = 250
    prune = 0.0007
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sarcasm/Sarcasm_train_set_{j}.csv")
        d = DTree(fileList, "entropy", None, prune, num, depth, 1, False, False)
        percentList[i-1][2] = d
        percent = d.Predict(f"../Sarcasm/Sarcasm_train_set_{i}.csv", num, False, f"../Sarcasm/Sarcasm_result_{i}.csv")
        percentList[i-1][0] += percent
    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sarcasm/Sarcasm_test_set.csv", num, False, "../Sarcasm/Sarcasm_final.csv")
    print("Tree Depth: " + str(percentList[0][2].clf.tree_.max_depth))
    print("Percent Match: " + str(percent))

# SARCASM TESTS
# Depth of the tree was 304
# Looks like some very minor improvement (1%) to limit the tree to 200 depth
# With perfect overfitting we still get about 75% correctness
# Removing features with low variance tends to get worse results
# min_samples_leaf=1 seems to be best
# max features - log2 performed worse, sqrt performed better, None performes best
# wow with pruning @ 0.0015 I get a depth of 37 with 72% matching
# wow with pruning @ 0.001 I get a depth of 75 with 74% matching
# with pruning @ 0.0005 it goes off the rails with a depth of 356
# limiting to 200 max with pruning @ 0.0005 it goes to depth 175 and gets 75% matching
# limiting to 150 max with pruning @ 0.0005 it goes to depth 150 and gets 75% matching (slightly better)
# finally after running tests for depth cross pruning I get the most ideal location at depth 250 and prune 0.0007; This produces a tree of depth 129.
# final percent is 76.17978 with tree depth of 70

num = 6000

# Single test run to try different values for sentiment set with sentences
if False:
    depth = 410
    prune = 0.00048
    d = DTree(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, False)
    percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num, False, "../Sentiment/Sentiment_result_0.csv")
    
# Testing multiple combinations of prune and max depth for sentiment set with sentences
if False:
    testMinPrune = 0.0005
    testMaxPrune = 0.001
    testPruneStep = 0.00005
    testMinDepth = 410
    testMaxDepth = 411
    testDepthStep = 20

    pruneList = [[d for d in range(testMinDepth, testMaxDepth, testDepthStep)]]
    for prune in np.arange(testMinPrune, testMaxPrune, testPruneStep):
        depthList = [prune]
        for depth in np.arange(testMinDepth, testMaxDepth, testDepthStep):
            d = DTree(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, False)
            print("Max Depth: " + str(depth))
            print("Prune: " + str(prune))
            percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num, False, "../Sentiment/Sentiment_result_0.csv")
            depthList.append(percent)
        pruneList.append(depthList)
    pruneList[0].insert(0,0)
    result = np.asfarray(pruneList)
    np.savetxt("../Sentiment/Sentiment_result_grid_2.csv", result, delimiter=",")
    
# The final test run for sentiment set with sentences
if False:
    depth = 410
    prune = 0.00048
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment/Sentiment_train_set_{j}.csv")
        d = DTree(fileList, "entropy", None, prune, num, depth, 1, False, False)
        percentList[i-1][2] = d
        percent = d.Predict(f"../Sentiment/Sentiment_train_set_{i}.csv", num, False, f"../Sentiment/Sentiment_result_{i}.csv")
        percentList[i-1][0] += percent
    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment/Sentiment_test_set.csv", num, False, "../Sentiment/Sentiment_final.csv")
    print("Tree Depth: " + str(percentList[0][2].clf.tree_.max_depth))
    print("Percent Match: " + str(percent))

# SENTIMENT TESTS
# Num Features: 38956
# Num Samples: 18000
# Tree Depth: 580
# Percent: 64
# Min Leaf Samples: 1

# Num Features: 3966
# Num Samples: 18000
# Tree Depth: 679
# Percent: 61
# Removed Low Variance

# Final
# Max Depth: 410
# Tree Depth: 53
# Prune: 0.00048
# Percent Match: 64.95



num = 3000

# Single test run to try different values for sentiment set with authors
if False:
    depth = 410
    prune = 0.00006 #0.00006 #65.4 #0.00012 #64.2 #0.00024
    d = DTree(["../Sentiment2/Sentiment_train_set_1.csv","../Sentiment2/Sentiment_train_set_2.csv","../Sentiment2/Sentiment_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, True)
    percent = d.Predict("../Sentiment2/Sentiment_train_set_4.csv", num, True, "../Sentiment2/Sentiment_result_0.csv")
    
# Testing multiple combinations of prune and max depth for sentiment set with authors
if False:
    testMinPrune = 0.00006
    testMaxPrune = 0.00007
    testPruneStep = 0.00001
    testMinDepth = 50
    testMaxDepth = 550
    testDepthStep = 50

    pruneList = [[d for d in range(testMinDepth, testMaxDepth, testDepthStep)]]
    for prune in np.arange(testMinPrune, testMaxPrune, testPruneStep):
        depthList = [prune]
        for depth in np.arange(testMinDepth, testMaxDepth, testDepthStep):
            d = DTree(["../Sentiment2/Sentiment_train_set_1.csv","../Sentiment2/Sentiment_train_set_2.csv","../Sentiment2/Sentiment_train_set_3.csv"], "entropy", None, prune, num, depth, 1, False, True)
            print("Max Depth: " + str(depth))
            print("Prune: " + str(prune))
            percent = d.Predict("../Sentiment2/Sentiment_train_set_4.csv", num, True, "../Sentiment2/Sentiment_result_0.csv")
            depthList.append(percent)
        pruneList.append(depthList)
    pruneList[0].insert(0,0)
    result = np.asfarray(pruneList)
    np.savetxt("../Sentiment2/Sentiment_result_grid.csv", result, delimiter=",")
    
# The final test run for sentiment set with authors
if False:
    depth = 410
    prune = 0.00006
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment2/Sentiment_train_set_{j}.csv")
        d = DTree(fileList, "entropy", None, prune, num, depth, 1, False, True)
        percentList[i-1][2] = d
        percent = d.Predict(f"../Sentiment2/Sentiment_train_set_{i}.csv", num, True, f"../Sentiment2/Sentiment_result_{i}.csv")
        percentList[i-1][0] += percent
    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment2/Sentiment_test_set.csv", num, True, "../Sentiment2/Sentiment_final.csv")
    print("Tree Depth: " + str(percentList[0][2].clf.tree_.max_depth))
    print("Percent Match: " + str(percent))

# SENTIMENT 2 TESTS

# Final
# Max Depth: 410
# Tree Depth: 410
# Prune: 0.00006
# Percent Match: 65.07