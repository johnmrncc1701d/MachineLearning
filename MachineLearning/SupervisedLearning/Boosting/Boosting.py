#
# use some form of pruning
#

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
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

class EL_Boosting:
    def __init__(self, file, maxLines, pruneValue, maxDepth, min_samples_split, multipleText, removeLowVariance, estimators=100, random_state=None):
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
            sel = VarianceThreshold(threshold=(.999 * (1 - .999)))
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
        print("\nCreating the Boost Learner", flush=True)
        print("Num Features: " + str(len(self.dct.keys())))
        print("Num Samples: " + str(len(valueList)))
        print("Estimators: " + str(estimators))
        start = time.perf_counter()
        dt = tree.DecisionTreeClassifier(criterion="entropy", ccp_alpha=pruneValue, max_depth=maxDepth, min_samples_split=min_samples_split)
        clf = AdaBoostClassifier(base_estimator=dt, n_estimators=estimators,  random_state=random_state)
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

if True:
    depth = 3
    prune = 0.0016
    percent = 0
    random_state = 0
    while percent < 65:
        random_state += 1
        d = EL_Boosting(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], num, prune, depth, 2, False, True, 100, random_state)
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_1.csv", num, False, "../Sarcasm/Sarcasm_ELB_result_0a.csv")
        #if percent >= 65:
            #d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_ELB_result_0b.csv")
    print(random_state)
    
if False:
    depth_min = 1
    depth_max = 5
    depth_step = 1
    prune_min = 0.0001
    prune_max = 0.002
    prune_step = 0.0001
    percent = 0
    random_state = 1
    
    hlsList = [[d for d in np.arange(prune_min, prune_max, prune_step)]]
    hlsList[0].insert(0,0)
    for depth in np.arange(depth_min, depth_max, depth_step):
        numList = [depth]
        for prune in np.arange(prune_min, prune_max, prune_step):
            d = EL_Boosting(["../Sarcasm/Sarcasm_train_set_1.csv"], num, prune, depth, 2, False, True, 100, random_state)
            print("Max Depth: " + str(depth))
            print("Prune: " + str(prune))
            percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num, False, "../Sarcasm/Sarcasm_ELB_result_0.csv")
            numList.append(percent)
        hlsList.append(numList)
    result = np.asfarray(hlsList)
    np.savetxt("../Sarcasm/Sarcasm_ELB_result_grid.csv", result, delimiter=",")

if False:
    depth = 3
    prune = 0.0016
    random_state = 1
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sarcasm/Sarcasm_train_set_{j}.csv")
            
        d = EL_Boosting(fileList, num, prune, depth, 2, False, True, 100, random_state)
        percentList[i-1][0] = d.Predict(f"../Sarcasm/Sarcasm_train_set_{i}.csv", num*3, False, f"../Sarcasm/Sarcasm_ELB_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sarcasm/Sarcasm_test_set.csv", num*3, False, "../Sarcasm/Sarcasm_ELB_final.csv")
    print("Percent Match: " + str(percent))
    
# removing low variance speeds it up but doesn't seem to improve or worsen the results much
# at 10 iterations ~ 72.5%
# at 100 iterations ~ 80%

################

num = 6000 #332 #6000
    
if False:
    depth_min = 1
    depth_max = 5
    depth_step = 1
    prune_min = 0.0001
    prune_max = 0.002
    prune_step = 0.0001
    percent = 0
    random_state = 1
    
    hlsList = [[d for d in np.arange(prune_min, prune_max, prune_step)]]
    hlsList[0].insert(0,0)
    for depth in np.arange(depth_min, depth_max, depth_step):
        numList = [depth]
        for prune in np.arange(prune_min, prune_max, prune_step):
            d = EL_Boosting(["../Sentiment/Sentiment_train_set_1.csv"], num, prune, depth, 2, False, True, 100, random_state)
            print("Max Depth: " + str(depth))
            print("Prune: " + str(prune))
            percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num, False, "../Sentiment/Sentiment_ELB_result_0.csv")
            numList.append(percent)
        hlsList.append(numList)
    result = np.asfarray(hlsList)
    np.savetxt("../Sentiment/Sentiment_ELB_result_grid.csv", result, delimiter=",")

if False:
    depth = 3
    prune = 0.0008
    random_state = 1
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment/Sentiment_train_set_{j}.csv")
            
        d = EL_Boosting(fileList, num, prune, depth, 2, False, True, 100, random_state)
        percentList[i-1][0] = d.Predict(f"../Sentiment/Sentiment_train_set_{i}.csv", num*3, False, f"../Sentiment/Sentiment_ELB_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment/Sentiment_test_set.csv", num*3, False, "../Sentiment/Sentiment_ELB_final.csv")
    print("Percent Match: " + str(percent))
    
################

num = 2500

if False:
    depth = 3
    prune = 0.0008
    random_state = 1
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sentiment2/Sentiment_train_set_{j}.csv")
            
        d = EL_Boosting(fileList, num, prune, depth, 2, True, True, 100, random_state)
        percentList[i-1][0] = d.Predict(f"../Sentiment2/Sentiment_train_set_{i}.csv", num*3, True, f"../Sentiment2/Sentiment_ELB_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sentiment2/Sentiment_test_set.csv", num*3, True, "../Sentiment2/Sentiment_ELB_final.csv")
    print("Percent Match: " + str(percent))
