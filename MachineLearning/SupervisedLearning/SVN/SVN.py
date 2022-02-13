#
# swap out kernel functions. I'd like to see at least two
#

from sklearn import svm
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

class SVM:
    def __init__(self, file, method_func, maxLines, multipleText, removeLowVariance, random_state=None, kernel_func='rbf', tolerance=0.001, max_iter=-1):
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
        print("\nCreating the Vector Machine", flush=True)
        print("Num Features: " + str(len(self.dct.keys())))
        print("Num Samples: " + str(len(valueList)))
        print("Method Function: " + method_func)
        print("Kernel Function: " + kernel_func)
        start = time.perf_counter()
        if method_func == "SVC":
            clf = svm.SVC(cache_size=1000, kernel=kernel_func, tol=tolerance, verbose=True, random_state=random_state, max_iter=max_iter)
        elif method_func == "NuSVC":
            clf = svm.NuSVC(cache_size=1000, kernel=kernel_func, tol=tolerance, verbose=True, random_state=random_state, max_iter=max_iter)
        elif method_func == "LinearSVC":
            clf = svm.LinearSVC(verbose=True, tol=tolerance, random_state=random_state, max_iter=max_iter)
        if kernel_func == 'precomputed':
            self.clf = clf.fit(sampleList, sampleList)
        else:
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
    percent = 0
    random_state = 0
    while percent < 65:
        random_state += 1
        d = SVM(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], "LinearSVC", num, False, True, random_state)
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_1.csv", num, False, "../Sarcasm/Sarcasm_SVM_result_0a.csv")
        #if percent >= 65:
            #d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_SVM_result_0b.csv")
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

if False:
    percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
    for i in range(1,5):
        fileList = []
        for j in range(1,5):
            if i == j: continue
            fileList.append(f"../Sarcasm/Sarcasm_train_set_{j}.csv")
            
        random_state = 1
        d = SVM(fileList, "SVC", num, False, True, random_state, 'rbf', 0.037, -1)
        #d = SVM(fileList, "NuSVC", num, False, True, random_state)
        #d = SVM(fileList, "LinearSVC", num, False, True, random_state)

        percentList[i-1][0] = d.Predict(f"../Sarcasm/Sarcasm_train_set_{i}.csv", num*3, False, f"../Sarcasm/Sarcasm_SVM_LinearSVC_result_{i}.csv")
        percentList[i-1][2] = d

    percentList.sort(reverse=True)
    percent = percentList[0][2].Predict("../Sarcasm/Sarcasm_test_set.csv", num*3, False, "../Sarcasm/Sarcasm_SVM_LinearSVC_final.csv")
    print("Percent Match: " + str(percent))

if False:
    func = "SVC"
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500
    random_state = 1
    kernel_func='rbf'
    tolerance=0.001

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = SVM(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], func, numb, False, True, random_state, kernel_func, tolerance)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_SVN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_SVN_vsSize.csv", result, delimiter=",")
    
num = 4000

if False:
    func = "SVC"
    random_state = 1
    tolerance=0.001

    numList = [[d for d in range(0, 4, 1)], [0 for d in range(0, 4, 1)]]
    i = 0
    for kernel_func in ['linear', 'poly', 'rbf', 'sigmoid']:
        d = SVM(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance)
        print("Kernel: " + kernel_func)
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_SVN_kernel.csv", result, delimiter=",")

if False:
    func = "SVC"
    kernel_func='rbf'
    random_state = 1
    min_tol=0.001
    max_tol=0.041
    step_tol=0.002

    numList = [[d for d in np.arange(min_tol, max_tol, step_tol)], [0.0 for d in np.arange(min_tol, max_tol, step_tol)]]
    i = 0
    for tolerance in np.arange(min_tol, max_tol, step_tol):
        d = SVM(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance)
        print("Tolerance: " + str(tolerance))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_SVN_tolerance.csv", result, delimiter=",")

if True:
    func = "SVC"
    testMinIter = 1000
    testMaxIter = 8000
    testStepIter = 1000
    random_state = 1
    kernel_func='rbf'
    tolerance=0.001

    numList = [[d for d in range(testMinIter, testMaxIter, testStepIter)], [0 for d in range(testMinIter, testMaxIter, testStepIter)]]
    i = 0
    for iter in np.arange(testMinIter, testMaxIter, testStepIter):
        d = SVM(["../Sarcasm/Sarcasm_train_set_1.csv","../Sarcasm/Sarcasm_train_set_2.csv","../Sarcasm/Sarcasm_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance, iter)
        print("Iter: " + str(iter))
        percent = d.Predict("../Sarcasm/Sarcasm_train_set_4.csv", num*3, False, "../Sarcasm/Sarcasm_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sarcasm/Sarcasm_result_SVN_iter.csv", result, delimiter=",")

##############

num = 6000 #332 #6000

if False:
    percent = 0
    random_state = 0
    while percent < 65:
        random_state += 1
        d = SVM(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], "SVC", num, False, True, random_state)
        percent = d.Predict("../Sentiment/Sentiment_train_set_1.csv", num, False, "../Sentiment/Sentiment_SVM_result_0a.csv")
        if percent >= 65:
            d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_SVM_result_0b.csv")
    print(random_state)

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

if False:
    for k in ["SVC"]: #, "NuSVC", "LinearSVC"
        percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
        for i in range(1,5):
            fileList = []
            for j in range(1,5):
                if i == j: continue
                fileList.append(f"../Sentiment/Sentiment_train_set_{j}.csv")
            
            random_state = 1
            d = SVM(fileList, k, num, False, True, random_state, 'rbf', 0.011, -1)

            percentList[i-1][0] = d.Predict(f"../Sentiment/Sentiment_train_set_{i}.csv", num*3, False, f"../Sentiment/Sentiment_SVM_{k}_result_{i}.csv")
            percentList[i-1][2] = d

        percentList.sort(reverse=True)
        percent = percentList[0][2].Predict("../Sentiment/Sentiment_test_set.csv", num*3, False, f"../Sentiment/Sentiment_SVM_{k}_final.csv")
        print("Percent Match: " + str(percent))

if False:
    func = "SVC"
    testMinNum = 500
    testMaxNum = 6100
    testNumStep = 500
    random_state = 1
    kernel_func='rbf'
    tolerance=0.001

    numList = [[d for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)], [0 for d in range(testMinNum, testMaxNum, testNumStep)]]
    i = 0
    for numb in np.arange(testMinNum, testMaxNum, testNumStep):
        d = SVM(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], func, numb, False, True, random_state, kernel_func, tolerance)
        print("Num: " + str(numb))
        start = time.perf_counter()
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_SVN_0.csv")
        totalTime = time.perf_counter() - start
        print("Total Time: " + str(totalTime))
        numList[1][i] = percent
        numList[2][i] = totalTime
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_SVN_vsSize.csv", result, delimiter=",")
    
num = 3500

if False:
    func = "SVC"
    random_state = 1
    tolerance=0.001

    numList = [[d for d in range(0, 4, 1)], [0 for d in range(0, 4, 1)]]
    i = 0
    for kernel_func in ['linear', 'poly', 'rbf', 'sigmoid']:
        d = SVM(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance)
        print("Kernel: " + kernel_func)
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_SVN_kernel.csv", result, delimiter=",")

if False:
    func = "SVC"
    kernel_func='rbf'
    random_state = 1
    min_tol=0.001
    max_tol=0.041
    step_tol=0.002
    
    numList = [[d for d in np.arange(min_tol, max_tol, step_tol)], [0.0 for d in np.arange(min_tol, max_tol, step_tol)]]
    i = 0
    for tolerance in np.arange(min_tol, max_tol, step_tol):
        d = SVM(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance)
        print("Tolerance: " + str(tolerance))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_SVN_tolerance.csv", result, delimiter=",")

if True:
    func = "SVC"
    testMinIter = 1000
    testMaxIter = 8000
    testStepIter = 1000
    random_state = 1
    kernel_func='rbf'
    tolerance=0.001

    numList = [[d for d in range(testMinIter, testMaxIter, testStepIter)], [0 for d in range(testMinIter, testMaxIter, testStepIter)]]
    i = 0
    for iter in np.arange(testMinIter, testMaxIter, testStepIter):
        d = SVM(["../Sentiment/Sentiment_train_set_1.csv","../Sentiment/Sentiment_train_set_2.csv","../Sentiment/Sentiment_train_set_3.csv"], func, num, False, True, random_state, kernel_func, tolerance, iter)
        print("Iter: " + str(iter))
        percent = d.Predict("../Sentiment/Sentiment_train_set_4.csv", num*3, False, "../Sentiment/Sentiment_result_SVN_0.csv")
        numList[1][i] = percent
        i += 1
    result = np.asfarray(numList)
    np.savetxt("../Sentiment/Sentiment_result_SVN_iter.csv", result, delimiter=",")

################

num = 2500

if False:
    for k in ["LinearSVC"]:
        percentList = [[0, 0, None],[0, 1, None],[0, 2, None],[0, 3, None]]
        for i in range(1,5):
            fileList = []
            for j in range(1,5):
                if i == j: continue
                fileList.append(f"../Sentiment2/Sentiment_train_set_{j}.csv")
            
            random_state = 1
            d = SVM(fileList, k, num, True, True, random_state)

            percentList[i-1][0] = d.Predict(f"../Sentiment2/Sentiment_train_set_{i}.csv", num*3, True, f"../Sentiment2/Sentiment_SVM_{k}_result_{i}.csv")
            percentList[i-1][2] = d

        percentList.sort(reverse=True)
        percent = percentList[0][2].Predict("../Sentiment2/Sentiment_test_set.csv", num*3, True, f"../Sentiment2/Sentiment_SVM_{k}_final.csv")
        print("Percent Match: " + str(percent))
