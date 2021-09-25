
import json
import csv
import random

if False:
    SarcasmFile = None

    def parse_data(file):
        for l in open(file,'r'):
            yield json.loads(l)

    SarcasmFile = list(parse_data('../Sarcasm/Sarcasm_Headlines_Dataset.json'))

    random.shuffle(SarcasmFile)

    def WriteSarcLine(i):
        global SarcasmFile
        s = SarcasmFile[i]["is_sarcastic"]
        if s == 1: s = 2
        f.write('"' + SarcasmFile[i]["headline"] + '",' + str(s) + '\n')

    ln = len(SarcasmFile)
    lines = ln // 10
    with open('../Sarcasm/Sarcasm_test_set.csv','w') as f:
        for i in range(lines):
            WriteSarcLine(i)

    start = lines
    fold = (ln - lines + 3) // 4
    for j in range(1,5):
        if fold > ln - start:
            fold = ln - start
        with open('../Sarcasm/Sarcasm_train_set_' + str(j) + '.csv','w') as f:
            for i in range(start, start + fold):
                WriteSarcLine(i)
        start = start + fold
        
if False:
    SentimentFile = []

    with open('../Sentiment/training.1600000.processed.noemoticon.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            SentimentFile.append(row)

    random.shuffle(SentimentFile)

    def WriteSentLine(i):
        global SentimentFile
        s = int(SentimentFile[i][0])
        if s == 2: s = 1
        if s == 4: s = 2
        f.write('"' + SentimentFile[i][5] + '",' + str(s) + '\n')

    ln = len(SentimentFile)
    lines = ln // 10
    with open('../Sentiment/Sentiment_test_set.csv','w') as f:
        for i in range(lines):
            WriteSentLine(i)

    start = lines
    fold = (ln - lines + 3) // 4
    for j in range(1,5):
        if fold > ln - start:
            fold = ln - start
        with open('../Sentiment/Sentiment_train_set_' + str(j) + '.csv','w') as f:
            for i in range(start, start + fold):
                WriteSentLine(i)
        start = start + fold

if True:
    SentimentDict = {}
    SentimentResult = {}

    with open('../Sentiment/training.1600000.processed.noemoticon.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if row[4] in SentimentDict:
                SentimentDict[row[4]].append(row[5])
            else:
                SentimentDict[row[4]] = [row[5]]
                s = int(row[0])
                if s == 2: s = 1
                if s == 4: s = 2
                SentimentResult[row[4]] = s

    SentimentKeys = list(SentimentDict.keys())
    random.shuffle(SentimentKeys)

    def WriteSent2Line(k):
        global SentimentDict
        global SentimentResult
        f.write('"' + k + '",' + str(SentimentResult[k]))
        for i in SentimentDict[k]:
            f.write(',"' + i + '"')
        f.write('\n')

    ln = len(SentimentKeys)
    lines = ln // 10
    with open('../Sentiment2/Sentiment_test_set.csv','w') as f:
        for i in range(lines):
            WriteSent2Line(SentimentKeys[i])

    start = lines
    fold = (ln - lines + 3) // 4
    for j in range(1,5):
        if fold > ln - start:
            fold = ln - start
        with open('../Sentiment2/Sentiment_train_set_' + str(j) + '.csv','w') as f:
            for i in range(start, start + fold):
                WriteSent2Line(SentimentKeys[i])
        start = start + fold