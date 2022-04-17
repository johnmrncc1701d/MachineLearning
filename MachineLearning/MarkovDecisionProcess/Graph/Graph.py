
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


if True:
    fullData = []
    with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\FishQLearnTime_EpsilonGreedy.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData.append(np.array(row, dtype=np.float))
    with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\FishQLearnTime_GreedyDetQ.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData.append(np.array(row, dtype=np.float))
    with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\FishQLearnTime_GreedyQ.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData.append(np.array(row, dtype=np.float))
    with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\FishQLearnTime_BzmnQ.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            fullData.append(np.array(row, dtype=np.float))

    fig, ax = plt.subplots()
    ax.plot(fullData[0], label="Epsilon Greedy")
    ax.plot(fullData[1], label="Greedy Deterministic Q")
    ax.plot(fullData[2], label="Greedy Q")
    ax.plot(fullData[3], label="Boltzmann Q")
    ax.set_yscale('log')
    plt.ylabel("Time (seconds)")
    plt.xlabel("Trial #")
    plt.legend()
    plt.show()

if False:
    for index in range(1,4):
        for typ in ["Value_Iteration", "Policy_Iteration", "EpsilonGreedy", "GreedyQ", "GreedyDetQ"]:
            size = 0 # 501, 407, 288
            if index == 1: size = 501
            elif index == 2: size = 407
            elif index == 3: size = 288
            grid = [["#" for j in range(size)] for i in range(size)]
            with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridStates_' + typ + '_' + str(index) + '.csv', 'r', newline='') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for row in reader:
                    state = row[0]
                    dir = row[1]

                    pos = state.find("x: {") + 4
                    pos2 = state.find("}", pos)
                    x = int(state[pos:pos2])

                    pos = state.find("y: {", pos2) + 4
                    pos2 = state.find("}", pos)
                    y = int(state[pos:pos2])
                    y = size - y - 1

                    d = " "
                    if dir == "north": d = "^"
                    elif dir == "south": d = "v"
                    elif dir == "east": d = ">"
                    elif dir == "west": d = "<"

                    grid[y][x] = d
            with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridDirs_' + typ + '_' + str(index) + '.txt', 'w') as f:
                for y in grid:
                    for x in y:
                        f.write(x)
                    f.write('\n')

                    
if False:
    for index in range(1,4):
        size = 0 # 501, 407, 288
        if index == 1: size = 501
        elif index == 2: size = 407
        elif index == 3: size = 288
        fullData = []
        with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridLearnTime_EpsilonGreedy_' + str(index) + '.csv', 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                fullData.append(np.array(row, dtype=np.float))
        with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridLearnTime_GreedyDetQ_' + str(index) + '.csv', 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                fullData.append(np.array(row, dtype=np.float))
        with open('D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridLearnTime_GreedyQ_' + str(index) + '.csv', 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                fullData.append(np.array(row, dtype=np.float))

        fig, ax = plt.subplots()
        ax.plot(fullData[0], label="Epsilon Greedy")
        ax.plot(fullData[1], label="Greedy Deterministic Q")
        ax.plot(fullData[2], label="Greedy Q")
        ax.set_yscale('log')
        plt.ylabel("Time (seconds)")
        plt.xlabel("Trial #")
        plt.title(str(size) + "x" + str(size) + " Grid")
        plt.legend()
        plt.show()