import DecisionTreeAlgorithm
import sys
import accuracyNew
import time

train = sys.argv[1]
test = sys.argv[2]
FmaxMin = sys.argv[3]
FmaxMax = sys.argv[4]
numTreesMin = sys.argv[5]
numTreesMax = sys.argv[6]

print train
print "-----------"
for F in range(int(FmaxMin),int(FmaxMax)):
    for numTrees in range(int(numTreesMin),int(numTreesMax)):
        start = time.time()
        acc = accuracyNew.getAccuracy(DecisionTreeAlgorithm.runDecisionTree("RandomForest",train,test,F,numTrees))
        end = time.time()
        print str(F), str(numTrees), str(acc), str(float(float(end-start) / 60))