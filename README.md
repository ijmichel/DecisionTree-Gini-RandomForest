# DecisionTree-Gini-RandomForest

This code implements a Decision Tree based on Gini index.  Also it has the ability to run a Random Forest ensemble method
for N number of trees


How to run:

python DecisionTree.py ~/data/led.train ~/data/led.train 

python RandomForest.py ~/data/led.train ~/data/led.test 

This will output the confusion matrix.   To get accuracy from that run with a pipe to accuracy.py:

python DecisionTree.py ~/data/balance.scale.train ~/data/balance.scale.train | python accuracy.py 

To change the number of trees and F value for Random Forest alter this section of code:

