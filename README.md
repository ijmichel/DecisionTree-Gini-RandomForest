# DecisionTree-Gini-RandomForest

This code implements a Decision Tree based on Gini index.  Also it has the ability to run a Random Forest ensemble method
for N number of trees


How to run:

python DecisionTree.py ~/data/led.train ~/data/led.train 

python RandomForest.py ~/data/led.train ~/data/led.test 

This will output the confusion matrix.   To get accuracy from that run with a pipe to accuracy.py:

python DecisionTree.py ~/data/balance.scale.train ~/data/balance.scale.train | python accuracy.py 

To change the number of trees and F value for Random Forest alter this section of code below.  The first value is
the F value, the second is the # of trees:

```
def getRunParameters (trainPath):  
  if "nursery" in trainPath:   
    return 7,25        
  elif "synthetic.social" in trainPath:    
    return 8,20    
  elif "led" in trainPath
    return 7,20     
  elif "balance.scale" in trainPath:
    return 1,93        
```

Or you can use this code to do it automatically to run tests:

RandomForestIterate.py

python RandomForestIterate.py ~/nursery.train ~/nursery.test 1 10 1 70

The above executes the code to build a tree on nursery,train and test with nursery.test, with 1 to 10 F values and 1 to 80 trees
