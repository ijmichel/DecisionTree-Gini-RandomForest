
# coding: utf-8

# In[487]:


import sys
import random 
import threading
import random
import time

from collections import OrderedDict

# In[488]:


class ClassItem:
    def __init__(self):
        self.classLabel = ""
        self.classAttributesAndValue = OrderedDict()
    
    def __repr__(self):
        rep = self.classLabel + "-->"
        for attribute in self.classAttributesAndValue:
            rep += "[%s:%s]'" % (attribute,self.classAttributesAndValue[attribute])
        rep+="\n"
        return rep
    
    def addClassAttributeAndValue(self,attribute,value):
        self.classAttributesAndValue.update({attribute:value})
    


# In[489]:


class QuestionValueItem:
    def __init__(self, attribute, attValue):
        self.attValue = attValue
        self.attribute = attribute
        
    def match(self, classItemToCompareValueTo):
        valOfToCompareTo = classItemToCompareValueTo.classAttributesAndValue[self.attribute]
        return valOfToCompareTo == self.attValue
    
    def __repr__(self):
        return "Is %s = %s?" % (self.attValue,self.attribute)


# In[490]:


class DataSet:            
    def __init__(self,fileName,dataLines,type,randomForestF,parseOnly,setOnly):
        self.type = type
        self.randomForestF = randomForestF
        self.dataLines = set()
        self.attributes = set()
        self.attributesByValues= OrderedDict()   
        self.allLabels = set()        
        self.nodeStructure = None        

        if setOnly:
            self.dataLines = dataLines
        elif parseOnly:
            self.readDataFile(fileName)
        else:
            if fileName is None:
                self.dataLines = list(dataLines)
                self.calculateAttributesByValue()
                #print self.attributes
                self.nodeStructure = self.buildNodeStructure(None)
            else:
                #print "Reading data..."
                self.readDataFile(fileName)
                #print "File: ",fileName,"|", self.allLabels
                #print "Done"
                if "train" in fileName:
                    self.nodeStructure = self.buildNodeStructure(None)
                
    def __repr__(self):
        return "\ndataLines: %s \nattributes: %s \nattributesByValues: %s" % (
            self.dataLines, 
            self.attributes,
            self.attributesByValues,
        )
        
    def calculateAttributesByValue(self):
        for classItem in self.dataLines:
            for attribute in classItem.classAttributesAndValue:
                self.attributes.add(attribute)
                self.updateAttributesByValue(attribute, classItem.classAttributesAndValue[attribute])
    
    def updateAttributesByValue(self,attribute,value):                     
        #build attribute by values in entire data file     
        if attribute in self.attributesByValues:
            values = self.attributesByValues[attribute]
            values.add(value)
        else: 
            values = set()
            values.add(value)

        self.attributesByValues.update({attribute:values}) 
    
    def updateAllLabels(self,label):
        if label not in self.allLabels:
            self.allLabels.add(label)          
            # print "Add : ", self.allLabels
    
    def addClassItem(self,classItem):
        self.dataLines.add(classItem)            
            
    def readDataFile(self,fileName):
        with open(fileName) as f:
            for line in f:   
                data = line.split()
                classLabel = data[0]
                self.updateAllLabels(classLabel)
                classItem = ClassItem()
                classItem.classLabel = classLabel
                for i,classDefLine in enumerate(data): 
                    if i > 0:
                        attributeAndValue = classDefLine.split(":")
                        attribute = attributeAndValue[0]
                        self.attributes.add(attribute)
                        value = attributeAndValue[1]

                        #build single line values

                        self.updateAttributesByValue(attribute, value)
                        classItem.addClassAttributeAndValue(attribute,value)

                self.addClassItem(classItem)
                
    def getLabelCounts(self,dataLines):
        labelCounts = dict()
        for classItem in dataLines:
            label = classItem.classLabel
            if label in labelCounts:
                labelCounts[label] = labelCounts[label] + 1
            else:
                labelCounts[label] = 1     
        return labelCounts
    
    def getMyLabelCounts(self):
        return self.getLabelCounts(self.dataLines)
        
    
    def calculateGiniForDataLines(self,dataLines):
        labelCounts = self.getLabelCounts(dataLines)
        sum=0
        for label in labelCounts:      
            sum1 = (float(labelCounts[label])/float(len(dataLines)))**2
            sum = sum + sum1
        return 1-sum 

    def calculateGini(self):
        return self.calculateGiniForDataLines(self.dataLines)
    
    def calculateInformationGain(self,matchingDataLines, nonMatchingDataLines, currentGini):

        propbabilityOfMatchingDataLines = float(len(matchingDataLines)) / (len(matchingDataLines) + len(nonMatchingDataLines))
        ginForMatching = self.calculateGiniForDataLines(matchingDataLines)
        ginForNonMatching = self.calculateGiniForDataLines(nonMatchingDataLines) 
        return currentGini - propbabilityOfMatchingDataLines * ginForMatching - (1 - propbabilityOfMatchingDataLines) * ginForNonMatching
    
    def findBestQuestionAndGin(self):
        currentGini = self.calculateGiniForDataLines(self.dataLines)
        bestAttributeValueGain = 0
        bestAttributeValue = None
        attributesToUse = set()
        # print "----"
        # print len(self.attributes)
        
        if self.type == "RandomForest":  
            numAttributesToGrab = self.randomForestF
    
            if self.randomForestF > len(self.attributes) :
                numAttributesToGrab = len(self.attributes)
  
            if len(self.attributes) > 0 :
                for i in range(numAttributesToGrab):
                    randAttribute = random.choice(tuple(self.attributes))
                    attributesToUse.add(randAttribute) 
        else:
            attributesToUse = self.attributes

        # print len(attributesToUse)
        for anAttributeInData in attributesToUse:
            if anAttributeInData in self.attributesByValues:
                valuesForAttribute = self.attributesByValues[anAttributeInData]
                for aValueOfAattribute in valuesForAttribute:
                    questionValue = QuestionValueItem(anAttributeInData, aValueOfAattribute)
                    matchingDataLines, nonMatchingDataLines = self.splitItDataOnQuestions(questionValue)

                    if len(matchingDataLines) == 0 or len(nonMatchingDataLines) == 0:
                        continue

                    infoGain = self.calculateInformationGain(matchingDataLines, nonMatchingDataLines, currentGini)

                    if infoGain > bestAttributeValueGain:
                        bestAttributeValueGain = infoGain
                        bestAttributeValue = questionValue
                        
        return bestAttributeValueGain,bestAttributeValue
                        
                                
    def splitItDataOnQuestions(self,questionValCombo):
        matchingDataLines = []
        nonMatchingDataLines = []
        for classItem in self.dataLines:
            if questionValCombo.match(classItem):
                matchingDataLines.append(classItem)
            else:
                nonMatchingDataLines.append(classItem)
        return matchingDataLines, nonMatchingDataLines     
    
    def buildNodeStructure(self,dataSet):
        if(dataSet is None):
            dataSet = self
            
        bestAttributeValueGain,bestAttributeValue = dataSet.findBestQuestionAndGin()
        
        if bestAttributeValueGain ==0:
            return EndNode(dataSet)
        
        matchingDataLines, nonMatchingDataLines = dataSet.splitItDataOnQuestions(bestAttributeValue)

        matchingDataSet = DataSet(None,matchingDataLines,dataSet.type,dataSet.randomForestF,False,True)
        matchingDataSet.attributes = dataSet.attributes
        matchingDataSet.attributesByValues = dataSet.attributesByValues
        matchingNodeStructure = matchingDataSet.buildNodeStructure(None)
        matchingDataSet.nodeStructure = matchingNodeStructure

        nonMatchingDataSet = DataSet(None,nonMatchingDataLines,dataSet.type,dataSet.randomForestF,False,True)
        nonMatchingDataSet.attributes = dataSet.attributes
        nonMatchingDataSet.attributesByValues = dataSet.attributesByValues
        nonMatchingNodeStructure = nonMatchingDataSet.buildNodeStructure(None)
        nonMatchingDataSet.nodeStructure = nonMatchingNodeStructure
        
        return PartialDecisionNode(matchingDataSet.nodeStructure,nonMatchingDataSet.nodeStructure,bestAttributeValue)


# In[491]:


class EndNode:
    def __init__(self, dataSet):
        self.endNodeDecisions = dataSet.getMyLabelCounts()
       # print "endNode --> ", self.endNodeDecisions


# In[492]:


class PartialDecisionNode:
    def __init__(self,
                 nodeStructureMatching,
                 nodeStructureNotMatching,
                 aQuestionValue):
        self.nodeStructureMatching = nodeStructureMatching
        self.nodeStructureNotMatching = nodeStructureNotMatching
        self.aQuestionValue = aQuestionValue

    def __repr__(self):
        return "\nmnodeStructureNotMatching: %s \nnodeStructureNotMatching: %s \nQuestionValue: %s" % (
            self.nodeStructureMatching, 
            self.nodeStructureNotMatching,
            self.aQuestionValue,
        )        


# In[493]:


class NodeUtility:
    def printNodeDetails(self,endOrPartialDecisionNode, indent=""):
        
        if isinstance(endOrPartialDecisionNode, EndNode):
            print (indent + "End", endOrPartialDecisionNode.endNodeDecisions)
            return

        print (indent + str(endOrPartialDecisionNode.aQuestionValue))

        print (indent + '--> T:')
        self.printNodeDetails(endOrPartialDecisionNode.nodeStructureMatching, indent + "  ")

        print (indent + '--> F:')
        self.printNodeDetails(endOrPartialDecisionNode.nodeStructureNotMatching, indent + "  ")   

    def getPredictedLabelForDataLine(self,classItem, partialDecisionNode):

        if isinstance(partialDecisionNode, EndNode):
            return partialDecisionNode.endNodeDecisions

        if partialDecisionNode.aQuestionValue.match(classItem):
            return self.getPredictedLabelForDataLine(classItem, partialDecisionNode.nodeStructureMatching)
        else:
            return self.getPredictedLabelForDataLine(classItem, partialDecisionNode.nodeStructureNotMatching)        


# In[494]:


class LabelUtility:
    def getLabelProbabilities(self,labelCounts):
        total = sum(labelCounts.values()) * 1.0
        probabilitiesOfLabels = {}
        for aLabel in labelCounts.keys():
            probabilitiesOfLabels[aLabel] = float(labelCounts[aLabel] / total)
        return probabilitiesOfLabels  
    
    def getMajorityValue(self,labelCounts):
        stats = self.getLabelProbabilities(labelCounts)
        highestProb = 0
        highestProbLabel = ""
        for label in stats:
            thisProb = stats[label]
            if thisProb > highestProb: 
                highestProb = thisProb
                highestProbLabel = label
        return highestProbLabel        


# In[495]:


class AnalysisUtility:
    def __init__(self):
        self.nodeUtility = NodeUtility()
        self.labelUtility = LabelUtility()
        
    def createConffMatrix(self,expected, predicted, n_classes):
        m = [[0] * n_classes for i in range(n_classes)]
        for i in range(0,len(expected)):
            m[expected[i]-1][predicted[i]-1] += 1        
        return m
    #sum of diagonal divided by sum of all 
    def calcAccuracy(self,conf_matrix):
        t = sum(sum(l) for l in conf_matrix)
        return float(sum(conf_matrix[i][i] for i in range(len(conf_matrix)))) / t
        
    #Sum of diagonal divided by total labels
    def calcPrecision(self,conf_matrix,total):
        sum = 0
        for i in range(0,len(conf_matrix[0])):
            sum = sum + conf_matrix[i][i]

        print "Precision: ", sum , " / " , total
        return float(float(sum) / float(total))  

    #sum of diagonal divided by total false positives (erroneous labels)
    def calcRecall(self,conf_matrix,total):
        sum = 0
        for i in range(len(conf_matrix[0])):
            for j in range(len(conf_matrix[0])):
                if i!=j:
                    sum = sum + conf_matrix[i][j]

        print "Recall: ", sum , " / " , total
        return float(float(sum) / float(total))  


    def getActualExpected(self,dataSetTest, dataSetTrain):
        actual = []
        expected = []
        for dataLine in dataSetTest.dataLines:
            expected.append(int(dataLine.classLabel))
            actualVal = self.getPredictedValueForDataLine(dataLine, dataSetTrain.nodeStructure)
            actual.append(int(actualVal)) 
            
        return actual,expected
    
    def getPredictedValueForDataLine(self,dataLine, nodeStructure):
            labelCounts = self.nodeUtility.getPredictedLabelForDataLine(dataLine, nodeStructure)
            actualVal = self.labelUtility.getMajorityValue(labelCounts) #in case its not 100% certain, take the greatest

            return actualVal

    def getForestActualExpected(self,dataSetTest,randomForestDataSets):
        actualAll = []
        expected = []
        for dataLine in dataSetTest.dataLines:
            expected.append(int(dataLine.classLabel))
            actual = []
            for dataSetTrain in randomForestDataSets:
                 actualVal = self.getPredictedValueForDataLine(dataLine, dataSetTrain.nodeStructure)
                 actual.append(int(actualVal)) 
            actualAll.append(actual)

        actuals = self.getMajority(actualAll,dataSetTest)
        
        #print "length-a:",len(actuals)
        #print "length-e",len(expected)

        return actuals,expected
    
    def getMajority(self,actualAll,dataSetTest):
        actuals = []

        for i in range(0,len(dataSetTest.dataLines)):
            decisionsTotal = []
            actualResultsFromTrees = actualAll[i]
            for aDecision in actualResultsFromTrees:
                decisionsTotal.append(aDecision)
            majorityWinner = max(set(decisionsTotal), key = decisionsTotal.count)
            actuals.append(majorityWinner)
        return actuals
                                
    def printDebugExpectedProbs(self,dataSetTest,nodeStructure): 
        for dataLine in dataSetTest.dataLines:
            labelCounts = self.nodeUtility.getPredictedLabelForDataLine(dataLine, nodeStructure)
            stats = self.labelUtility.getLabelProbabilities(labelCounts)
            print ("Actual: %s Predicted: %s" %
                   (dataLine.classLabel, stats))  

    def printConfMatrixPretty(self,m):
        for row in m:
            rowString = ""
            for rowItem in row:
                rowString+=str(rowItem)+ " " 
            print rowString + "\n"

    def getConfMatrixPretty(self,m):
           output = []
           for row in m:
               rowString = ""
               for rowItem in row:
                   rowString+=str(rowItem)+ " " 
            
               output.append(rowString + "\n") 
           return output


# In[496]:


def getRunParameters (trainPath):
    if "nursery" in trainPath:
        return 7,25
    elif "synthetic.social" in trainPath:
        return 8,20
    elif "led" in trainPath:
        return 7,20
    elif "balance.scale" in trainPath:
        return 1,93

# In[497]
class DataSetBuilderThread(threading.Thread):
     def __init__(self,trainPath,dataLines,type,F):
         self.trainPath = trainPath
         self.dataLines = dataLines
         self.type = type
         self.F = F
         super(DataSetBuilderThread, self).__init__()

     def run(self):
         #start = time.time()
         self.dataSetTrain = DataSet(self.trainPath,self.dataLines,self.type,self.F,False,False)   
         #end = time.time()
         #print (end - start)

def runDecisionTree(type,trainPath,testPath,F,numTrees) :
    analysisUtility = AnalysisUtility()
        
    actual = []
    expected = []
    dataSetTrain = None
    allLabels = None
    if type is "RandomForest": #Do many trees picking F attributs at random (Forest-RI)
        #print "F: ",F
        #print "numTrees: ",numTrees
        dataSetThreads = []
        dataSetTrain = DataSet(trainPath,None,type,F,True,False)  
        allLabels = list(dataSetTrain.allLabels)
        if F is None or numTrees is None:
            F,numTrees = getRunParameters(trainPath)
        else:
            F = int(F)
            numTrees = int(numTrees)
        randomForestDataSets = set()
        for i in range(numTrees):
            dataSetTrainThread = DataSetBuilderThread(None,dataSetTrain.dataLines,type,F)    
            dataSetThreads.append(dataSetTrainThread)
            dataSetTrainThread.start()

        for dataSetBuildThreadRunning in dataSetThreads:
           dataSetBuildThreadRunning.join()

        for dataSetBuildThreadRunning in dataSetThreads:
            dataSetTrain = dataSetBuildThreadRunning.dataSetTrain
            randomForestDataSets.add(dataSetBuildThreadRunning.dataSetTrain)
        
        dataSetTest = DataSet(testPath,None,type,F,True,False)    

        actual,expected = analysisUtility.getForestActualExpected(dataSetTest,randomForestDataSets)         
            
    else:
        dataSetTrain = DataSet(trainPath,None,type,F,False,False)    
        allLabels = dataSetTrain.allLabels
        dataSetTest = DataSet(testPath,None,type,F,False,False)

      #  print dataSetTrain.nodeStructure

        #actual is predicted
        actual,expected = analysisUtility.getActualExpected(dataSetTest,dataSetTrain)


    # print dataSetTrain.allLabels
    m = analysisUtility.createConffMatrix(actual,expected,len(allLabels))
    analysisUtility.printConfMatrixPretty(m)
    #return analysisUtility.getConfMatrixPretty(m)
    # print "Accuracy:",analysisUtility.calcAccuracy(m)
    # print "Precision:",analysisUtility.calcPrecision(m,len(actual))
    # print "Recall:",analysisUtility.calcRecall(m,len(actual))
 




# In[498]:


#    runDecisionTree("RandomForest",2,10)
#    runDecisionTree("Normal",0,None)

