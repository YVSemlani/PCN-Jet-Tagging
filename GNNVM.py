import numpy as np

import pandas as pd

from operator import truth
import numpy as np
import awkward as ak

import torch

from tqdm import tqdm

import os

import dgl
import pickle

class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, loadFromDisk=False):
        
        self.jetNames = jetNames
        
        self.graphs = []
        self.sampleCountPerClass = []
        
        for jetType in tqdm(jetNames, total=len(jetNames)):
            if type(jetType) != list:
                
                if loadFromDisk:
                    saveFilePath = f'pickleFiles/{jetType}.pkl'
                else:
                    saveFilePath = f'../data/{jetType}.pkl'
                with open(saveFilePath, 'rb') as f:
                    singleJetGraphs = pickle.load(f)
                    self.graphs += singleJetGraphs
                
                self.sampleCountPerClass.append(len(singleJetGraphs))

                del singleJetGraphs
            else:
                totalCount = 0
                for item in jetType:
                    saveFilePath = saveFilePath = f'../data/{item}.pkl'
                    with open(saveFilePath, 'rb') as f:
                        singleJetGraphs = pickle.load(f)
                        self.graphs += singleJetGraphs
                    totalCount += len(singleJetGraphs)

                    del singleJetGraphs
                
                self.sampleCountPerClass.append(totalCount)


                
        self.labels = []
        label = 0
        for sampleCount in self.sampleCountPerClass:
            print(f'Class {label} has {sampleCount} samples')
            for _ in range(sampleCount):
                self.labels.append(label)
            label += 1

        print(f'Samples per class: {self.sampleCountPerClass}')
        

    def process(self):
        return
                
                
    def __getitem__(self, idx):
        
        return {'graph': self.graphs[idx], 'label': self.labels[idx]}

    def __len__(self):
        return len(self.graphs)

    # process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

testingSet = Top + Vector + QCD + Higgs
testingSet = [s + "-Testing" for s in testingSet]

jetNames = testingSet
print(jetNames)

dataset = GraphDataset(jetNames, 3, loadFromDisk=False)

dataset.process()

from dgllife.utils import RandomSplitter

maxEpochs = int(input("Max Epochs: "))

if maxEpochs != 0:

    train, val, test = RandomSplitter().train_val_test_split(dataset, frac_train=0.8, frac_test=0.1, 
                                                         frac_val=0.1, random_state=42)
else:
    train = dataset # use the full testing dataset if running testing evaluation


import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.batch import batch

class GNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(GNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        # Apply sigmoid activation function
        #prob = torch.sigmoid(logits)
        
        return logits #prob

class DGCNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(DGCNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.edgeconv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.edgeconv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        # Apply sigmoid activation function
        #prob = torch.sigmoid(logits)
        
        return logits #prob

batchSize = int(input("Batch Size: "))

#batches graphs in dataloader
def collateFunction(batch):
    graphs = [item['graph'] for item in batch]
    labels = [item['label'] for item in batch]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

if maxEpochs != 0:
# pass in train set here
    trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    validationLoader = DataLoader(val, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
    testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
else:
    testLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)
# Define the device to run the model on / Change if using with CUDA
device = input("cuda or cpu: ")

classificationLevel = input("Classification Level: ")
modelArchitecture = input("Model Architecture Name: ")


modelSaveFile =  "../modelSaveFiles/" + classificationLevel + modelArchitecture + ".pt"
load = input("Load from a save file (Y or N): ")

while type(load) != bool:
    if load == "Y":
        load = True
    elif load == "N":
        load = False
    else: 
        print("Invalid Input Please Enter Y or N: ")
        load = input("Load from a save file (Y or N): ")
convergence_threshold = float(input("Convergence Threshold: "))

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames) # Number of output classes

chebFilterSize = 16

modelType = input("Model Type:")
if modelType == "GCNN":
  model = GNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
elif modelType == "DGCNN":
  model = DGCNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
else:
  print("Invalid selection. Erroring out!")

if load:
    model.load_state_dict(torch.load(modelSaveFile))

for name, param in model.named_parameters():
    print(f'{name}: {param.dtype}')

model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

trainingLossTracker = []
trainingAccuracyTracker = []

validationLossTracker = []
validationAccuracyTracker = []

bestLoss = float('inf')
epochs_without_improvement = 0
epochsTillQuit = 10

# Train the model
for epoch in range(maxEpochs):
    runningLoss = 0
    totalCorrectPredictions = 0
    totalSamples = 0
    valTotalCorrectPredictions = 0
    valTotalSamples = 0
    model.train() # Set the model to training mode
    
    for batchIndex, (graph, labels) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False):
        # port data to the device in use
        graph = graph.to(device)
        labels = labels.to(device)
        labels = labels.long()

        # make prediction on the data
        logits = model(graph)
        
        # calculate loss and do backpropagation
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update running loss
        runningLoss += loss.item()

        # Compute accuracy by selecting the class with highest probability for prediction and comparing
        predictions = logits.argmax(dim=1)

        batchCorrectPredictions = (predictions == labels).sum().item()
        batchTotalSamples = labels.numel()
        totalCorrectPredictions += batchCorrectPredictions
        totalSamples += batchTotalSamples

        del graph, labels, logits, predictions

    # Compute epoch statistics
    epochLoss = runningLoss / len(trainLoader)
    trainingLossTracker.append(epochLoss)
    
    epochAccuracy = totalCorrectPredictions / totalSamples
    trainingAccuracyTracker.append(epochAccuracy)
    
    torch.save(model.state_dict(), modelSaveFile)
    print(f'Saved Model to file {modelSaveFile}')
    
    # Check for convergence on a validation set
    model.eval()
    validationLoss = 0.0

    with torch.no_grad():
        for graph, labels in tqdm(validationLoader, total=len(validationLoader), leave=False):
            graph, labels = graph.to(device), labels.to(device)

            logits = model(graph)
            loss = criterion(logits, labels)

            validationLoss += loss.item()
            
            predictions = logits.argmax(dim=1)
            batchCorrectPredictions = (predictions == labels).sum().item()
            batchTotalSamples = labels.numel()
            
            valTotalCorrectPredictions += batchCorrectPredictions
            valTotalSamples += batchTotalSamples

            del graph, labels, logits, predictions
    
    avgValidationLoss = validationLoss / len(validationLoader)
    validationLossTracker.append(avgValidationLoss)
    
    validationAccuracy = valTotalCorrectPredictions / valTotalSamples
    validationAccuracyTracker.append(validationAccuracy)
    

    # Check for convergence
    if avgValidationLoss < bestLoss - convergence_threshold:
        bestLoss = avgValidationLoss
        bestStateDict = model.state_dict()
        epochsWithoutImprovement = 0
    else:
        epochsWithoutImprovement += 1

    # Print training and validation losses
    print(f"Epoch {epoch + 1} - Training Loss={epochLoss} - Validation Loss={avgValidationLoss} - Training Accuracy={epochAccuracy} - Validation Accuracy={validationAccuracy}")

    # Check convergence criteria
    if epochsWithoutImprovement >= epochsTillQuit:
        print(f'Convergence achieved at epoch {epoch + 1}. Stopping training.')
        exitEpoch = epoch
        break
if maxEpochs != 0:
    torch.save(bestStateDict, modelSaveFile)

imageSavePath = f'{classificationLevel} {modelArchitecture}'
try:
    os.mkdir(imageSavePath)
except Exception as e:
    print(e)

if maxEpochs != 0:
    # plot training loss

    import matplotlib.pyplot as plt

    plt.plot(range(epoch + 1), trainingLossTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Training Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    #plt.show()
    plt.savefig(f'{imageSavePath}/Training Loss.png')

    # plot training accuracy

    plt.plot(range(epoch + 1), trainingAccuracyTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Training Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(f'{imageSavePath}/Training Accuracy.png')

    # plot validation loss

    plt.plot(range(epoch + 1), validationLossTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Validation Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(f'{imageSavePath}/Validation Loss.png')

    # plot validation accuracy

    plt.plot(range(epoch + 1), validationAccuracyTracker)
    plt.title(f'{classificationLevel} {modelArchitecture} Validation Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(f'{imageSavePath}/Validation Accuracy.png')

logitsTracker = []
predictionsTracker = []
targetsTracker = []

cfs = np.zeros((out_feats, out_feats))

model.eval()

import sklearn

if maxEpochs != 0:
    with torch.no_grad():
        for graph, labels in tqdm(testLoader, total=len(testLoader), leave=False):
            graph, labels = graph.to(device), labels.to(device)

            logits = model(graph)
            logitsTracker.append(logits.cpu())
            
            targetsTracker.append(labels.cpu())


            predictions = logits.argmax(dim=1)

            predictionsTracker.append(predictions.cpu())
            
            # update confusion matrix
            
            for idx, pred in enumerate(predictions):
                cfs[pred][labels[idx]] += 1
else:
    with torch.no_grad():
        for graph, labels in tqdm(testLoader, total=len(testLoader), leave=False):
            graph, labels = graph.to(device), labels.to(device)

            logits = model(graph)
            logitsTracker.extend(logits.cpu().tolist())
            
            targetsTracker.extend(labels.cpu().tolist())


            predictions = logits.argmax(dim=1)

            predictionsTracker.extend(predictions.cpu().tolist())
            
            # update confusion matrix
            
            for idx, pred in enumerate(predictions):
                cfs[pred][labels[idx]] += 1


logitsTrackerFile = f'../metrics/{classificationLevel}-{modelArchitecture}-Logits.pkl'
targetsTrackerFile =  f'../metrics/{classificationLevel}-{modelArchitecture}-Targets.pkl'
predictionsTrackerFile =  f'../metrics/{classificationLevel}-{modelArchitecture}-Predictions.pkl'

with open(logitsTrackerFile, 'wb') as f:
    pickle.dump(logitsTracker, f)

with open(targetsTrackerFile, 'wb') as f:
    pickle.dump(targetsTracker, f)

with open(predictionsTrackerFile, 'wb') as f:
    pickle.dump(predictionsTracker, f)

import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')

ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print(cfs/np.sum(cfs))

## Display the visualization of the Confusion Matrix.
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')

from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

logitsTracker = np.stack(logitsTracker)
rocLogits = logitsTracker.reshape((logitsTracker.shape[0] * logitsTracker.shape[1], logitsTracker.shape[2]))

targetsTracker = np.stack(targetsTracker)
rocTargets = targetsTracker.flatten()


skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(8, 6), title=f'{classificationLevel} {modelArchitecture} ROC-AUC Curve')

plt.savefig(f'{imageSavePath}/ROC-AUC.png')


def calculateConfusionMetrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    metrics = []

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        true_negative = np.sum(confusion_matrix) - true_positive - false_positive - false_negative

        accuracy = (true_positive + true_negative) / np.sum(confusion_matrix)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)

        metrics.append([accuracy, precision, recall, specificity])

    return metrics

metrics = calculateConfusionMetrics(cfs)

classLabels = jetNames
metricsDF = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'Specificity'], index=classLabels)

# Calculate micro and macro averages
microAvg = metricsDF.mean(axis=0)
macroAvg = metricsDF.mean(axis=0)

# Add micro and macro averages to the DataFrame
metricsDF.loc['Micro Avg'] = microAvg
metricsDF.loc['Macro Avg'] = macroAvg

# Print the metrics table
print(metricsDF)



