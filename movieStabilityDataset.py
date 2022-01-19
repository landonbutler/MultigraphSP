# 2019/04/10~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Test the movie recommendation dataset on several architectures.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       parameters of each realization on a directory named 'savedModels'.
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       in pickle format. These variables are saved in a trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in pickle format. 
#       These plots (and variables) are in a figs directory.

colorPenn = ['#01256E', # blue
             '#95001A', # red
             '#C35A00', # orange
             #'#F2C100', # yellow
             '#008E00', # green
             '#4A0042' # purple
             ]

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.training as training
import Modules.evaluation as evaluation
import Modules.loss as loss

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

graphType = 'movie' # Graph type: 'user'-based or 'movie'-based
labelID = [50] # Which node to focus on (either a list or the str 'all')
# When 'movie': [1]: Toy Story, [50]: Star Wars, [258]: Contact,
# [100]: Fargo, [181]: Return of the Jedi, [294]: Liar, liar
if labelID == 'all':
    labelIDstr = 'all'
elif len(labelID) == 1:
    labelIDstr = '%03d' % labelID[0]
else:
    labelIDstr = ['%03d_' % i for i in labelID]
    labelIDstr = "".join(labelIDstr)
    labelIDstr = labelIDstr[0:-1]
thisFilename = 'movieStabilityDataset' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run
dataDir = os.path.join('datasets','movielens')

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + labelIDstr + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

useGPU = True # If true, and GPU is available, use it.

rTestMin = 0.1 # Minimum value of perturbation
rTestMax = 0.9 # Maximum value of perturbation
nSimPoints = 10 # Number of simulation points
ratioTest = np.linspace(rTestMin, rTestMax, num=nSimPoints)

ratioTrain = 0.9 # Ratio of training samples
ratioValid = 0.1 # Ratio of validation samples (out of the total training
# samples)
# Final split is:
#   nValidation = round(ratioValid * ratioTrain * nTotal)
#   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
#   nTest = nTotal - nTrain - nValidation
maxNodes = None # Maximum number of nodes (select the ones with the largest
#   number of ratings)
minRatings = 0 # Discard samples (rows and columns) with less than minRatings 
    # ratings
interpolateRatings = False # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

nDataSplits = 5 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to obtain some result that is more robust to this split.
nPerturb = 100 # The perturbations are random, so we need to run for a couple of
    # them
    
# Given that we build the graph from a training split selected at random, it
# could happen that it is disconnected, or directed, or what not. In other 
# words, we might want to force (by removing nodes) some useful characteristics
# on the graph
keepIsolatedNodes = True # If True keeps isolated nodes
forceUndirected = True # If True forces the graph to be undirected
forceConnected = False # If True returns the largest connected component of the
    # graph as the main graph
kNN = 10 # Number of nearest neighbors

maxDataPoints = None # None to consider all data points

#\\\ Save values:
writeVarValues(varsFile,
               {'labelID': labelID,
                'graphType': graphType,
                'rTestMin': rTestMin,
                'rTestMax': rTestMax,
                'nSimPoints': nSimPoints,
                'ratioTest': ratioTest,
                'ratioTrain': ratioTrain,
                'ratioValid': ratioValid,
                'maxNodes': maxNodes,
                'minRatings': minRatings,
                'interpolateRatings': interpolateRatings,
                'nDataSplits': nDataSplits,
                'keepIsolatedNodes': keepIsolatedNodes,
                'forceUndirected': forceUndirected,
                'forceConnected': forceConnected,
                'kNN': kNN,
                'maxDataPoints': maxDataPoints,
                'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.SmoothL1Loss

#\\\ Overall training options
nEpochs = 40 # Number of epochs
batchSize = 5 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Just four architecture one- and two-layered Selection and Local GNN. The main
# difference is that the Local GNN is entirely local (i.e. the output is given
# by a linear combination of the features at a single node, instead of a final
# MLP layer combining the features at all nodes).
    
# Select desired architectures
doLinearFlt = True
doNoPenalty = True
doILpenalty = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the model dictionary is called 'model' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the model dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []
modelLegend = {}

#\\\\\\\\\\\\\\\\\\\\\
#\\\ LINEAR FILTER \\\
#\\\\\\\\\\\\\\\\\\\\\

if doLinearFlt:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelLnrFlt = {} # Model parameters for the Local GNN (LclGNN)
    modelLnrFlt['name'] = 'LnrFlt'
    modelLnrFlt['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelLnrFlt['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelLnrFlt['dimNodeSignals'] = [1, 64] # Features per layer
    modelLnrFlt['nFilterTaps'] = [5] # Number of filter taps per layer
    modelLnrFlt['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelLnrFlt['nonlinearity'] = gml.NoActivation # Selected nonlinearity
    # Pooling
    modelLnrFlt['poolingFunction'] = gml.NoPool # Summarizing function
    modelLnrFlt['nSelectedNodes'] = None # To be determined later on
    modelLnrFlt['poolingSize'] = [1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelLnrFlt['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelLnrFlt['GSO'] = None # To be determined later on, based on data
    modelLnrFlt['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelLnrFlt['trainer'] = training.TrainerSingleNode
    
    #\\\ EVALUATOR
    
    modelLnrFlt['evaluator'] = evaluation.evaluateSingleNode
    
    #\\\ LOSS FUNCTION
    
    modelLnrFlt['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)

    #\\\ Save Values:
    writeVarValues(varsFile, modelLnrFlt)
    modelList += [modelLnrFlt['name']]
    modelLegend[modelLnrFlt['name']] = 'Linear Filter'

#\\\\\\\\\\\\\\\\\\
#\\\ NO PENALTY \\\
#\\\\\\\\\\\\\\\\\\

if doNoPenalty:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelNoPnlt = {} # Model parameters for the Local GNN (LclGNN)
    modelNoPnlt['name'] = 'NoPnlt'
    modelNoPnlt['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelNoPnlt['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelNoPnlt['dimNodeSignals'] = [1, 64] # Features per layer
    modelNoPnlt['nFilterTaps'] = [5] # Number of filter taps per layer
    modelNoPnlt['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelNoPnlt['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelNoPnlt['poolingFunction'] = gml.NoPool # Summarizing function
    modelNoPnlt['nSelectedNodes'] = None # To be determined later on
    modelNoPnlt['poolingSize'] = [1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelNoPnlt['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelNoPnlt['GSO'] = None # To be determined later on, based on data
    modelNoPnlt['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelNoPnlt['trainer'] = training.TrainerSingleNode
    
    #\\\ EVALUATOR
    
    modelNoPnlt['evaluator'] = evaluation.evaluateSingleNode
    
    #\\\ LOSS FUNCTION
    
    modelNoPnlt['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)

    #\\\ Save Values:
    writeVarValues(varsFile, modelNoPnlt)
    modelList += [modelNoPnlt['name']]
    modelLegend[modelNoPnlt['name']] = 'GNN'

#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ WITH IL PENALTY \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doILpenalty:    
    
    #\\\/// Model 1: 1 penalty \\\///

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelILpnl1 = {} # Model parameters for the Local GNN (LclGNN)
    modelILpnl1['name'] = 'ILpnl1'
    modelILpnl1['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelILpnl1['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelILpnl1['dimNodeSignals'] = [1, 64] # Features per layer
    modelILpnl1['nFilterTaps'] = [5] # Number of filter taps per layer
    modelILpnl1['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelILpnl1['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelILpnl1['poolingFunction'] = gml.NoPool # Summarizing function
    modelILpnl1['nSelectedNodes'] = None # To be determined later on
    modelILpnl1['poolingSize'] = [1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelILpnl1['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelILpnl1['GSO'] = None # To be determined later on, based on data
    modelILpnl1['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelILpnl1['trainer'] = training.TrainerSingleNode
    
    #\\\ EVALUATOR
    
    modelILpnl1['evaluator'] = evaluation.evaluateSingleNode
    
    #\\\ LOSS FUNCTION
    
    modelILpnl1['penalty'] = ('ILconstant', 1.) # Penalty function name, and
        # penalty multiplier
    modelILpnl1['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)

    #\\\ Save Values:
    writeVarValues(varsFile, modelILpnl1)
    modelList += [modelILpnl1['name']]
    modelLegend[modelILpnl1['name']] = 'GNN.1 (IL)'
    
    #\\\/// Model 2: 0.5 penalty \\\///
    
    modelILpnl2 = deepcopy(modelILpnl1)
    modelILpnl2['name'] = 'ILpnl2'
    modelILpnl2['penalty'] = ('ILconstant', 100.) # Penalty function name, and
        # penalty multiplier
    modelILpnl2['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)
    writeVarValues(varsFile, modelILpnl2)
    modelList += [modelILpnl2['name']]
    modelLegend[modelILpnl2['name']] = 'GNN.2 (IL)'
    
    #\\\/// Model 2: 2 penalty \\\///
    
    modelILpnl3 = deepcopy(modelILpnl1)
    modelILpnl3['name'] = 'ILpnl3'
    modelILpnl3['penalty'] = ('ILconstant', 1000.) # Penalty function name, and
        # penalty multiplier
    modelILpnl3['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)
    writeVarValues(varsFile, modelILpnl3)
    modelList += [modelILpnl3['name']]
    modelLegend[modelILpnl3['name']] = 'GNN.3 (IL)'

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 20 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers
fontSize = 12 # font size

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        modelDict = eval('model' + thisModel)
        print("\t%s: %s" % (thisModel, modelDict['device']))

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
costBest = {} # Cost for the best model
costLast = {} # Cost for the last model
costBestHat = {} # Cost for the best model under relative perturbation
costLastHat = {} # Cost for the last model under relative perturbation
ILconstantBest = {} # Save the value of the IL constant for best model
ILconstantLast = {} # Save the value of the IL constant for last model
LpConstantBest = {} # Save the value of the Lipschitz constant for best model
LpConstantLast = {} # Save the value of the Lipschitz constant for last model
GNNdiffBest = {} # Value of the difference in the GNN output for best model
GNNdiffLast = {} # Value of the difference in the GNN output for last model
eRelN = {} # Actual value of epsilon (after normalization) for the relative model
eRelU = {} # Actual value of epsilon (after normalization) for the relative model
eAbsN = {} # Actual value of epsilon (after normalization) for the absolute model
eAbsU = {} # Actual value of epsilon (after normalization) for the absolute model
deltaN = {} # Value of delta * sqrt(N) for each model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nSimPoints
    costLast[thisModel] = [None] * nSimPoints
    costBestHat[thisModel] = [None] * nSimPoints
    costLastHat[thisModel] = [None] * nSimPoints
    ILconstantBest[thisModel] = [None] * nSimPoints
    ILconstantLast[thisModel] = [None] * nSimPoints
    LpConstantBest[thisModel] = [None] * nSimPoints
    LpConstantLast[thisModel] = [None] * nSimPoints
    GNNdiffBest[thisModel] = [None] * nSimPoints
    GNNdiffLast[thisModel] = [None] * nSimPoints
    eRelN[thisModel] = [None] * nSimPoints
    eRelU[thisModel] = [None] * nSimPoints
    eAbsN[thisModel] = [None] * nSimPoints
    eAbsU[thisModel] = [None] * nSimPoints
    deltaN[thisModel] = [None] * nSimPoints
    for n in range(nSimPoints):
        costBest[thisModel][n] = [None] * nDataSplits
        costLast[thisModel][n] = [None] * nDataSplits
        costBestHat[thisModel][n] = [None] * nDataSplits
        costLastHat[thisModel][n] = [None] * nDataSplits
        ILconstantBest[thisModel][n] = [None] * nDataSplits
        ILconstantLast[thisModel][n] = [None] * nDataSplits
        LpConstantBest[thisModel][n] = [None] * nDataSplits
        LpConstantLast[thisModel][n] = [None] * nDataSplits
        GNNdiffBest[thisModel][n] = [None] * nDataSplits
        GNNdiffLast[thisModel][n] = [None] * nDataSplits
        eRelN[thisModel][n] = [None] * nDataSplits
        eRelU[thisModel][n] = [None] * nDataSplits
        eAbsN[thisModel][n] = [None] * nDataSplits
        eAbsU[thisModel][n] = [None] * nDataSplits
        deltaN[thisModel][n] = [None] * nDataSplits
        for split in range(nDataSplits):
            costBestHat[thisModel][n][split] = [None] * nPerturb
            costLastHat[thisModel][n][split] = [None] * nPerturb
            GNNdiffBest[thisModel][n][split] = [None] * nPerturb
            GNNdiffLast[thisModel][n][split] = [None] * nPerturb
            eRelN[thisModel][n][split] = [None] * nPerturb
            eRelU[thisModel][n][split] = [None] * nPerturb
            eAbsN[thisModel][n][split] = [None] * nPerturb
            eAbsU[thisModel][n][split] = [None] * nPerturb
            deltaN[thisModel][n][split] = [None] * nPerturb
            

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options, then we create a 
# separate dictionary per model.

trainingOptsPerModel= {}

#%%##################################################################
#                                                                   #
#                    PERTURBATION REALIZATION                       #
#                                                                   #
#####################################################################

# For each value of epsilon that we are going to perturb the matrix.
# Recall that this value of perturbation only shows at evaluation (testing)

for n in range(nSimPoints):
    
    if doPrint:
        print("\n[Sim %d] Ratio Test = %.4f" % (n, ratioTest[n]))

    #%%##################################################################
    #                                                                   #
    #                    DATA SPLIT REALIZATION                         #
    #                                                                   #
    #####################################################################
    
    # Start generating a new data split for each of the number of data splits that
    # we previously specified
    
    for split in range(nDataSplits):
    
        #%%##################################################################
        #                                                                   #
        #                    DATA HANDLING                                  #
        #                                                                   #
        #####################################################################
    
        ############
        # DATASETS #
        ############
        
        if doPrint:
            print("\n[Sim %d] Loading data" % n, end = '')
            if nDataSplits > 1:
                print(" for split %d" % split, end = '')
            print("...", end = ' ', flush = True)
    
        #   Load the data, which will give a specific split
        data = Utils.dataTools.MovieLens(graphType, # 'user' or 'movies'
                                         labelID, # ID of node to interpolate
                                         ratioTrain, # ratio of training samples
                                         ratioValid, # ratio of validation samples
                                         dataDir, # directory where dataset is
                                         # Extra options
                                         keepIsolatedNodes,
                                         forceUndirected,
                                         forceConnected,
                                         kNN, # Number of nearest neighbors
                                         maxNodes = maxNodes,
                                         maxDataPoints = maxDataPoints,
                                         minRatings = minRatings,
                                         interpolate = interpolateRatings)
        
        if doPrint:
            print("OK")
    
        #########
        # GRAPH #
        #########
        
        if doPrint:
            print("[Sim %d] Setting up the graph..." % n, end = ' ', flush = True)
    
        # Create graph
        adjacencyMatrix = data.getGraph()
        G = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                             {'adjacencyMatrix': adjacencyMatrix})
        G.computeGFT() # Compute the GFT of the stored GSO
    
        # And re-update the number of nodes for changes in the graph (due to
        # enforced connectedness, for instance)
        nNodes = G.N
    
        # Once data is completely formatted and in appropriate fashion, change its
        # type to torch
        data.astype(torch.float64)
        # And the corresponding feature dimension that we will need to use
        data.expandDims()
        
        if doPrint:
            print("OK")
    
        #%%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################
    
        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}
    
        # If a new model is to be created, it should be called for here.
        
        if doPrint:
            print("[Sim %d] Model initialization..." % n, flush = True)
            
        for thisModel in modelList:
            
            # Get the corresponding parameter dictionary
            modelDict = deepcopy(eval('model' + thisModel))
            # and training options
            trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
            
            # Now, this dictionary has all the hyperparameters that we need to pass
            # to the architecture, but it also has the 'name' and 'archit' that
            # we do not need to pass them. So we are going to get them out of
            # the dictionary
            thisName = modelDict.pop('name')
            callArchit = modelDict.pop('archit')
            thisDevice = modelDict.pop('device')
            thisTrainer = modelDict.pop('trainer')
            thisEvaluator = modelDict.pop('evaluator')
            thisLossFunction = modelDict.pop('lossFunction')
            if 'penalty' in modelDict.keys():
                thisPenalty = modelDict.pop('penalty')
            else:
                thisPenalty = None
            
            # If more than one graph or data realization is going to be carried out,
            # we are going to store all of thos models separately, so that any of
            # them can be brought back and studied in detail.
            if nDataSplits > 1:
                thisName += 'G%02d' % split
                
            if doPrint:
                print("\tInitializing %s..." % thisName,
                      end = ' ',flush = True)
                
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisOptimAlg = optimAlg
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S = G.S.copy()/np.max(np.real(G.E))
            # Do not forget to add the GSO to the input parameters of the archit
            modelDict['GSO'] = S
            # Add the number of nodes for the no-pooling part
            modelDict['nSelectedNodes'] = [nNodes]
            
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = callArchit(**modelDict)
            
            #############
            # OPTIMIZER #
            #############
    
            if thisOptimAlg == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate,
                                       betas = (beta1, beta2))
            elif thisOptimAlg == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(),
                                      lr = learningRate)
            elif thisOptimAlg == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            #########
            # MODEL #
            #########
    
            modelCreated = model.Model(thisArchit,
                                       thisLossFunction,
                                       thisOptim,
                                       thisTrainer,
                                       thisEvaluator,
                                       thisDevice,
                                       thisName,
                                       saveDir,
                                       penalty = thisPenalty)
    
            modelsGNN[thisName] = modelCreated
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisOptimizationAlgorithm': thisOptimAlg,
                            'thisLossFunction': thisLossFunction,
                            'thisTrainer': thisTrainer,
                            'thisEvaluator': thisEvaluator,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
                
        if doPrint:
            print("Model initialization... COMPLETE")
    
        #%%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################
        
        for thisModel in modelsGNN.keys():
            
            if doPrint:
                print("\n[Sim %d] Training model %s..." % (n, thisModel))
                
            for m in modelList:
                if m in thisModel:
                    modelName = m
        
            if nDataSplits > 1:
                trainingOptsPerModel[modelName]['graphNo'] = split
                
            thisTrainVars = modelsGNN[thisModel].train(data,
                                                       nEpochs,
                                                       batchSize,
                                                       **trainingOptsPerModel[modelName])
            
            if doPrint:
                print("[Sim %d] Training model %s... COMPLETE" % (n, thisModel))
    
        #%%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################
    
        # Now that the model has been trained, we evaluate them on the test
        # samples.
    
        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.
    
        if doPrint:
            print("\n[Sim %d] Total testing RMSE" % n, end = '', flush = True)
            if nDataSplits > 1:
                print(" (Split %02d)" % split, end = '', flush = True)
            print(":", flush = True)
            
    
        for thisModel in modelsGNN.keys():
            
            for m in modelList:
                if m in thisModel:
                    modelName = m
    
            thisEvalVars = modelsGNN[thisModel].evaluate(data)
            
            thisCostBest = thisEvalVars['costBest']
            thisCostLast = thisEvalVars['costLast']
            # Save the actual output vector (shape: B x F[-1])
            ySbest = thisEvalVars['yGNNbest']
            ySlast = thisEvalVars['yGNNlast']
            
            with torch.no_grad():
                # While the model has already been trained and everything relevant
                # computed, we still enfore toch.no_grad() just in case
                #   Be sure we're working on the best model
                modelsGNN[thisModel].load(label = 'Best')
                thisCbest = modelsGNN[thisModel].archit.ILconstant().item()
                thisLbest = modelsGNN[thisModel].archit.LipschitzConstant().item()
                #   Reload the last model
                modelsGNN[thisModel].load(label = 'Last')
                thisClast = modelsGNN[thisModel].archit.ILconstant().item()
                thisLlast = modelsGNN[thisModel].archit.LipschitzConstant().item()
            
            # Save values
            writeVarValues(varsFile,
                           {'costBest%s' % thisModel: thisCostBest,
                            'costLast%s' % thisModel: thisCostLast,
                            'ILconstantBest%s' % thisModel: thisCbest,
                            'ILconstantLast%s' % thisModel: thisClast})
    
            # Now check which is the model being trained
            costBest[modelName][n][split] = thisCostBest
            costLast[modelName][n][split] = thisCostLast
            ILconstantBest[modelName][n][split] = thisCbest
            ILconstantLast[modelName][n][split] = thisClast
            LpConstantBest[modelName][n][split] = thisLbest
            LpConstantLast[modelName][n][split] = thisLlast
            # This is so that we can later compute a total accuracy with
            # the corresponding error.
            
            if doPrint:
                print("\t%s: %.4f (C=%.4f) [Best] %.4f (C=%.4f) [Last]" % 
                      (thisModel,
                       thisCostBest, thisCbest,
                       thisCostLast, thisClast), end = ' ', flush = True)
                
            ################
            # PLOT FILTERS #
            ################
            
    
            modelsGNN[thisModel].load(label='Best') # Load best model, which is
                # usually the one that works better
            thisFilters, eigenvalues = modelsGNN[thisModel].archit.getFilters()
            saveDirFilters = os.path.join(saveDir,'filters')
            if nDataSplits > 1:
                saveDirFilters = os.path.join(saveDirFilters,'split%02d' % split)
            if not os.path.exists(saveDirFilters):
                os.makedirs(saveDirFilters)
            # thisFilters is a dictionary where each key corresponds to a layer
            # and the output is a Fl x E x Gl x nSamples np.array containing all
            # the filters of that layer
            # eigenvalues is the x-axis of shape E x nSamples
            for l in thisFilters.keys():
                for e in range(eigenvalues.shape[0]):
                    filterFig = plt.figure(figsize = (1.61*figSize, 1.61*figSize))
                    for f in range(thisFilters[l].shape[0]):
                        for g in range(thisFilters[l].shape[2]):
                            plt.plot(eigenvalues[e], thisFilters[l][f,e,g,:])
                    plt.ylabel(r'$h^{fg}(\lambda)$ ($E=%d$, $L=%d$)' % (e+1,l+1))
                    plt.xlabel(r'$\lambda$')
                    filterFig.savefig(os.path.join(
                        saveDirFilters,'filters%sLayer%02d.pdf' % (thisModel, l+1)),
                                      bbox_inches='tight')
                    plt.close(fig = filterFig)
                    
                    
            ################
            # PERTURBATION #
            ################
            
            if doPrint:
                print("Perturbing", end = '', flush = True)
            
            for p in range(nPerturb):
                
                # Get the used GSO
                #   Normalized GSO
                thisSn = modelsGNN[thisModel].archit.S.data.cpu().numpy() #ExNxN
                #   Unnormalized GSO
                thisSu = G.S.copy().reshape((1, G.N, G.N))
                
                # Get the new data partition
                dataTest = Utils.dataTools.MovieLens(graphType,
                                                     labelID,
                                                     1.-ratioTest[n],
                                                     ratioValid,
                                                     dataDir,
                                                     keepIsolatedNodes,
                                                     forceUndirected,
                                                     forceConnected,
                                                     kNN,
                                                     maxNodes = maxNodes,
                                                     maxDataPoints = \
                                                                  maxDataPoints,
                                                     minRatings = \
                                                                     minRatings,
                                                     interpolate = \
                                                             interpolateRatings)
                    
                # Create graph
                adjacencyMatrixTest = dataTest.getGraph()
                Gtest=graphTools.Graph('adjacency',
                                       adjacencyMatrixTest.shape[0],
                                       {'adjacencyMatrix': adjacencyMatrixTest})
                Gtest.computeGFT()
                
                # Adequate data
                dataTest.astype(torch.float64) # Convert it to torch after
                    # getting the adjacency matrix, just in case
                dataTest.expandDims()
                
                assert Gtest.N == G.N # So that we can compare them
                
                # Compute the value of epsilon 
                #   Upper bound on the value of epsilon, since we are not
                #   accounting for permutations, although these shouldn't make
                #   a big difference problem because the nodes are the same and
                #   thus the permutation won't often be different from the
                #   identity
                # Get the new GSOs
                ShatU = Gtest.S.copy().reshape((1, Gtest.N, Gtest.N))
                ShatN = ShatU / np.max(np.real(Gtest.E))
                # Compute the epsilon
                for e in range(thisSn.shape[0]):
                    eRelN[modelName][n][split][p] = \
                            np.linalg.norm(thisSn[e] - ShatN[e], ord = 2)/\
                                           np.linalg.norm(thisSn[e], ord = 2)
                    eRelU[modelName][n][split][p] = \
                            np.linalg.norm(thisSu[e] - ShatU[e], ord = 2)/\
                                           np.linalg.norm(thisSu[e], ord = 2)
                    eAbsN[modelName][n][split][p] = \
                            np.linalg.norm(thisSn[e] - ShatN[e], ord = 2)
                    eAbsU[modelName][n][split][p] = \
                            np.linalg.norm(thisSu[e] - ShatU[e], ord = 2)
                    _, V = graphTools.computeGFT(thisSu[e])
                    _, U = graphTools.computeGFT(thisSu[e] - ShatU[e])
                    thisDelta = (np.linalg.norm(U - V, ord = 2) + 1) ** 2 - 1
                    deltaN[modelName][n][split][p] = thisDelta * np.sqrt(nNodes)
                
                #\\\ Compute the output with the new matrix
                # Change the GSO
                modelsGNN[thisModel].archit.changeGSO(ShatN)
                #   And move it to device
                modelsGNN[thisModel].archit.to(modelsGNN[thisModel].device)
                
                # Compute the output
                thisEvalVarsHat = modelsGNN[thisModel].evaluate(data)
                #   Get the output values
                thisCostBestHat = thisEvalVarsHat['costBest']
                thisCostLastHat = thisEvalVarsHat['costLast']
                ySbestHat = thisEvalVarsHat['yGNNbest']
                ySlastHat = thisEvalVarsHat['yGNNlast']
                
                # Compare
                #   The output is just a single scalar, so the comparison is 
                #   straightforward absolute value
                #   Now we have B x F[-1] vectors, so we need to compute the 
                #   norm across the F[-1] dimension
                thisGNNdiffBest = np.sum((ySbest - ySbestHat) ** 2,
                                            axis = 1)
                thisGNNdiffBest = np.mean(np.sqrt(thisGNNdiffBest))
                thisGNNdiffLast = np.sum((ySlast - ySlastHat) ** 2,
                                            axis = 1)
                thisGNNdiffLast = np.mean(np.sqrt(thisGNNdiffLast))
                
                # Save
                costBestHat[modelName][n][split][p] = thisCostBestHat
                costLastHat[modelName][n][split][p] = thisCostLastHat
                GNNdiffBest[modelName][n][split][p] = thisGNNdiffBest
                GNNdiffLast[modelName][n][split][p] = thisGNNdiffLast
                
                if doPrint and np.mod(p+1, nPerturb//3) == 0:
                    print(".", end = '', flush = True)
            
            if doPrint:
                print(" OK", flush = True)

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanCostBest = {} # Mean across data splits
meanCostLast = {} # Mean across data splits
stdDevCostBest = {} # Standard deviation across data splits
stdDevCostLast = {} # Standard deviation across data splits

meanILconstantBest = {}
meanILconstantLast = {}
stdDevILconstantBest = {}
stdDevILconstantLast = {}

if doPrint:
    print("\nFinal evaluations (%02d data splits)" % (nDataSplits))

for thisModel in modelList:
    # Convert the lists into a nDataSplits vector
    costBest[thisModel] = np.array(costBest[thisModel])
    costLast[thisModel] = np.array(costLast[thisModel])
    ILconstantBest[thisModel] = np.array(ILconstantBest[thisModel])
    ILconstantLast[thisModel] = np.array(ILconstantLast[thisModel])

    # And now compute the statistics (across graphs)
    meanCostBest[thisModel] = np.mean(costBest[thisModel])
    meanCostLast[thisModel] = np.mean(costLast[thisModel])
    stdDevCostBest[thisModel] = np.std(costBest[thisModel])
    stdDevCostLast[thisModel] = np.std(costLast[thisModel])
    meanILconstantBest[thisModel] = np.mean(ILconstantBest[thisModel])
    meanILconstantLast[thisModel] = np.mean(ILconstantLast[thisModel])
    stdDevILconstantBest[thisModel] = np.std(ILconstantBest[thisModel])
    stdDevILconstantLast[thisModel] = np.std(ILconstantLast[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.4f (+-%6.4f) C=%.4f(+-%.4f) [Best] %6.4f (+-%6.4f) C=%.4f(+-%.4f) [Last]" % (
                thisModel,
                meanCostBest[thisModel],
                stdDevCostBest[thisModel],
                meanILconstantBest[thisModel],
                stdDevILconstantBest[thisModel],
                meanCostLast[thisModel],
                stdDevCostLast[thisModel],
                meanILconstantLast[thisModel],
                stdDevILconstantLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanCostBest%s' % thisModel: meanCostBest[thisModel],
                'stdDevCostBest%s' % thisModel: stdDevCostBest[thisModel],
                'meanCostLast%s' % thisModel: meanCostLast[thisModel],
                'stdDevCostLast%s' % thisModel: stdDevCostLast[thisModel],
                'meanILconstantBest%s' % thisModel: meanILconstantBest[thisModel],
                'stdDevILconstantBest%s' % thisModel: stdDevILconstantBest[thisModel],
                'meanILconstantLast%s' % thisModel: meanILconstantLast[thisModel],
                'stdDevILconstantLast%s' % thisModel: stdDevILconstantLast[thisModel]})
    
# Save the printed info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Final evaluations (%02d data splits)\n" % (nDataSplits))
    for thisModel in modelList:
        file.write("\t%s: %6.4f (+-%6.4f) C=%.4f(+-%.4f) [Best] %6.4f (+-%6.4f) C=%.4f(+-%.4f) [Last]\n" % (
                   thisModel,
                   meanCostBest[thisModel],
                   stdDevCostBest[thisModel],
                   meanILconstantBest[thisModel],
                   stdDevILconstantBest[thisModel],
                   meanCostLast[thisModel],
                   stdDevCostLast[thisModel],
                   meanILconstantLast[thisModel],
                   stdDevILconstantLast[thisModel]))
    file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs:
    
    # Set the epsilon values
    epsRelN = np.zeros((nSimPoints, len(modelList)))
    epsRelU = np.zeros((nSimPoints, len(modelList)))
    epsAbsN = np.zeros((nSimPoints, len(modelList)))
    epsAbsU = np.zeros((nSimPoints, len(modelList)))
    l = 0
    for thisModel in modelList:
        eRelN[thisModel] = np.array(eRelN[thisModel]) # nSimPoints x nDataSplits x nPerturb
        eRelU[thisModel] = np.array(eRelU[thisModel]) # nSimPoints x nDataSplits x nPerturb
        eAbsN[thisModel] = np.array(eAbsN[thisModel]) # nSimPoints x nDataSplits x nPerturb
        eAbsU[thisModel] = np.array(eAbsU[thisModel]) # nSimPoints x nDataSplits x nPerturb
        
        thisEpsRelN = np.mean(eRelN[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsRelN[:,l] = np.mean(thisEpsRelN, axis = 1) # nSimPoints
        thisEpsRelU = np.mean(eRelU[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsRelU[:,l] = np.mean(thisEpsRelU, axis = 1) # nSimPoints
        thisEpsAbsN = np.mean(eAbsN[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsAbsN[:,l] = np.mean(thisEpsAbsN, axis = 1) # nSimPoints
        thisEpsAbsU = np.mean(eAbsU[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsAbsU[:,l] = np.mean(thisEpsAbsU, axis = 1) # nSimPoints
        
        l += 1
    epsRelN = np.mean(epsRelN, axis = 1) # nSimPoints
    epsRelU = np.mean(epsRelU, axis = 1) # nSimPoints
    epsAbsN = np.mean(epsAbsN, axis = 1) # nSimPoints
    epsAbsU = np.mean(epsAbsU, axis = 1) # nSimPoints
    
    if doPrint:
        print(" ")
        print("epsilon relative normalized   = %s" % [round(d,4) for d in epsRelN.tolist()])
        print("epsilon relative unnormalized = %s" % [round(d,4) for d in epsRelU.tolist()])
        print("epsilon absolute normalized   = %s" % [round(d,4) for d in epsAbsN.tolist()])
        print("epsilon absolute unnormalized = %s" % [round(d,4) for d in epsAbsU.tolist()])
    
    # Save the printed info into the .txt file as well
    with open(varsFile, 'a+') as file:
        file.write("\nepsilon relative normalized   = %s" % [round(d,4) for d in epsRelN.tolist()])
        file.write("\nepsilon relative unnormalized = %s" % [round(d,4) for d in epsRelU.tolist()])
        file.write("\nepsilon absolute normalized   = %s" % [round(d,4) for d in epsAbsN.tolist()])
        file.write("\nepsilon absolute unnormalized = %s" % [round(d,4) for d in epsAbsU.tolist()])
        file.write('\n')

    ###################
    # DATA PROCESSING #
    ###################
    
    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)
    
    #\\\ SAVE SPACE
    #   Cost difference and bound
    costDiffBest = {}
    costDiffLast = {}
    boundBestIL = {}
    boundLastIL = {}
    boundBestLp = {}
    boundLastLp = {}
    
    meanGNNdiffBest = {}
    meanGNNdiffLast = {}
    stdDevGNNdiffBest = {}
    stdDevGNNdiffLast = {}
    
    meanCostDiffBest = {}
    meanCostDiffLast = {}
    meanBoundBestIL = {}
    meanBoundLastIL = {}
    meanBoundBestLp = {}
    meanBoundLastLp = {}
    stdDevCostDiffBest = {}
    stdDevCostDiffLast = {}
    stdDevBoundBestIL = {}
    stdDevBoundLastIL = {}
    stdDevBoundBestLp = {}
    stdDevBoundLastLp = {}
        
    for thisModel in modelList:
        #Transform into np.array
        costBest[thisModel] = np.array(costBest[thisModel])
        costLast[thisModel] = np.array(costLast[thisModel])
        costBestHat[thisModel] = np.array(costBestHat[thisModel])
        costLastHat[thisModel] = np.array(costLastHat[thisModel])
        ILconstantBest[thisModel] = np.array(ILconstantBest[thisModel])
        ILconstantLast[thisModel] = np.array(ILconstantLast[thisModel])
        LpConstantBest[thisModel] = np.array(LpConstantBest[thisModel])
        LpConstantLast[thisModel] = np.array(LpConstantLast[thisModel])
        GNNdiffBest[thisModel] = np.array(GNNdiffBest[thisModel])
        GNNdiffLast[thisModel] = np.array(GNNdiffLast[thisModel])
        deltaN[thisModel] = np.array(deltaN[thisModel])

        #\\\ COMPUTE RELEVANT QUANTITIES:
        # Bound: C L F epsilon
        boundBestIL[thisModel] = ILconstantBest[thisModel] * np.sqrt(64) *\
                             np.tile(epsRelU.reshape(nSimPoints,1), (1,nDataSplits))
        boundLastIL[thisModel] = ILconstantBest[thisModel] * np.sqrt(64) *\
                             np.tile(epsRelU.reshape(nSimPoints,1), (1,nDataSplits))
        # Bound: C (1+delta sqrt(N)) epsilon L F
        boundBestLp[thisModel] = np.tile(LpConstantBest[thisModel].reshape(nSimPoints,nDataSplits,1), (1, 1, nPerturb)) \
                                          * (1 + deltaN[thisModel]) * np.sqrt(64) * \
                                            np.tile(epsAbsU.reshape(nSimPoints,1,1),
                                                    (1,nDataSplits,nPerturb))
        boundBestLp[thisModel] = np.mean(boundBestLp[thisModel], axis = 2)
        boundLastLp[thisModel] = np.tile(LpConstantLast[thisModel].reshape(nSimPoints,nDataSplits,1), (1, 1, nPerturb)) \
                                          * (1 + deltaN[thisModel]) * np.sqrt(64) * \
                                            np.tile(epsAbsU.reshape(nSimPoints,1,1),
                                                    (1,nDataSplits,nPerturb))
        boundLastLp[thisModel] = np.mean(boundLastLp[thisModel], axis = 2)
        
        # Relative cost difference
        # Add the extra dimension to costBest and costLast to handle the 
        # different number of perturbation realizations
        expandCostBest = costBest[thisModel].reshape(nSimPoints, nDataSplits, 1)
        expandCostBest = np.tile(expandCostBest, (1, 1, nPerturb))
        expandCostLast = costLast[thisModel].reshape(nSimPoints, nDataSplits, 1)
        expandCostLast = np.tile(expandCostLast, (1, 1, nPerturb))
        # And compute the cost difference
        costDiffBest[thisModel] = np.abs(expandCostBest-costBestHat[thisModel])\
                                        / expandCostBest
        costDiffLast[thisModel] = np.abs(expandCostLast-costLastHat[thisModel])\
                                        / expandCostLast
        # Average out the perturbation realizations
        costDiffBest[thisModel] = np.mean(costDiffBest[thisModel], axis=2)
        costDiffLast[thisModel] = np.mean(costDiffLast[thisModel], axis=2)
        GNNdiffBest[thisModel] = np.mean(GNNdiffBest[thisModel], axis = 2)
        GNNdiffLast[thisModel] = np.mean(GNNdiffLast[thisModel], axis = 2)
    
        #\\\ COMPUTE STATISTICS:
        meanBoundBestIL[thisModel] = np.mean(boundBestIL[thisModel], axis = 1)
        meanBoundLastIL[thisModel] = np.mean(boundLastIL[thisModel], axis = 1)
        stdDevBoundBestIL[thisModel] = np.std(boundBestIL[thisModel], axis = 1)
        stdDevBoundLastIL[thisModel] = np.std(boundLastIL[thisModel], axis = 1)
        meanBoundBestLp[thisModel] = np.mean(boundBestLp[thisModel], axis = 1)
        meanBoundLastLp[thisModel] = np.mean(boundLastLp[thisModel], axis = 1)
        stdDevBoundBestLp[thisModel] = np.std(boundBestLp[thisModel], axis = 1)
        stdDevBoundLastLp[thisModel] = np.std(boundLastLp[thisModel], axis = 1)
        
        meanCostDiffBest[thisModel] = np.mean(costDiffBest[thisModel], axis = 1)
        meanCostDiffLast[thisModel] = np.mean(costDiffLast[thisModel], axis = 1)
        stdDevCostDiffBest[thisModel] = np.std(costDiffBest[thisModel],axis = 1)
        stdDevCostDiffLast[thisModel] = np.std(costDiffLast[thisModel],axis = 1)
        
        meanGNNdiffBest[thisModel] = np.mean(GNNdiffBest[thisModel], axis = 1)
        meanGNNdiffLast[thisModel] = np.mean(GNNdiffLast[thisModel], axis = 1)
        stdDevGNNdiffBest[thisModel] = np.std(GNNdiffBest[thisModel], axis = 1)
        stdDevGNNdiffLast[thisModel] = np.std(GNNdiffLast[thisModel], axis = 1)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['modelLegend'] = modelLegend
    varsPickle['ratioTest'] = ratioTest
    varsPickle['epsRelN'] = epsRelN
    varsPickle['epsRelU'] = epsRelU
    varsPickle['epsAbsN'] = epsAbsN
    varsPickle['epsAbsU'] = epsAbsU
    
    varsPickle['costDiffBest'] = costDiffBest
    varsPickle['costDiffLast'] = costDiffLast
    varsPickle['boundBestIL'] = boundBestIL
    varsPickle['boundLastIL'] = boundLastIL
    varsPickle['boundBestLp'] = boundBestLp
    varsPickle['boundLastLp'] = boundLastLp
    
    varsPickle['meanCostDiffBest'] = meanCostDiffBest
    varsPickle['meanCostDiffLast'] = meanCostDiffLast
    varsPickle['meanBoundBestIL'] = meanBoundBestIL
    varsPickle['meanBoundLastIL'] = meanBoundLastIL
    varsPickle['meanBoundBestLp'] = meanBoundBestLp
    varsPickle['meanBoundLastLp'] = meanBoundLastLp
    varsPickle['stdDevCostDiffBest'] = stdDevCostDiffBest
    varsPickle['stdDevCostDiffLast'] = stdDevCostDiffLast
    varsPickle['stdDevBoundBestIL'] = stdDevBoundBestIL
    varsPickle['stdDevBoundLastIL'] = stdDevBoundLastIL
    varsPickle['stdDevBoundBestLp'] = stdDevBoundBestLp
    varsPickle['stdDevBoundLastLp'] = stdDevBoundLastLp
    
    varsPickle['meanGNNdiffBest'] = meanGNNdiffBest
    varsPickle['meanGNNdiffLast'] = meanGNNdiffLast
    varsPickle['stdDevGNNdiffBest'] = stdDevGNNdiffBest
    varsPickle['stdDevGNNdiffLast'] = stdDevGNNdiffLast
   
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
        
    ########
    # PLOT #
    ########

    diffBestFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(ratioTest, meanGNNdiffBest[thisModel],
                     yerr = stdDevGNNdiffBest[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(ratioTest, meanBoundBestIL[thisModel],
                     yerr = stdDevBoundBestIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'Ratio Test', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffBestFig.savefig(os.path.join(saveDirFigs,'diffBest.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffBestFig)
    
    diffBestRelFigU = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(epsRelU, meanGNNdiffBest[thisModel],
                     yerr = stdDevGNNdiffBest[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(epsRelU, meanBoundBestIL[thisModel],
                     yerr = stdDevBoundBestIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffBestRelFigU.savefig(os.path.join(saveDirFigs,'diffBestRelU.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffBestRelFigU)
    
    diffBestAbsFigU = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(epsAbsU, meanGNNdiffBest[thisModel],
                     yerr = stdDevGNNdiffBest[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(epsAbsU, meanBoundBestLp[thisModel],
                     yerr = stdDevBoundBestLp[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffBestAbsFigU.savefig(os.path.join(saveDirFigs,'diffBestAbsU.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffBestAbsFigU)
    
    diffLastFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(ratioTest, meanGNNdiffLast[thisModel],
                     yerr = stdDevGNNdiffLast[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(ratioTest, meanBoundLastIL[thisModel],
                     yerr = stdDevBoundLastIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'Ratio Test', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffLastFig.savefig(os.path.join(saveDirFigs,'diffLast.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffLastFig)
    
    diffLastRelFigU = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(epsRelU, meanGNNdiffLast[thisModel],
                     yerr = stdDevGNNdiffLast[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(epsRelU, meanBoundLastIL[thisModel],
                     yerr = stdDevBoundLastIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffLastRelFigU.savefig(os.path.join(saveDirFigs,'diffLastRelU.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffLastRelFigU)
    
    diffLastAbsFigU = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(epsAbsU, meanGNNdiffLast[thisModel],
                     yerr = stdDevGNNdiffLast[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(epsAbsU, meanBoundLastLp[thisModel],
                     yerr = stdDevBoundLastLp[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffLastAbsFigU.savefig(os.path.join(saveDirFigs,'diffLastAbsU.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffLastAbsFigU)
    
    costDiffBestFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(ratioTest, meanCostDiffBest[thisModel],
                     yerr = stdDevCostDiffBest[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    #plt.xscale('log')
    plt.xlabel(r'Ratio Test', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffBestFig.savefig(os.path.join(saveDirFigs,'costDiffBest.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffBestFig)
    
    costDiffLastFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(ratioTest, meanCostDiffLast[thisModel],
                     yerr = stdDevCostDiffLast[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    #plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffLastFig.savefig(os.path.join(saveDirFigs,'costDiffLast.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffLastFig)

#%%##################################################################
#                                                                   #
#                    END RUNNING TIME                               #
#                                                                   #
#####################################################################

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))