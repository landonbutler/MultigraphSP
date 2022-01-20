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
thisFilename = 'movieStabilitySynthetic' # This is the general name of all related files

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
useLaplacian = True # Decide to use the Laplacian as the GSO

epsMin = 1e-3 # Minimum value of perturbation
epsMax = 1e0 # Maximum value of perturbation
nSimPoints = 10 # Number of simulation points
eps = np.logspace(np.log10(epsMin), np.log10(epsMax), num=nSimPoints)

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
                'useLaplacian': useLaplacian,
                'epsMin': epsMin,
                'epsMax': epsMax,
                'nSimPoints': nSimPoints,
                'eps': eps,
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
doNoPenaltyGNN = True
doNoPenaltyMGNN = True
doILpenaltyMGNN = True

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
# be called by unpacking the dictionary

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

if doNoPenaltyGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelNoPnlt = {} # Model parameters for the Local GNN (LclGNN)
    modelNoPnlt['name'] = 'NoPnltGNN'
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
    modelLegend[modelNoPnlt['name']] = 'Multi-Channel GNN'

#\\\\\\\\\\\\\\\\\\
#\\\ NO PENALTY \\\
#\\\\\\\\\\\\\\\\\\

if doNoPenaltyMGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelNoPnlt = {} # Model parameters for the Local GNN (LclGNN)
    modelNoPnlt['name'] = 'NoPnltGNN'
    modelNoPnlt['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelNoPnlt['archit'] = archit.MultiGNN
    # Graph convolutional parameters
    modelNoPnlt['dimNodeSignals'] = [1, 64] # Features per layer
    modelNoPnlt['nFilterTaps'] = [3] # Number of filter taps per layer
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
    modelLegend[modelNoPnlt['name']] = 'MultiGNN'


#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ WITH IL PENALTY \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doILpenaltyMGNN:    
    
    #\\\/// Model 1: 1 penalty \\\///

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelILpnl1 = {} # Model parameters for the Local GNN (LclGNN)
    modelILpnl1['name'] = 'MGNN ILpnl 0.1'
    modelILpnl1['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelILpnl1['archit'] = archit.MultiGNN
    # Graph convolutional parameters
    modelILpnl1['dimNodeSignals'] = [1, 64] # Features per layer
    modelILpnl1['nFilterTaps'] = [3] # Number of filter taps per layer
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
    
    modelILpnl1['penalty'] = ('ILconstant', 0.1) # Penalty function name, and
        # penalty multiplier
    modelILpnl1['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)

    #\\\ Save Values:
    writeVarValues(varsFile, modelILpnl1)
    modelList += [modelILpnl1['name']]
    modelLegend[modelILpnl1['name']] = 'MGNN 0.1 (IL)'
    
    #\\\/// Model 2: 0.5 penalty \\\///
    
    modelILpnl2 = deepcopy(modelILpnl1)
    modelILpnl2['name'] = 'MGNN ILpnl 0.5'
    modelILpnl2['penalty'] = ('ILconstant', 0.5) # Penalty function name, and
        # penalty multiplier
    modelILpnl2['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)
    writeVarValues(varsFile, modelILpnl2)
    modelList += [modelILpnl2['name']]
    modelLegend[modelILpnl2['name']] = 'MGNN 0.5 (IL)'
    
    #\\\/// Model 3: 2 penalty \\\///
    
    modelILpnl3 = deepcopy(modelILpnl1)
    modelILpnl3['name'] = 'MGNN ILpnl 1.0'
    modelILpnl3['penalty'] = ('ILconstant', 1.) # Penalty function name, and
        # penalty multiplier
    modelILpnl3['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)
    writeVarValues(varsFile, modelILpnl3)
    modelList += [modelILpnl3['name']]
    modelLegend[modelILpnl3['name']] = 'MGNN 1.0 (IL)'

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
costBestRel = {} # Cost for the best model under relative perturbation
costLastRel = {} # Cost for the last model under relative perturbation
costBestAbs = {} # Cost for the best model under absolute perturbation
costLastAbs = {} # Cost for the last model under absolute perturbation
ILconstantBest = {} # Save the value of the IL constant for best model
ILconstantLast = {} # Save the value of the IL constant for last model
LpConstantBest = {} # Save the value of the Lipschitz constant for best model
LpConstantLast = {} # Save the value of the Lipschitz constant for last model
GNNdiffBestRel = {} # Value of the difference in the GNN output for best model
GNNdiffLastRel = {} # Value of the difference in the GNN output for last model
GNNdiffBestAbs = {} # Value of the difference in the GNN output for best model
GNNdiffLastAbs = {} # Value of the difference in the GNN output for last model
deltaN = {} # Value of delta * sqrt(N) for each model
eRel = {} # Actual value of epsilon (after normalization) for the relative model
eAbs = {} # Actual value of epsilon (after normalization) for the absolute model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nSimPoints
    costLast[thisModel] = [None] * nSimPoints
    costBestRel[thisModel] = [None] * nSimPoints
    costLastRel[thisModel] = [None] * nSimPoints
    costBestAbs[thisModel] = [None] * nSimPoints
    costLastAbs[thisModel] = [None] * nSimPoints
    ILconstantBest[thisModel] = [None] * nSimPoints
    ILconstantLast[thisModel] = [None] * nSimPoints
    LpConstantBest[thisModel] = [None] * nSimPoints
    LpConstantLast[thisModel] = [None] * nSimPoints
    GNNdiffBestRel[thisModel] = [None] * nSimPoints
    GNNdiffLastRel[thisModel] = [None] * nSimPoints
    GNNdiffBestAbs[thisModel] = [None] * nSimPoints
    GNNdiffLastAbs[thisModel] = [None] * nSimPoints
    deltaN[thisModel] = [None] * nSimPoints
    eRel[thisModel] = [None] * nSimPoints
    eAbs[thisModel] = [None] * nSimPoints
    for n in range(nSimPoints):
        costBest[thisModel][n] = [None] * nDataSplits
        costLast[thisModel][n] = [None] * nDataSplits
        costBestRel[thisModel][n] = [None] * nDataSplits
        costLastRel[thisModel][n] = [None] * nDataSplits
        costBestAbs[thisModel][n] = [None] * nDataSplits
        costLastAbs[thisModel][n] = [None] * nDataSplits
        ILconstantBest[thisModel][n] = [None] * nDataSplits
        ILconstantLast[thisModel][n] = [None] * nDataSplits
        LpConstantBest[thisModel][n] = [None] * nDataSplits
        LpConstantLast[thisModel][n] = [None] * nDataSplits
        GNNdiffBestRel[thisModel][n] = [None] * nDataSplits
        GNNdiffLastRel[thisModel][n] = [None] * nDataSplits
        GNNdiffBestAbs[thisModel][n] = [None] * nDataSplits
        GNNdiffLastAbs[thisModel][n] = [None] * nDataSplits
        deltaN[thisModel][n] = [None] * nDataSplits
        eRel[thisModel][n] = [None] * nDataSplits
        eAbs[thisModel][n] = [None] * nDataSplits
        for split in range(nDataSplits):
            costBestRel[thisModel][n][split] = [None] * nPerturb
            costLastRel[thisModel][n][split] = [None] * nPerturb
            costBestAbs[thisModel][n][split] = [None] * nPerturb
            costLastAbs[thisModel][n][split] = [None] * nPerturb
            GNNdiffBestRel[thisModel][n][split] = [None] * nPerturb
            GNNdiffLastRel[thisModel][n][split] = [None] * nPerturb
            GNNdiffBestAbs[thisModel][n][split] = [None] * nPerturb
            GNNdiffLastAbs[thisModel][n][split] = [None] * nPerturb
            deltaN[thisModel][n][split] = [None] * nPerturb
            eRel[thisModel][n][split] = [None] * nPerturb
            eAbs[thisModel][n][split] = [None] * nPerturb
            

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
        print("\n[Sim %d] Epsilon = %.4f" % (n, eps[n]))

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
    
    
        G1 = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                             {'adjacencyMatrix': adjacencyMatrix})
        G1.computeGFT() # Compute the GFT of the stored GSO

        genreMatrix = data.getGenreGraph()
        G2 = graphTools.Graph('adjacency', genreMatrix.shape[0],
                             {'adjacencyMatrix': adjacencyMatrix})
        G2.computeGFT() # Compute the GFT of the stored GSO


    
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
            if useLaplacian:
                SL1 = graphTools.normalizeLaplacian(G1.L)
                EL1, _ = graphTools.computeGFT(SL1, order = 'increasing')
                S1 = 2 * SL1/np.max(np.real(EL1)) - np.eye(G1.N)

                SL2 = graphTools.normalizeLaplacian(G2.L)
                EL2, _ = graphTools.computeGFT(SL2, order = 'increasing')
                S2 = 2 * SL2/np.max(np.real(EL2)) - np.eye(G2.N)

                S = np.stack((S1, S2), axis = 0)
            else:
                S1 = G1.S.copy()/np.max(np.real(G1.E))
                S2 = G2.S.copy()/np.max(np.real(G2.E))
                S = np.stack((S1, S2), axis = 0)
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
            yHatSbest = thisEvalVars['yGNNbest']
            yHatSlast = thisEvalVars['yGNNlast']
            
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
                #thisS = modelsGNN[thisModel].archit.S.data.cpu().numpy() # ExNxN
                thisS = G.S.copy().reshape((1, G.N, G.N))
                nEdgeFeatures = thisS.shape[0] 
                nNodes = thisS.shape[1]
                # Compute the perturbation matrix
                #   Compute the error matrix
                #   Don't forget to satisfy the structural constraint
                #   ||E/||E||-I|| < eps. For a diagonal matrix, this translates
                #   into |e_i/e_max - 1| < eps, and translates into
                #   (1-eps) e_max < e_i < (1+eps) e_max. Also, 
                #   e_i < e_max < eps/2.
                #   Since both constraints depend on e_max, we fix e_max to
                #   eps/2 and then get the rest at random between (1-eps) e_max
                #   and e_max
                eMax = eps[n]/2
                Erel = np.random.uniform(low = (1-eps[n])*eMax, high = eMax,
                                         size = (nEdgeFeatures, nNodes-1))
                eMax = np.array([eMax] * nEdgeFeatures).reshape(nEdgeFeatures,1)
                Erel = np.concatenate((Erel, eMax), axis = 1) # E x N
                #Erel = eps[n]/2 * np.random.rand(nEdgeFeatures, nNodes) # E x N
                #   Compute the perturbation
                ShatRel = np.zeros((nEdgeFeatures, nNodes, nNodes))
                ShatAbs = np.zeros((nEdgeFeatures, nNodes, nNodes))
                for e in range(nEdgeFeatures):
                    ShatRel[e] = thisS[e] \
                                    + np.diag(Erel[e]) @ thisS[e] \
                                    + thisS[e] @ np.diag(Erel[e])
                    # Absolute perturbation (respect the sparsity)
                    Eabs = np.random.rand(nNodes, nNodes)
                    Eabs = Eabs * (np.abs(thisS[e]) > graphTools.zeroTolerance)\
                                                         .astype(thisS[e].dtype)
                    Eabs = 0.5 * (Eabs + Eabs.T) # Symmetrize it
                    Eabs = eps[n]/np.linalg.norm(Eabs, ord = 2) * Eabs
                    # Add it
                    ShatAbs[e] = thisS[e] + Eabs
                #   Normalize Shat
                if useLaplacian:
                    for e in range(nEdgeFeatures):
                        LhatRel = graphTools.adjacencyToLaplacian(ShatRel[e])
                        LhatRel = graphTools.normalizeLaplacian(LhatRel)
                        egvs, _ = graphTools.computeGFT(LhatRel,
                                                        order = 'increasing')
                        ShatRel[e] = 2 * LhatRel/np.max(np.real(egvs)) \
                                                                - np.eye(nNodes)
                        
                        LhatAbs = graphTools.adjacencyToLaplacian(ShatAbs[e])
                        LhatAbs = graphTools.normalizeLaplacian(LhatAbs)
                        egvs, _ = graphTools.computeGFT(LhatAbs,
                                                        order = 'increasing')
                        ShatAbs[e] = 2 * LhatAbs/np.max(np.real(egvs)) \
                                                                - np.eye(nNodes)
                else:
                    egvs, _ = graphTools.computeGFT(ShatRel, doMatrix = False)
                    ShatRel = ShatRel / np.max(np.real(egvs))
                    egvs, _ = graphTools.computeGFT(ShatAbs, doMatrix = False)
                    ShatAbs = ShatAbs / np.max(np.real(egvs))
                #   Compute the delta that we need for the bound (the maximum
                #   across all edges)
                thisDelta = 0.
                for e in range(nEdgeFeatures):
                    _, V = graphTools.computeGFT(thisS[e])
                    _, U = graphTools.computeGFT(Eabs)
                    thisDelta = np.max((thisDelta,
                                       (np.linalg.norm(U-V, ord = 2)+1)**2 - 1))
                    deltaN[modelName][n][split][p] = thisDelta * np.sqrt(nNodes)
                #   Compute the actual values of epsilon for the normalized
                #   versions
                thisSnorm = modelsGNN[thisModel].archit.S.data.cpu().numpy()
                for e in range(nEdgeFeatures):
                    eRel[modelName][n][split][p] = \
                            np.linalg.norm(thisSnorm[e] - ShatRel[e], ord = 2)/\
                                           np.linalg.norm(thisSnorm[e], ord = 2)
                    eAbs[modelName][n][split][p] = \
                            np.linalg.norm(thisSnorm[e] - ShatAbs[e], ord = 2)
                
                #\\\ Relative error
                # Change the GSO
                modelsGNN[thisModel].archit.changeGSO(ShatRel)
                #   And move it to device
                modelsGNN[thisModel].archit.to(modelsGNN[thisModel].device)
                
                # Compute the output
                thisEvalRelVars = modelsGNN[thisModel].evaluate(data)
                #   Get the output values
                thisCostBestRel = thisEvalRelVars['costBest']
                thisCostLastRel = thisEvalRelVars['costLast']
                yHatSbestRel = thisEvalRelVars['yGNNbest']
                yHatSlastRel = thisEvalRelVars['yGNNlast']
                
                # Compare
                #   The output is just a single scalar, so the comparison is 
                #   straightforward absolute value
                #   Now we have B x F[-1] vectors, so we need to compute the 
                #   norm across the F[-1] dimension
                thisGNNdiffBestRel = np.sum((yHatSbest - yHatSbestRel) ** 2,
                                            axis = 1)
                thisGNNdiffBestRel = np.mean(np.sqrt(thisGNNdiffBestRel))
                thisGNNdiffLastRel = np.sum((yHatSlast - yHatSlastRel) ** 2,
                                            axis = 1)
                thisGNNdiffLastRel = np.mean(np.sqrt(thisGNNdiffLastRel))
                
                # Save
                costBestRel[modelName][n][split][p] = thisCostBestRel
                costLastRel[modelName][n][split][p] = thisCostLastRel
                GNNdiffBestRel[modelName][n][split][p] = thisGNNdiffBestRel
                GNNdiffLastRel[modelName][n][split][p] = thisGNNdiffLastRel
                
                #\\\ Absolute error
                # Change the GSO
                modelsGNN[thisModel].archit.changeGSO(ShatAbs)
                #   And move it to device
                modelsGNN[thisModel].archit.to(modelsGNN[thisModel].device)
                
                # Compute the output
                thisEvalAbsVars = modelsGNN[thisModel].evaluate(data)
                #   Get the output values
                thisCostBestAbs = thisEvalAbsVars['costBest']
                thisCostLastAbs = thisEvalAbsVars['costLast']
                yHatSbestAbs = thisEvalAbsVars['yGNNbest']
                yHatSlastAbs = thisEvalAbsVars['yGNNlast']
                
                # Compare
                #   The output is just a single scalar, so the comparison is 
                #   straightforward absolute value
                thisGNNdiffBestAbs = np.sum((yHatSbest - yHatSbestAbs) ** 2,
                                            axis = 1)
                thisGNNdiffBestAbs = np.mean(np.sqrt(thisGNNdiffBestAbs))
                thisGNNdiffLastAbs = np.sum((yHatSlast - yHatSlastAbs) ** 2,
                                            axis = 1)
                thisGNNdiffLastAbs = np.mean(np.sqrt(thisGNNdiffLastAbs))
            
                # Save
                costBestAbs[modelName][n][split][p] = thisCostBestAbs
                costLastAbs[modelName][n][split][p] = thisCostLastAbs
                GNNdiffBestAbs[modelName][n][split][p] = thisGNNdiffBestAbs
                GNNdiffLastAbs[modelName][n][split][p] = thisGNNdiffLastAbs
                
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
    epsRel = np.zeros((nSimPoints, len(modelList)))
    epsAbs = np.zeros((nSimPoints, len(modelList)))
    l = 0
    for thisModel in modelList:
        eRel[thisModel] = np.array(eRel[thisModel]) # nSimPoints x nDataSplits x nPerturb
        eAbs[thisModel] = np.array(eAbs[thisModel]) # nSimPoints x nDataSplits x nPerturb
        
        thisEpsRel = np.mean(eRel[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsRel[:,l] = np.mean(thisEpsRel, axis = 1) # nSimPoints
        thisEpsAbs = np.mean(eAbs[thisModel], axis = 2) # nSimPoints x nDataSplits
        epsAbs[:,l] = np.mean(thisEpsAbs, axis = 1) # nSimPoints
        
        l += 1
    epsRel = np.mean(epsRel, axis = 1) # nSimPoints
    epsAbs = np.mean(epsAbs, axis = 1) # nSimPoints

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
    costDiffBestRel = {}
    costDiffLastRel = {}
    costDiffBestAbs = {}
    costDiffLastAbs = {}
    boundBestIL = {}
    boundLastIL = {}
    boundBestLp = {}
    boundLastLp = {}
    
    meanGNNdiffBestRel = {}
    meanGNNdiffLastRel = {}
    stdDevGNNdiffBestRel = {}
    stdDevGNNdiffLastRel = {}
    meanGNNdiffBestAbs = {}
    meanGNNdiffLastAbs = {}
    stdDevGNNdiffBestAbs = {}
    stdDevGNNdiffLastAbs = {}
    
    meanCostDiffBestRel = {}
    meanCostDiffLastRel = {}
    meanCostDiffBestAbs = {}
    meanCostDiffLastAbs = {}
    meanBoundBestIL = {}
    meanBoundLastIL = {}
    meanBoundBestLp = {}
    meanBoundLastLp = {}
    stdDevCostDiffBestRel = {}
    stdDevCostDiffLastRel = {}
    stdDevCostDiffBestAbs = {}
    stdDevCostDiffLastAbs = {}
    stdDevBoundBestIL = {}
    stdDevBoundLastIL = {}
    stdDevBoundBestLp = {}
    stdDevBoundLastLp = {}
        
    for thisModel in modelList:
        #Transform into np.array
        costBest[thisModel] = np.array(costBest[thisModel])
        costLast[thisModel] = np.array(costLast[thisModel])
        costBestRel[thisModel] = np.array(costBestRel[thisModel])
        costLastRel[thisModel] = np.array(costLastRel[thisModel])
        costBestAbs[thisModel] = np.array(costBestAbs[thisModel])
        costLastAbs[thisModel] = np.array(costLastAbs[thisModel])
        ILconstantBest[thisModel] = np.array(ILconstantBest[thisModel])
        ILconstantLast[thisModel] = np.array(ILconstantLast[thisModel])
        LpConstantBest[thisModel] = np.array(LpConstantBest[thisModel])
        LpConstantLast[thisModel] = np.array(LpConstantLast[thisModel])
        GNNdiffBestRel[thisModel] = np.array(GNNdiffBestRel[thisModel])
        GNNdiffLastRel[thisModel] = np.array(GNNdiffLastRel[thisModel])
        GNNdiffBestAbs[thisModel] = np.array(GNNdiffBestAbs[thisModel])
        GNNdiffLastAbs[thisModel] = np.array(GNNdiffLastAbs[thisModel])
        deltaN[thisModel] = np.array(deltaN[thisModel])

        #\\\ COMPUTE RELEVANT QUANTITIES:
        # Bound: C L F epsilon
        boundBestIL[thisModel] = ILconstantBest[thisModel] * np.sqrt(64) *\
                             np.tile(eps.reshape(nSimPoints,1), (1,nDataSplits))
        boundLastIL[thisModel] = ILconstantBest[thisModel] * np.sqrt(64) *\
                             np.tile(eps.reshape(nSimPoints,1), (1,nDataSplits))
        # Bound: C (1+delta sqrt(N)) epsilon L F
        boundBestLp[thisModel] = np.tile(LpConstantBest[thisModel].reshape(nSimPoints,nDataSplits,1), (1, 1, nPerturb)) \
                                          * (1 + deltaN[thisModel]) * np.sqrt(64) * \
                                            np.tile(eps.reshape(nSimPoints,1,1),
                                                    (1,nDataSplits,nPerturb))
        boundBestLp[thisModel] = np.mean(boundBestLp[thisModel], axis = 2)
        boundLastLp[thisModel] = np.tile(LpConstantLast[thisModel].reshape(nSimPoints,nDataSplits,1), (1, 1, nPerturb)) \
                                          * (1 + deltaN[thisModel]) * np.sqrt(64) * \
                                            np.tile(eps.reshape(nSimPoints,1,1),
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
        costDiffBestRel[thisModel] = np.abs(expandCostBest \
                                                      - costBestRel[thisModel])\
                                        / expandCostBest
        costDiffLastRel[thisModel] = np.abs(expandCostLast \
                                                      - costLastRel[thisModel])\
                                        / expandCostLast
        costDiffBestAbs[thisModel] = np.abs(expandCostBest \
                                                      - costBestAbs[thisModel])\
                                        / expandCostBest
        costDiffLastAbs[thisModel] = np.abs(expandCostLast \
                                                      - costLastAbs[thisModel])\
                                        / expandCostLast
        # Average out the perturbation realizations
        costDiffBestRel[thisModel] = np.mean(costDiffBestRel[thisModel], axis=2)
        costDiffLastRel[thisModel] = np.mean(costDiffLastRel[thisModel], axis=2)
        costDiffBestAbs[thisModel] = np.mean(costDiffBestAbs[thisModel], axis=2)
        costDiffLastAbs[thisModel] = np.mean(costDiffLastAbs[thisModel], axis=2)
        GNNdiffBestRel[thisModel] = np.mean(GNNdiffBestRel[thisModel], axis = 2)
        GNNdiffLastRel[thisModel] = np.mean(GNNdiffLastRel[thisModel], axis = 2)
        GNNdiffBestAbs[thisModel] = np.mean(GNNdiffBestAbs[thisModel], axis = 2)
        GNNdiffLastAbs[thisModel] = np.mean(GNNdiffLastAbs[thisModel], axis = 2)
    
        #\\\ COMPUTE STATISTICS:
        meanBoundBestIL[thisModel] = np.mean(boundBestIL[thisModel], axis = 1)
        meanBoundLastIL[thisModel] = np.mean(boundLastIL[thisModel], axis = 1)
        stdDevBoundBestIL[thisModel] = np.std(boundBestIL[thisModel], axis = 1)
        stdDevBoundLastIL[thisModel] = np.std(boundLastIL[thisModel], axis = 1)
        meanBoundBestLp[thisModel] = np.mean(boundBestLp[thisModel], axis = 1)
        meanBoundLastLp[thisModel] = np.mean(boundLastLp[thisModel], axis = 1)
        stdDevBoundBestLp[thisModel] = np.std(boundBestLp[thisModel], axis = 1)
        stdDevBoundLastLp[thisModel] = np.std(boundLastLp[thisModel], axis = 1)
        
        meanCostDiffBestRel[thisModel] = np.mean(costDiffBestRel[thisModel],
                                                 axis = 1)
        meanCostDiffLastRel[thisModel] = np.mean(costDiffLastRel[thisModel],
                                                 axis = 1)
        stdDevCostDiffBestRel[thisModel] = np.std(costDiffBestRel[thisModel],
                                                  axis = 1)
        stdDevCostDiffLastRel[thisModel] = np.std(costDiffLastRel[thisModel],
                                                  axis = 1)
        meanCostDiffBestAbs[thisModel] = np.mean(costDiffBestAbs[thisModel],
                                                 axis = 1)
        meanCostDiffLastAbs[thisModel] = np.mean(costDiffLastAbs[thisModel],
                                                 axis = 1)
        stdDevCostDiffBestAbs[thisModel] = np.std(costDiffBestAbs[thisModel],
                                                  axis = 1)
        stdDevCostDiffLastAbs[thisModel] = np.std(costDiffLastAbs[thisModel],
                                                  axis = 1)
        
        meanGNNdiffBestRel[thisModel] = np.mean(GNNdiffBestRel[thisModel],
                                                axis = 1)
        meanGNNdiffLastRel[thisModel] = np.mean(GNNdiffLastRel[thisModel],
                                                axis = 1)
        stdDevGNNdiffBestRel[thisModel] = np.std(GNNdiffBestRel[thisModel],
                                                 axis = 1)
        stdDevGNNdiffLastRel[thisModel] = np.std(GNNdiffLastRel[thisModel],
                                                 axis = 1)
        meanGNNdiffBestAbs[thisModel] = np.mean(GNNdiffBestAbs[thisModel],
                                                axis = 1)
        meanGNNdiffLastAbs[thisModel] = np.mean(GNNdiffLastAbs[thisModel],
                                                axis = 1)
        stdDevGNNdiffBestAbs[thisModel] = np.std(GNNdiffBestAbs[thisModel],
                                                 axis = 1)
        stdDevGNNdiffLastAbs[thisModel] = np.std(GNNdiffLastAbs[thisModel],
                                                 axis = 1)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['modelLegend'] = modelLegend
    varsPickle['eps'] = eps
    
    varsPickle['costDiffBestRel'] = costDiffBestRel
    varsPickle['costDiffLastRel'] = costDiffLastRel
    varsPickle['costDiffBestAbs'] = costDiffBestAbs
    varsPickle['costDiffLastAbs'] = costDiffLastAbs
    varsPickle['boundBestIL'] = boundBestIL
    varsPickle['boundLastIL'] = boundLastIL
    varsPickle['boundBestLp'] = boundBestLp
    varsPickle['boundLastLp'] = boundLastLp
    
    varsPickle['meanCostDiffBestRel'] = meanCostDiffBestRel
    varsPickle['meanCostDiffLastRel'] = meanCostDiffLastRel
    varsPickle['meanCostDiffBestAbs'] = meanCostDiffBestAbs
    varsPickle['meanCostDiffLastAbs'] = meanCostDiffLastAbs
    varsPickle['meanBoundBestIL'] = meanBoundBestIL
    varsPickle['meanBoundLastIL'] = meanBoundLastIL
    varsPickle['meanBoundBestLp'] = meanBoundBestLp
    varsPickle['meanBoundLastLp'] = meanBoundLastLp
    varsPickle['stdDevCostDiffBestRel'] = stdDevCostDiffBestRel
    varsPickle['stdDevCostDiffLastRel'] = stdDevCostDiffLastRel
    varsPickle['stdDevCostDiffBestAbs'] = stdDevCostDiffBestAbs
    varsPickle['stdDevCostDiffLastAbs'] = stdDevCostDiffLastAbs
    varsPickle['stdDevBoundBestIL'] = stdDevBoundBestIL
    varsPickle['stdDevBoundLastIL'] = stdDevBoundLastIL
    varsPickle['stdDevBoundBestLp'] = stdDevBoundBestLp
    varsPickle['stdDevBoundLastLp'] = stdDevBoundLastLp
    
    varsPickle['meanGNNdiffBestRel'] = meanGNNdiffBestRel
    varsPickle['meanGNNdiffLastRel'] = meanGNNdiffLastRel
    varsPickle['stdDevGNNdiffBestRel'] = stdDevGNNdiffBestRel
    varsPickle['stdDevGNNdiffLastRel'] = stdDevGNNdiffLastRel
    varsPickle['meanGNNdiffBestAbs'] = meanGNNdiffBestAbs
    varsPickle['meanGNNdiffLastAbs'] = meanGNNdiffLastAbs
    varsPickle['stdDevGNNdiffBestAbs'] = stdDevGNNdiffBestAbs
    varsPickle['stdDevGNNdiffLastAbs'] = stdDevGNNdiffLastAbs
   
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
        
    ########
    # PLOT #
    ########
    
    
    ###############
    ### RESCALE ###
    ###         ###
    eps = eps/1e2 #
    ###         ###
    ### RESCALE ###
    ###############
    

    diffBestRelFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanGNNdiffBestRel[thisModel],
                     yerr = stdDevGNNdiffBestRel[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(eps, meanBoundBestIL[thisModel],
                     yerr = stdDevBoundBestIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffBestRelFig.savefig(os.path.join(saveDirFigs,'diffBestRel.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffBestRelFig)
    
    diffBestAbsFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanGNNdiffBestAbs[thisModel],
                     yerr = stdDevGNNdiffBestAbs[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(eps, meanBoundBestLp[thisModel],
                     yerr = stdDevBoundBestLp[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffBestAbsFig.savefig(os.path.join(saveDirFigs,'diffBestAbs.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffBestAbsFig)
    
    diffLastRelFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanGNNdiffLastRel[thisModel],
                     yerr = stdDevGNNdiffLastRel[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(eps, meanBoundLastIL[thisModel],
                     yerr = stdDevBoundLastIL[thisModel],
                     linestyle = '--',
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList)
    diffLastRelFig.savefig(os.path.join(saveDirFigs,'diffLastRel.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffLastRelFig)
    
    diffLastAbsFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanGNNdiffLastAbs[thisModel],
                     yerr = stdDevGNNdiffLastAbs[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        plt.errorbar(eps, meanBoundLastLp[thisModel],
                     yerr = stdDevBoundLastLp[thisModel],
                     linestyle = '--', 
                     linewidth = 0.75*lineWidth,
                     color = colorPenn[selectColor])
        legendList.append(r'%s (bound)' % modelLegend[thisModel])
        nColor += 1
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'$\| \boldsymbol{\Phi}(\mathbf{S}, \mathbf{x}) - \boldsymbol{\Phi}(\hat{\mathbf{S}}, \mathbf{x})\|$', fontsize = fontSize)
    plt.legend(legendList, fontsize = fontSize)
    diffLastAbsFig.savefig(os.path.join(saveDirFigs,'diffLastAbs.pdf'),
                        bbox_inches = 'tight')
    plt.close(fig = diffLastAbsFig)
    
    costDiffBestRelFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanCostDiffBestRel[thisModel],
                     yerr = stdDevCostDiffBestRel[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffBestRelFig.savefig(os.path.join(saveDirFigs,'costDiffBestRel.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffBestRelFig)
    
    costDiffBestAbsFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanCostDiffBestAbs[thisModel],
                     yerr = stdDevCostDiffBestAbs[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffBestAbsFig.savefig(os.path.join(saveDirFigs,'costDiffBestAbs.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffBestAbsFig)
    
    costDiffLastRelFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanCostDiffLastRel[thisModel],
                     yerr = stdDevCostDiffLastRel[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffLastRelFig.savefig(os.path.join(saveDirFigs,'costDiffLastRel.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffLastRelFig)
    
    costDiffLastAbsFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    legendList = []
    nColor = 0
    for thisModel in modelList:
        selectColor = np.mod(nColor+1, 5)
        plt.errorbar(eps, meanCostDiffLastAbs[thisModel],
                     yerr = stdDevCostDiffLastAbs[thisModel],
                     linewidth = lineWidth,
                     color = colorPenn[selectColor],
                     marker = markerShape,
                     markersize = markerSize)
        legendList.append(r'%s' % modelLegend[thisModel])
        nColor += 1
    plt.xscale('log')
    plt.xlabel(r'$\varepsilon$', fontsize = fontSize)
    plt.ylabel(r'RMSE Difference', fontsize = fontSize)
    plt.legend(legendList)
    costDiffLastAbsFig.savefig(os.path.join(saveDirFigs,'costDiffLastAbs.pdf'),
                            bbox_inches = 'tight')
    plt.close(fig = costDiffLastAbsFig)

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