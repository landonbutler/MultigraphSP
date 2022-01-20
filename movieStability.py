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
thisFilename = 'movieStability' # This is the general name of all related files

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

ratioTrain = 0.9 # Ratio of training samples
ratioValid = 0.1 # Ratio of validation samples (out of the total training
# samples)
# Final split is:
#   nValidation = round(ratioValid * ratioTrain * nTotal)
#   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
#   nTest = nTotal - nTrain - nValidation
maxNodes = 200 # Maximum number of nodes (select the ones with the largest
#   number of ratings)
minRatings = 0 # Discard samples (rows and columns) with less than minRatings 
    # ratings
interpolateRatings = False # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

nDataSplits = 2 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to obtain some result that is more robust to this split.
    
# Given that we build the graph from a training split selected at random, it
# could happen that it is disconnected, or directed, or what not. In other 
# words, we might want to force (by removing nodes) some useful characteristics
# on the graph
keepIsolatedNodes = True # If True keeps isolated nodes ---> FALSE
forceUndirected = True # If True forces the graph to be undirected
forceConnected = False # If True returns the largest connected component of the
    # graph as the main graph ---> TRUE
kNN = 10 # Number of nearest neighbors

maxDataPoints = None # None to consider all data points

#\\\ Save values:
writeVarValues(varsFile,
               {'labelID': labelID,
                'graphType': graphType,
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

#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ WITH IL PENALTY \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doILpenalty:

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelILpnlt = {} # Model parameters for the Local GNN (LclGNN)
    modelILpnlt['name'] = 'ILpnlt'
    modelILpnlt['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelILpnlt['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelILpnlt['dimNodeSignals'] = [1, 64] # Features per layer
    modelILpnlt['nFilterTaps'] = [5] # Number of filter taps per layer
    modelILpnlt['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelILpnlt['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelILpnlt['poolingFunction'] = gml.NoPool # Summarizing function
    modelILpnlt['nSelectedNodes'] = None # To be determined later on
    modelILpnlt['poolingSize'] = [1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelILpnlt['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelILpnlt['GSO'] = None # To be determined later on, based on data
    modelILpnlt['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER

    modelILpnlt['trainer'] = training.TrainerSingleNode
    
    #\\\ EVALUATOR
    
    modelILpnlt['evaluator'] = evaluation.evaluateSingleNode
    
    #\\\ LOSS FUNCTION
    
    modelILpnlt['penalty'] = ('ILconstant', 0.5) # Penalty function name, and
        # penalty multiplier
    modelILpnlt['lossFunction'] = loss.adaptExtraDimensionLoss(lossFunction)

    #\\\ Save Values:
    writeVarValues(varsFile, modelILpnlt)
    modelList += [modelILpnlt['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 5 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 20 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

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
costBest = {} # Accuracy for the best model
costLast = {} # Accuracy for the last model
ILconstantBest = {} # Save the value of the IL constant
ILconstantLast = {} # Save the value of the IL constant
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nDataSplits
    costLast[thisModel] = [None] * nDataSplits
    ILconstantBest[thisModel] = [None] * nDataSplits
    ILconstantLast[thisModel] = [None] * nDataSplits

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    lossValid = {}
    costValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nDataSplits
        costTrain[thisModel] = [None] * nDataSplits
        lossValid[thisModel] = [None] * nDataSplits
        costValid[thisModel] = [None] * nDataSplits

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
        print("Loading data", end = '')
        if nDataSplits > 1:
            print(" for split %d" % (split+1), end = '')
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
        print("Setting up the graph...", end = ' ', flush = True)

    # Create graph
    adjacencyMatrix = data.getGraph()
    
    
    G1 = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                         {'adjacencyMatrix': adjacencyMatrix})
    G1.computeGFT() # Compute the GFT of the stored GSO

    genreMatrix = data.getGenreGraph()
    G2 = graphTools.Graph('adjacency', genreMatrix.shape[0],
                         {'adjacencyMatrix': adjacencyMatrix})
    G2.computeGFT() # Compute the GFT of the stored GSO

    #\\\ Ordering
    S1 = G1.S.copy()/np.max(np.real(G1.E))
    S2 = G2.S.copy()/np.max(np.real(G2.E))
    S = np.stack((S1, S2), axis = 0)

    # And re-update the number of nodes for changes in the graph (due to
    # enforced connectedness, for instance)
    nNodes = G1.N

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
        print("Model initialization...", flush = True)
        
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
    
    print("")
    
    for thisModel in modelsGNN.keys():
        
        if doPrint:
            print("Training model %s..." % thisModel)
            
        for m in modelList:
            if m in thisModel:
                modelName = m
    
        if nDataSplits > 1:
            trainingOptsPerModel[modelName]['graphNo'] = split
            
        thisTrainVars = modelsGNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[modelName])

        if doFigs:
        # Find which model to save the results (when having multiple
        # realizations)
            lossTrain[modelName][split] = thisTrainVars['lossTrain']
            costTrain[modelName][split] = thisTrainVars['costTrain']
            lossValid[modelName][split] = thisTrainVars['lossValid']
            costValid[modelName][split] = thisTrainVars['costValid']
            
                    
    # And we also need to save 'nBatch' but is the same for all models, so
    if doFigs:
        nBatches = thisTrainVars['nBatches']

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
        print("Total testing RMSE", end = '', flush = True)
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
        
        with torch.no_grad():
            # While the model has already been trained and everything relevant
            # computed, we still enfore toch.no_grad() just in case
            #   Be sure we're working on the best model
            modelsGNN[thisModel].load(label = 'Best')
            thisCbest = modelsGNN[thisModel].archit.ILconstant().item()
            #   Reload the last model
            modelsGNN[thisModel].load(label = 'Last')
            thisClast = modelsGNN[thisModel].archit.ILconstant().item()
        
        # Save values
        writeVarValues(varsFile,
                       {'costBest%s' % thisModel: thisCostBest,
                        'costLast%s' % thisModel: thisCostLast,
                        'ILconstantBest%s' % thisModel: thisCbest,
                        'ILconstantLast%s' % thisModel: thisClast})

        # Now check which is the model being trained
        costBest[modelName][split] = thisCostBest
        costLast[modelName][split] = thisCostLast
        ILconstantBest[modelName][split] = thisCbest
        ILconstantLast[modelName][split] = thisClast
        # This is so that we can later compute a total accuracy with
        # the corresponding error.
        
        if doPrint:
            print("\t%s: %.4f (C=%.4f) [Best] %.4f (C=%.4f) [Last]" % 
                  (thisModel,
                   thisCostBest, thisCbest,
                   thisCostLast, thisClast))
            
        ################
        # PLOT FILTERS #
        ################
        
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

if doFigs and doSaveVars:

    ###################
    # DATA PROCESSING #
    ###################
    
    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrain = {}
    meanCostTrain = {}
    meanLossValid = {}
    meanCostValid = {}
    stdDevLossTrain = {}
    stdDevCostTrain = {}
    stdDevLossValid = {}
    stdDevCostValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        costTrain[thisModel] = np.array(costTrain[thisModel])
        lossValid[thisModel] = np.array(lossValid[thisModel])
        costValid[thisModel] = np.array(costValid[thisModel])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis = 0)
        meanCostTrain[thisModel] = np.mean(costTrain[thisModel], axis = 0)
        meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis = 0)
        meanCostValid[thisModel] = np.mean(costValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0)
        stdDevCostTrain[thisModel] = np.std(costTrain[thisModel], axis = 0)
        stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis = 0)
        stdDevCostValid[thisModel] = np.std(costValid[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanCostTrain'] = meanCostTrain
    varsPickle['stdDevCostTrain'] = stdDevCostTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanCostValid'] = meanCostValid
    varsPickle['stdDevCostValid'] = stdDevCostValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
        
    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
            meanCostTrain[thisModel] = meanCostTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevCostTrain[thisModel] = stdDevCostTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanCostValid[thisModel] = meanCostValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevCostValid[thisModel] = stdDevCostValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ RMSE (Training and validation) for EACH MODEL
    for key in meanCostTrain.keys():
        costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanCostTrain[key], yerr = stdDevCostTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'RMSE')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        costFig.savefig(os.path.join(saveDirFigs,'cost%s.pdf' % key),
                        bbox_inches = 'tight')

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')

    # RMSE (validation) for ALL MODELS
    allCostValidFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanCostValid.keys():
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'RMSE')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanCostValid.keys()))
    allCostValidFig.savefig(os.path.join(saveDirFigs,'allCostValid.pdf'),
                    bbox_inches = 'tight')

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