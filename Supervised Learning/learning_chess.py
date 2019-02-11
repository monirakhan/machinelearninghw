from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve
import pydotplus
from util import get_data, plot_data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(903226865)

def getInfo(dataFile="Data/chess.csv", amount=0.6):
    inf = open(dataFile)
    data = np.genfromtxt(inf, delimiter=',')
    #data = data[1:, 1:]
    np.random.shuffle(data)
    train_rows = int(amount * data.shape[0])
    test_rows = data.shape[0] - train_rows
    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]
    return trainX, trainY, testX, testY

def getXY(dataFile="Data/chess.csv"):
    inf = open(dataFile)
    data = np.genfromtxt(inf, delimiter=',')
    #data = data[1:, 1:]
    np.random.shuffle(data)
    # separate out training and testing data
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y

def split_data(data, amount):
    rows = int(amount * data.shape[0])
    return data[:rows,]
#Decision Tree without pruning
def dtl_without_pruning():
    X, y = getXY("Data/chess.csv")
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    dtl = DecisionTreeClassifier(max_depth=150, max_features=36)
    training_size, training_score, testing_score = learning_curve(dtl, X, y, train_sizes=training_sizes, cv=5)
    # export_graphviz(dtl, out_file='dot_data_chess_unpruned')
    # print(dtl.get_params())
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(training_score, index=training_sizes)
    outofsample_error = pd.DataFrame(testing_score, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "DTLearner Accuracy without Pruning (Chess Dataset)", "Training Size", "Score")
    plt.ylim(0.8,1.01)
    plt.savefig("Graphs/dtl_wo_pruning-chess.png")

#Decision Tree with pruning
def dtl():
    parameter_grid = {'max_depth': range(1, 11),
                      'max_features': range(1, 21)}
    #Mac depth, min samples leaf
    dtl = DecisionTreeClassifier()
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X, y = getXY("Data/chess.csv")
    grid_search = GridSearchCV(dtl, param_grid= parameter_grid,
        cv= 5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    #export_graphviz(best, out_file='dot_data_chess')
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    training_size, training_score, testing_score = learning_curve(best, X, y, train_sizes=training_sizes, cv=5)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(training_score, index=training_sizes)
    outofsample_error = pd.DataFrame(testing_score, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "DTLearner Accuracy with Pruning(Chess Game Dataset)", "Training Size", "Score")
    plt.ylim(0.8,1.01)
    plt.savefig("Graphs/dtl_score-chess.png")
    #return dtl.predict(testX)
def dtl_max_depth():
    #dtl = DecisionTreeClassifier()
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    insample = []
    outofsample = []
    depth_index = range(1,15)
    for depth in depth_index:
        dtl = DecisionTreeClassifier(max_depth=depth)
        dtl.fit(trainX, trainY)
        insample_score = dtl.score(trainX, trainY)
        outofsample_score = dtl.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=depth_index, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=depth_index)
    outofsample_error = pd.DataFrame(outofsample, index=depth_index)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "DTLearner Accuracy vs Max Depth(Chess Dataset)", "Max Depth", "Score")
    plt.ylim(.6, 1.01)
    plt.savefig("Graphs/dtl_max_depth-chess.png")


def unpruned_boosted_dtl():
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=150),
        n_estimators=150)
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X, y = getXY("Data/chess.csv")
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    training_size, insample, outofsample = learning_curve(classifier, X, y, train_sizes=training_sizes, cv=5)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Unpruned Boosted DTLearner Accuracy(Chess Dataset)", "Training Size", "Score")
    plt.ylim(0.8,1.01)
    plt.savefig("Graphs/unpruned_boosted_chess.png")


def boosted_dtl():
    classifer = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50)
    param_grid = {'n_estimators': range(1, 70, 10)}
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X, y = getXY("Data/chess.csv")
    grid_search = GridSearchCV(classifer, param_grid= param_grid,
        cv= 5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    training_size, insample, outofsample = learning_curve(best, X, y, train_sizes=training_sizes, cv=5)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Pruned Boosted DTLearner Accuracy(Chess Dataset)", "Training Size", "Score")
    plt.ylim(0.8,1.01)
    plt.savefig("Graphs/boosted_chess.png")

def boosted_dtl_estimators():
    insample = []
    outofsample = []
    estimators = []
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    for n_estimators in range(1,101, 5):
        estimators.append(n_estimators)
        classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators)
        classifier.fit(trainX, trainY)
        insample.append(classifier.score(trainX, trainY))
        outofsample.append(classifier.score(testX, testY))
    dataFrame = pd.DataFrame(index=estimators, columns=["In Sample", "Out of Sample"])
    insample_accuracy = pd.DataFrame(insample, index= estimators)
    outofsample_accuracy = pd.DataFrame(outofsample, index=estimators)
    dataFrame["In Sample"] = insample_accuracy
    dataFrame["Out of Sample"] = outofsample_accuracy
    plot_data(dataFrame, "Boosted DTL Accuracy vs # of Estmiators(Chess Dataset)", "Estminators", "Accuracy")
    plt.ylim(.9, 1.01)
    plt.savefig("Graphs/boosted_estimators-chess.png")
    pass

def neural_network():
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10))
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X, y = getXY()
    param_grid = {'hidden_layer_sizes': range(1, 100, 10),
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam']}
    grid_search = GridSearchCV(nn, param_grid=param_grid, cv=5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    best.fit(trainX, trainY)
    print(best.score(testX, testY))
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    training_size, insample, outofsample = learning_curve(best, X, y, train_sizes=training_sizes, cv=5)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Neutral Networks Accuracy(Chess Dataset)", "Training Size", "Score")
    plt.ylim(0.4,1.01)
    plt.savefig("Graphs/nn-chess-02.png")
    print(best.get_params())


def neural_network_layers():
    trainX, trainY, testX, testY = getInfo()
    insample = []
    outofsample = []
    for neurons in range(100):
        layers = []
        for neuron in range(neurons):
            layers.append(45)
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers))
        classifier.fit(trainX, trainY)
        insample.append(classifier.score(trainX, trainY))
        outofsample.append(classifier.score(testX, testY))
    dataList = pd.DataFrame(index=range(100), columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=range(100))
    outofsample_error = pd.DataFrame(outofsample, index=range(100))
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Neutral Networks Accuracy vs Numbers of Layers(Chess Dataset)", "Layers", "Score")
    plt.ylim(0.4,1.01)
    plt.savefig("Graphs/nn_vs_layers-chess.png")

def SVC():
    svm_learner = svm.SVC(kernel='linear')
    param_grid = {'kernel': ['linear', 'sigmoid', 'rbf']}
    grid_search = GridSearchCV(svm_learner, param_grid=param_grid, cv=5)
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X,y = getXY()
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    best.fit(trainX, trainY)
    print(best.score(testX, testY))
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    print("HERE")
    training_size, insample, outofsample = learning_curve(best, X, y, train_sizes=training_sizes, cv=5)
    print("DONE")
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "SVC Accuracy(Chess Dataset, rbf kernel)", "Training Size", "Score")
    plt.ylim(0.4,1.01)
    plt.savefig("Graphs/svc-chess.png")
    print(best.get_params())

def SVC_compared():
    svm_learner = svm.SVC(kernel='sigmoid')
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X,y = getXY()
    print(svm_learner.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    print("HERE")
    training_size, insample, outofsample = learning_curve(svm_learner, X, y, train_sizes=training_sizes, cv=5)
    print("DONE")
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "SVC Accuracy(Chess Dataset, sigmoid kernel)", "Training Size", "Score")
    plt.ylim(0.4,1.01)
    plt.savefig("Graphs/svc-chess-different-kernel.png")

def knearest():
    knearest_learner = KNeighborsClassifier(n_neighbors=1)
    param_grid = {'n_neighbors': range(1,50)}
    grid_search = GridSearchCV(knearest_learner, param_grid=param_grid, cv=5)
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    X, y = getXY()
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    training_size, insample, outofsample = learning_curve(best, X, y, train_sizes=training_sizes, cv=5)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "K Neighbors Accuracy(Chess Dataset, 7 Neighbors)", "Training Size", "Score")
    plt.ylim(0.5,1.01)
    plt.savefig("Graphs/knearest-chess.png")

def knearest_vs_neighbors():
    neighbors = range(1, 101, 5)
    trainX, trainY, testX, testY = getInfo("Data/chess.csv")
    insample = []
    outofsample = []
    for x in neighbors:
        knearest_learner = KNeighborsClassifier(n_neighbors=x)
        knearest_learner.fit(trainX, trainY)
        insample_score = knearest_learner.score(trainX, trainY)
        outofsample_score = knearest_learner.score(testX,testY)
        # print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        # print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=neighbors, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=neighbors)
    outofsample_error = pd.DataFrame(outofsample, index=neighbors)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "K Neighbors Accuracy(Chess Dataset)", "Neighbors", "Score")
    plt.ylim(0.7,1.01)
    plt.savefig("Graphs/knearest_vs_neighbors-chess.png")



if __name__ =="__main__":
    #dtl_without_pruning()
    #dtl()

    #dtl_max_depth()
    #unpruned_boosted_dtl()
    #boosted_dtl()
    #boosted_dtl_estimators()
    neural_network()
    #neural_network_neurons()
    #neural_network_layers()
    #SVC()
    #SVC_compared()
    #knearest()
    #knearest_vs_neighbors()





