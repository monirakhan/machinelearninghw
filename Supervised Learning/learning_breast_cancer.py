from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import export_graphviz
import pydotplus
from util import get_data, plot_data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(903226865)

def getInfo(dataFile, amount=0.6):
    inf = open(dataFile)
    data = np.genfromtxt(inf, delimiter=',')
    #data = data[1:, 1:]
    np.random.shuffle(data)
    train_rows = int(amount * data.shape[0])
    test_rows = data.shape[0] - train_rows
    # separate out training and testing data
    trainX = data[:train_rows, 1:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 1:-1]
    testY = data[train_rows:, -1]
    return trainX, trainY, testX, testY

def split_data(data, amount):
    rows = int(amount * data.shape[0])
    return data[:rows,]

#Decision Tree without pruning
def dtl_without_pruning():
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    training_sizes = [.01, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50, .55, .65, .70, .75, .80, .85, .90]
    insample = []
    outofsample = []
    dtl = DecisionTreeClassifier(max_depth=10, max_features=9)
    # dtl.fit(trainX, trainY)
    # dot_data = StringIO()
    # export_graphviz(dtl, out_file='dot_data_without_pruning')
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        dtl.fit(trainingX,trainingY)
        insample_score = dtl.score(trainingX, trainingY)
        outofsample_score = dtl.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "DTLearner Accuracy without Pruning (Breast Cancer Dataset)", "Training Size", "Score")
    plt.ylim(0.3,1.1)
    plt.savefig("Graphs/dtl_wo_pruning-breast-cancer.png")

#Decision Tree with pruning
def dtl():
    parameter_grid = {'max_depth': range(1, 8),
                      'max_features': range(1, 5)}
    #Mac depth, min samples leaf
    dtl = DecisionTreeClassifier()
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    grid_search = GridSearchCV(dtl, param_grid= parameter_grid,
        cv= 5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    #best.fit(trainX, trainY)
    # dot_data = StringIO()
    # export_graphviz(best, out_file='dot_data')
    # print(best.get_params())
    training_sizes = [.01, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50, .55, .65, .70, .75, .80, .85, .90]
    insample = []
    outofsample = []
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        best.fit(trainingX,trainingY)
        insample_score = best.score(trainingX, trainingY)
        outofsample_score = best.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "DTLearner Accuracy with Pruning(Breast Cancer Dataset)", "Training Size", "Score")
    plt.ylim(0.3,1.1)
    plt.savefig("Graphs/dtl_score-breast-cancer.png")
    #return dtl.predict(testX)
def dtl_max_depth():
    #dtl = DecisionTreeClassifier()
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    insample = []
    outofsample = []
    depth_index = range(1,20)
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
    plot_data(dataList, "DTLearner Accuracy vs Max Depth", "Max Depth", "Score")
    plt.ylim(.6, 1)
    plt.savefig("Graphs/dtl_max_depth-breast_cancer.png")
def boosted_dtl():
    classifer = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
        n_estimators=100)
    param_grid = {'n_estimators': range(1, 100, 10)}
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    grid_search = GridSearchCV(classifer, param_grid= param_grid,
        cv= 5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        best.fit(trainingX,trainingY)
        insample_score = best.score(trainingX, trainingY)
        outofsample_score = best.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Boosted DTLearner Accuracy", "Training Size", "Score")
    plt.ylim(0,1)
    plt.savefig("Graphs/boosted.png")

def boosted_dtl_estimators():
    insample = []
    outofsample = []
    estimators = []
    trainX, trainY, testX, testY = getInfo("Data/winequality-red.csv")
    for n_estimators in range(25):
        estminator = (n_estimators+1)*10
        estimators.append(estminator)
        insample_score, outofsample_score = boosted_decision_tree(trainX,trainY,
            testX,testY, n_estimators=estminator)
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataFrame = pd.DataFrame(index=estimators, columns=["In Sample", "Out of Sample"])
    insample_accuracy = pd.DataFrame(insample, index= estimators)
    outofsample_accuracy = pd.DataFrame(outofsample, index=estimators)
    dataFrame["In Sample"] = insample_accuracy
    dataFrame["Out of Sample"] = outofsample_accuracy
    plot_data(dataFrame, "Boosted DTL Accuracy vs # of Estmiators", "Estminators", "Accuracy")
    plt.savefig("Graphs/boosted_estimators.png")
    pass

def neural_network():
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10))
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    param_grid = {'hidden_layer_sizes': range(1, 200, 10),
        'activation': ['identity', 'logistic', 'tanh', 'relu']}
    grid_search = GridSearchCV(nn, param_grid=param_grid, cv=5)
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        best.fit(trainingX,trainingY)
        insample_score = best.score(trainingX, trainingY)
        outofsample_score = best.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "Neutral Networks Accuracy", "Training Size", "Score")
    plt.ylim(0,1)
    plt.savefig("Graphs/nn.png")
    print(best.get_params())

def SVC():
    svm_learner = svm.SVC(kernel='linear')
    param_grid = {'kernel': ['linear', 'sigmoid', 'rbf']}
    grid_search = GridSearchCV(svm_learner, param_grid=param_grid, cv=5)
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        best.fit(trainingX,trainingY)
        insample_score = best.score(trainingX, trainingY)
        outofsample_score = best.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "SVC Accuracy", "Training Size", "Score")
    plt.ylim(0.5,1)
    plt.savefig("Graphs/svc.png")
    print(best.get_params())

def knearest():
    knearest_learner = KNeighborsClassifier(n_neighbors=1)
    param_grid = {'n_neighbors': range(1,9)}
    grid_search = GridSearchCV(knearest_learner, param_grid=param_grid, cv=5)
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    grid_search.fit(trainX, trainY)
    best = grid_search.best_estimator_
    print(best.get_params())
    training_sizes = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
    insample = []
    outofsample = []
    for size in training_sizes:
        trainingX = split_data(trainX, size)
        trainingY = split_data(trainY, size)
        best.fit(trainingX,trainingY)
        insample_score = best.score(trainingX, trainingY)
        outofsample_score = best.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=training_sizes, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=training_sizes)
    outofsample_error = pd.DataFrame(outofsample, index=training_sizes)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "K Neighbors Accuracy", "Training Size", "Score")
    plt.ylim(0.5,1)
    plt.savefig("Graphs/knearest.png")

def knearest_vs_neighbors():
    neighbors = range(1, 101, 5)
    trainX, trainY, testX, testY = getInfo("Data/breast_cancer.csv")
    insample = []
    outofsample = []
    for x in neighbors:
        knearest_learner = KNeighborsClassifier(n_neighbors=x)
        knearest_learner.fit(trainX, trainY)
        insample_score = knearest_learner.score(trainX, trainY)
        outofsample_score = knearest_learner.score(testX,testY)
        print("Out of Sample Accuracy: %0.2f" % (outofsample_score))
        print("In Sample Accuraccy: %0.2f" % (insample_score))
        insample.append(insample_score)
        outofsample.append(outofsample_score)
    dataList = pd.DataFrame(index=neighbors, columns=["In Sample", "Out of Sample"])
    insample_error = pd.DataFrame(insample, index=neighbors)
    outofsample_error = pd.DataFrame(outofsample, index=neighbors)
    dataList["In Sample"] = insample_error
    dataList["Out of Sample"] = outofsample_error
    plot_data(dataList, "K Neighbors Accuracy", "Neighbors", "Score")
    plt.ylim(0.7,1)
    plt.savefig("Graphs/knearest_vs_neighbors.png")



if __name__ =="__main__":
    dtl_without_pruning()
    dtl()
    #dtl_max_depth()
    #boosted_dtl()
    #neural_network()
    #SVC()
    #knearest_vs_neighbors()

















