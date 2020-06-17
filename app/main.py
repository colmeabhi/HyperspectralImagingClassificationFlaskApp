from flask import render_template, request, flash
from scipy.io import loadmat
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os,sys


def model(dataset_file, gt_file):

    dataset = dataset_file[ 'indian_pines_corrected']
    gt = gt_file['indian_pines_gt']

    X = np.reshape(dataset, (21025,200)) # your way gives good accuracy
    y = gt.reshape(145*145,1)

    # Normalisation of data

    normalized_X =  preprocessing.normalize(X)

    # Remove the rows with 0 gt values

    zero_results_indexes = []
    for i in range(len(y)):
        if(y[i] == 0):
            zero_results_indexes.append(i)

    y_del_zero, X_del_zero = np.delete(y, zero_results_indexes), np.delete(normalized_X, zero_results_indexes, axis = 0)
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_del_zero, y_del_zero, test_size=0.3, random_state=3)

    return X_trainset, y_trainset, X_testset, y_testset



# Functions
def dataset_load():
    path_df = os.path.join(sys.path[0], "Indian_pines_corrected.mat")
    path_gtf = os.path.join(sys.path[0], "Indian_pines_gt.mat")
    dataset_file = loadmat(path_df)
    gt_file = loadmat(path_gtf)
    return dataset_file, gt_file



def model_DT(X_trainset, y_trainset, X_testset, y_testset):
    # Prediction Using decision tree Algo

    Clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

    # print(Clf_dt) # it shows the default parameters

    Clf_dt.fit(X_trainset,y_trainset)
    predTree = Clf_dt.predict(X_testset)

    # Metrics and Accuracy
    # print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
    accuDT = metrics.accuracy_score(y_testset, predTree)
    return ('The DecisionTrees Accuracy is : '+str(accuDT))



def model_MLP(X_trainset, y_trainset, X_testset, y_testset):
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_trainset)
    X_test = scalar.transform(X_testset)

    clf_mlp =  MLPClassifier(hidden_layer_sizes=(150, 150),
                          activation='relu',
                          solver='adam',
                          alpha=0.0001,
                          batch_size='auto',
                          learning_rate='constant',
                          learning_rate_init=0.001,
                          max_iter=200,
                          shuffle=True,
                          random_state=1,
                          n_iter_no_change=50).fit(X_train, y_trainset)

    # Accuracy of the Mlp Classification
    accuMLP = clf_mlp.score(X_test, y_testset)
    return ('The MLP Accuracy is : '+str(accuMLP))



def model_SVM(X_trainset, y_trainset, X_testset, y_testset):
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_trainset)
    X_test = scalar.transform(X_testset)
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_trainset)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuSVM = svm_model_linear.score(X_test, y_testset)
    return ('The SVM accuracy is : '+str(accuSVM))
    # creating a confusion matrix
    # cm = confusion_matrix(y_testset, svm_predictions)





# Flask Routes

from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results_MLP', methods = ["POST"])
def result_MLP():
    if request.method == 'POST':
        dataset_file, gt_file = dataset_load()
        X_trainset, y_trainset, X_testset, y_testset = model(dataset_file, gt_file)
        accuMLP = model_MLP(X_trainset, y_trainset, X_testset, y_testset)
        return render_template("results.html", accuMLP = accuMLP)

@app.route('/results_DT', methods = ["POST"])
def result_DT():
    if request.method == 'POST':
        dataset_file, gt_file = dataset_load()
        X_trainset, y_trainset, X_testset, y_testset = model(dataset_file, gt_file)
        accuDT = model_DT(X_trainset, y_trainset, X_testset, y_testset)
        return render_template("results.html", accuDT = accuDT)

@app.route('/results_SVM', methods = ["POST"])
def result_SVM():
    if request.method == 'POST':
        dataset_file, gt_file = dataset_load()
        X_trainset, y_trainset, X_testset, y_testset = model(dataset_file, gt_file)
        accuSVM = model_SVM(X_trainset, y_trainset, X_testset, y_testset)
        return render_template("results.html", accuSVM = accuSVM)

@app.route('/uploaddataset', methods = ['POST'])
def uploaddataset():
    if request.method == 'POST':
        dataset_load()
        return render_template('results.html', ld = "Success ! the dataset is loaded")


# @def.route('/upload')
# def img():
#     if request.method = 'POST'
#         f = request.files['file']
#         f.save(secure_filename(f.filename))
#         img = cv2.imread(f)   # reads an image in the BGR format
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
