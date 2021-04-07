import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import hog
import math
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

data_path = '101_ObjectCategories'


def GetDefaultParameters():
    """
    A function that returns all the necessary parameter for our project
    :return: A dictionary containing the default experiment parameters
    """
    global data_path
    params = {"Phase": "TuningPhase",  # "TuningPhase" # "ModelTrainPhase" #"Assignment"
              "Fold": {"1": np.arange(21, 31), "2": np.arange(41, 51)}, "3": [],
              "getData": {"img_size": (64, 64), "report_file_path": "Results"},
              "Split": 0.5,
              "Tuning": {"fold": 5,
                         "parameters": [{'kernel': ['linear'],
                                         'C': [0.01, 0.1, 1, 10, 100]},
                                        {'kernel': ['poly'],
                                         'C': [0.01, 0.1, 1, 10, 100],
                                         'degree': [2, 3, 4, 5, 6]}, {'kernel': ['rbf'],
                                                                      'C': np.logspace(-2, 10, 13),
                                                                      'gamma': np.logspace(-9, 3, 13)}]},
              "Train": {"C": 0.1, "degree": 3, "kernel": 'poly'},
              "Prepare": {
                  "Hog": {"orientations": 10, "pixels_per_cell": (8, 8), "cells_per_block": (4, 4),
                          "feature_vector": False, "multichannel": False}},
              "Summary": {},
              "Report": {}
              }

    return params


def get_results_pickles(result_folder_path):
    """
    Get the path for the future pickle file in the ReportResults func.
    Creates a list with all files in the Results folder and finds the name with the lowest index.
    After having the name it creates a path to this name.
    If folder Result does not exists - it creates it and assigns 'ResultsOfExp_00.pkl' as file name.
    If it fails to create Results folder - no file path returned and no pickle will be created in ReportResults func.
    :param result_folder_path: A string of path to Results folder
    :return: A string of the path to the future pickle file.
            In case of OS failure returned "OSError"
    """
    file_name = 'ResultsOfExp_00.pkl'

    if os.path.exists(result_folder_path):

        result_pickles = os.listdir(result_folder_path)

        for i in np.arange(100):
            if 'ResultsOfExp_{:02d}.pkl'.format(i) not in result_pickles:
                file_name = 'ResultsOfExp_{:02d}.pkl'.format(i)
                break
    else:
        try:
            os.mkdir(result_folder_path)
        except OSError:
            return "OSError"

    return os.path.join(result_folder_path, file_name)


def GetData(params):
    """
    Gets default parameters and loads data from memory
        Phase1- in case of tuning. nums of classes directed at 21-30 and training function aimed at TrainWithTuning.
                After TrainWithTuning the program ends.
        Phase2- in case of Model training. nums of classes directed at 41-50 and training function aimed at Train
        Phase3- in case of tuning. nums of classes directed at main.class_indices and training function aimed at Train
        test_set_list - list with tuples. Each Tuple contains
                                (class path, class name, list of the names of the test-set pics, label of class)
                            necessary for finding 2 worst pic prediction of each class.
        result_folder_path - path to Results folder. Necessary for finding path for pickle file to save
                            at the ReportResult func.
        classes_info - dict {class_name:{ "label":label, "train":train-set len, "test": test-set len}}.
                        necessary for print info in the ReportResults func.
    Gets pics from memory, transfers to grayscale and reshapes to (S,S) - (64,64) in our case
    :param params: Parameter for the data loading and other functionality farther in the program
    :return: Dictionary {Data: list of N pics, each in the size as defined in the GetDefaultParameter func
                                - in our case (128,128),
                         Labels: list of N labels of the pics from Data}
    """
    # get classes indices depending on the phase
    if params["Phase"] == "TuningPhase":
        fold = params["Fold"]["1"]
    elif params["Phase"] == "ModelTrainPhase":
        fold = params["Fold"]["2"]
    elif params["Phase"] == "Assignment":
        fold = params["Fold"]["3"]
    else:
        fold = params["Fold"]["1"]

    img_size = params["getData"]["img_size"]
    split = params["Split"]
    result_folder_path = params["getData"]["report_file_path"]

    classes_names = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])
    classes_names = [classes_names[class_index - 1] for class_index in fold]

    Data = []
    Labels = []

    test_set_list = []
    classes_info = {}

    for class_n, label in zip(classes_names, fold):
        class_p = os.path.join(data_path, class_n)
        class_images = sorted(os.listdir(class_p))[:50]

        test_set_len = math.floor(len(class_images) * split)
        train_set_len = len(class_images) - test_set_len
        classes_info[class_n] = {"label": label, "train": train_set_len, "test": test_set_len}

        test_set_list.append((class_p, class_n, class_images[-test_set_len:], label))

        for img_n in class_images:
            img_p = os.path.join(class_p, img_n)
            img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, img_size)
            Data.append(img)
            Labels.append(label)

    DandL = {"Data": Data, "Labels": Labels}
    params["Summary"] = {"test_set_list": test_set_list}
    params["Report"]["pickle_file_path"] = get_results_pickles(result_folder_path)
    params["Report"]["classes_info"] = classes_info

    return DandL


def TrainTestSplit(data, labels, split):
    """
    :param data: list of N pics, (S,S) shape each - (64,64) shape in our case.
    :param labels: list of N the labels
    :param split: float in 0-1 scale. defines the split between the Test-set and Train-set.
            The higher - the bigger the Train-set.
    :return: A dictionary with two fields - Train and Test. each contains its data and labels.
    """
    TrainData, TestData, TrainLabels, TestLabels = [], [], [], []
    curr_index = 0
    unique_labels, counts = np.unique(np.array(labels), return_counts=True)

    for unique_label, count in zip(unique_labels, counts):
        start = curr_index
        end = curr_index + count
        separation = curr_index + math.ceil(count * split)
        TrainData.extend(data[start:separation])
        TrainLabels.extend(labels[start:separation])
        TestData.extend(data[separation:end])
        TestLabels.extend(labels[separation:end])
        curr_index += count

    SplitData = {"Train": {"Data": TrainData, "Labels": TrainLabels},
                 "Test": {"Data": TestData, "Labels": TestLabels}}
    return SplitData


def Prepare(data, params):
    """
    Gets a list of pics and transfers the into Hog description.
        bins - int that defines the amount of bins for each cell - 10 in our case
        pixels_per_cell - tuple that defines the shape of each cell(in pixels) =  (8,8) in our case
        cells_per_block - tuple that defines the shape of each block(in cells) =  (4,4) in our case
    :param data: list of images in grayscale - (64,64) each
    :param params: parameters for HOG algorithm
    :return: a list of vectorized pics in HOG representation - 4000 features each
    """

    bins = params["Hog"]["orientations"]
    pixels_per_cell = params["Hog"]["pixels_per_cell"]
    cells_per_block = params["Hog"]["cells_per_block"]

    prepared_data = [np.concatenate(hog(img,
                                        orientations=bins,
                                        pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block,
                                        feature_vector=params["Hog"]["feature_vector"],
                                        multichannel=params["Hog"]["multichannel"])).ravel()
                     for img in data]

    return prepared_data


def Train(TrainDataRep, labels, params):
    """
    Gets a Train-set of pics and uses parameters to train the SVM model on the Data.
        C - int. Regularization parameter.
        kernel - string. Specifies the kernel type to be used in the algorithm.
        gamma - Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        degree  - Degree of the polynomial kernel function. Only for 'poly' kernel.
    :param TrainDataRep: list of pics to be trained by the model
    :param labels: list of the labels of the data
    :param params: parameters for the SVM model
    :return: a trained SVM model
    """
    kernel = params["kernel"]
    C = params["C"]
    degree = params["degree"]

    clf = SVC(kernel=kernel, C=C, degree=degree)
    return clf.fit(TrainDataRep, labels)


def plot_tuning_heatmap(means, params):
    """
    Plots the parameters tuning heat map of the polynomial/rbf SVM
    :param means: list of mean results of the parameter tuning - shape depends on the kernel
    :param params: dict{ dicts- each {"C":C of the current tuning iteration,
                                        if "kernel"='poly': "degree": degree of the current iteration,
                                        if "kernel"='rbf': "degree": degree of the current iteration,
                                        "kernel": kernel of current iteration- in our case 'rbf' or 'poly'}}
    """
    param_names = list(params[0])
    first_params = np.unique([params[i][param_names[0]] for i in np.arange(len(params))])
    second_params = np.unique([params[i][param_names[1]] for i in np.arange(len(params))])
    means = np.array(means).reshape(len(first_params), len(second_params))
    dark_threshold = means.min() + (means.max() - means.min()) / 3
    if len(first_params) > 5:
        plt.figure(figsize=(12, 10))

    else:
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(means, interpolation='nearest', cmap=plt.cm.hot, aspect='auto')
    plt.xlabel(param_names[0])
    if len(first_params) > 5:
        plt.xticks(rotation=45)
    plt.ylabel(param_names[1])
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Error', rotation=270)
    plt.xticks(np.arange(len(first_params)), first_params)
    plt.yticks(np.arange(len(second_params)), second_params)
    plt.title("{} SVM - {} & {} tuning".format(params[0][param_names[2]].upper(), param_names[0], param_names[1]))
    for i in range(len(first_params)):
        for j in range(len(second_params)):
            if means[i, j] > dark_threshold:
                plt.text(j, i, '{:.3f}'.format(means[i, j]), ha="center", va="center")
            else:
                plt.text(j, i, '{:.3f}'.format(means[i, j]), ha="center", va="center", color="w")
    plt.tight_layout()
    plt.show()


def plot_tuning_graph(means, params):
    """
    Plots the parameters tuning graph of the linear SVM
    :param means: list of 5 mean results of the parameter tuning
    :param params: dict{ 5 dicts- each {"C":C of the current tuning iteration,
                                        "kernel": kernel of current iteration- in our case always 'linear'}}
    """
    Cs = np.unique([str(float(params[i]["C"])) for i in np.arange(len(params))])
    means = [round(mean, 3) for mean in means]
    plt.plot(Cs, means)
    plt.ylabel("Error")
    plt.xlabel("C")
    plt.xticks(np.arange(len(Cs)), Cs)
    plt.yticks(means, means)
    plt.title("Linear SVM - C tuning")
    plt.show()


def plot_validation_graph(best_classifiers, split_lens):
    """
    Plots the graph of fold X accuracy, best parameters, mean val. score of the classifier
    and size of the validation sets
    :param best_classifiers: dict{"mean":mean score of the best classifier and it's parameters,
                                   "validation_scores": list of 5 scores of each val. set
                                   "grid_params": parameters of the best classifier found}
    :param split_lens: list of the size of each fold
    """
    mean = best_classifiers["mean"]
    validation_scores = best_classifiers["validation_scores"]
    grid_params = best_classifiers["grid_params"]

    plt.plot(np.arange(len(split_lens)), validation_scores)
    plt.ylabel("Error")
    plt.xlabel("Fold")
    plt.xticks(np.arange(len(split_lens)), np.arange(len(split_lens)))
    plt.yticks(validation_scores, ['{:.2f}'.format(score) for score in validation_scores])
    plt.title(
        "Validation set error curve\n{}\nMean error:{:.3f}   Val. sets:{}".format(grid_params, mean,
                                                                                  split_lens))
    plt.show()


def get_kernel_best_classifier_data(cv_results_, folds):
    """
    Prepares data about the best classifier from each kernel.
    Neccesary for plotting the k-fold accuracy graph
    :param cv_results_: data object that returns from GridSearchCV func
    :param folds: num of k in k-folds
    :return: dict{"mean": best classifier mean error,
            "validation_scores": best classifier validation set error for each fold,
            "grid_params": best_classifier_params}
    """
    cv_results_rank = cv_results_['rank_test_score']
    best_classifier_index = np.where(cv_results_rank == 1)[0][0]
    best_classifier_mean = 1 - cv_results_['mean_test_score'][best_classifier_index]
    best_classifier_validation_scores = [1 - cv_results_[f'split{i}_test_score'][best_classifier_index]
                                         for i in np.arange(folds)]
    best_classifier_params = cv_results_['params'][best_classifier_index]
    return {"mean": best_classifier_mean,
            "validation_scores": best_classifier_validation_scores,
            "grid_params": best_classifier_params}


### Tuning
def TrainWithTuning(TrainDataRep, labels, params_tuning):
    """
    Searches the best combination of parameters-tuning. Prints the parameters found and the best score,
    Plots the heatmap of poly-svm classifier, graph of linear-svm and the val-set of the best classifier found
    :param TrainDataRep: list of pics(HOGed & vectorised)
    :param labels: list of ints. The labels of the TrainDataRep.
    :param params_tuning: parameters to be tuned by the func
    """
    folds = params_tuning["fold"]
    kernels = [params_tuning["parameters"][i]['kernel'][0] for i in np.arange(len(params_tuning["parameters"]))]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    cv_split = skf.split(TrainDataRep, labels)
    split_lens = [len(test_indices) for train_indices, test_indices in cv_split]

    best_classifiers = {}
    for kernel, i in zip(kernels, np.arange(len(kernels))):
        print(f"Current kernel:{kernel}")
        print("Tuning hyper-parameters for accuracy")

        grid_clf = GridSearchCV((SVC(decision_function_shape='ovr')), scoring='accuracy',
                                param_grid=params_tuning["parameters"][i],
                                cv=skf, n_jobs=-1)
        grid_clf.fit(TrainDataRep, labels)

        means = [1 - mean for mean in grid_clf.cv_results_['mean_test_score']]

        grid_params = grid_clf.cv_results_['params']

        best_classifiers[kernel] = get_kernel_best_classifier_data(grid_clf.cv_results_, folds)

        print("Error     Parameters")
        for mean, curr_params in zip(means, grid_params):
            print("{:.3f} for {}".format(1 - mean, curr_params))
        print()

        if len(list(grid_params[0])) == 2:  # linear
            plot_tuning_graph(means, grid_params)
        elif len(list(grid_params[0])) == 3:  # poly/rbf
            plot_tuning_heatmap(means, grid_params)

    if best_classifiers['poly']["mean"] <= best_classifiers['linear']["mean"]:
        kernel = 'poly'
    else:
        kernel = 'linear'

    if best_classifiers['rbf']["mean"] <= best_classifiers[kernel]["mean"]:
        kernel = 'rbf'

    plot_validation_graph(best_classifiers[kernel], split_lens)


def Test(model, testDataRep):
    """
    :param model: A trained SVM model
    :param testDataRep: Data to be predicted by the model
    :return: dict - {"predict": list of the predictions of the testDataRep,
                    "scores": (N,K) matrix -(len(testDataRep),10) in our case -
                        10 class scores for each samples in testDataRep}
    """
    return {"predict": model.predict(testDataRep), "scores": model.decision_function(testDataRep)}


def get_largest_error_names(names, scores, true_index):
    """
    For each pic calculates margin of the score from the best false score, takes the worst two if negative.
        (negative - The best false classifier was more certain that the true classifier - SVM was wrong)
    :param true_index: i - The true index of the class
    :param names: A list of pics names of the i'th class
    :param scores: A list of scores - 10 scores for each pic
    :return: a list of max two tuples. In each tuple - (index of pic, name of pic, margin of the true score from
                                                                                                the best false score)
    """

    worst_pics = []
    true_class_score = [score[true_index] - max(score) for score in scores]
    idxes = np.argpartition(true_class_score, 2)
    for idx in idxes[:2]:
        if true_class_score[idx] < 0:
            worst_pics.append((idx, names[idx], true_class_score[idx]))
    return worst_pics


def Evaluate(results, labels, params):
    """
    Gets the predictions and true labels - calculates accuracy and confusion matrix
    Gets decisions for each pic and gets the 2 worst for each class.

    :param results: dict {"predict": list of prediction of the Test-set,
                           "scores": matrix of 10 class scores for each pic in the Test-set}
    :param labels: list of the labels of the Test-set
    :param params: {"test_set_list": list on 10 tuples.
                    In each(class path, class name, list of names of the Test-set pics in class, label of class)}
    :return: {"confusion_matrix": confusion_matrix of the Test-set,
            "error_rate": 1 - accuracy_score of the Test-set,
            "worst_pics": list of 10 classes list. in each class list- tuple of max 2 worst pics in class.
                            each tuple -  (index of pic, name of pic, margin of the true score from
                                                                                    the best false score)}
    """
    predict = results["predict"]
    scores = results["scores"]
    test_set_list = params["test_set_list"]

    unique_labels = np.unique(labels)

    worst_pics = {}
    for class_p, class_name, pics_names, pics_label in test_set_list:
        true_label_index = np.where(unique_labels == pics_label)[0][0]
        class_scores = [score for score, scores_label in zip(scores, labels) if pics_label == scores_label]
        worst_pics[class_name] = {"pics": get_largest_error_names(pics_names, class_scores, true_label_index),
                                  "path": class_p}

    return {"confusion_matrix": confusion_matrix(y_true=labels, y_pred=predict),
            "error_rate": 1 - accuracy_score(predict, labels), "worst_pics": worst_pics}


def plot_confusion_matrix(conf_matrix, classes_names):
    """
    :param conf_matrix: confusion matrix that was computed before
    :param classes_names: list of 10 classes names
    """

    dark_threshold = conf_matrix.min() + (conf_matrix.max() - conf_matrix.min()) / 3
    fig, ax = plt.subplots()
    ax.imshow(conf_matrix)
    ax.set_xticks(np.arange(len(classes_names)))
    ax.set_yticks(np.arange(len(classes_names)))
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(classes_names)):
        for j in range(len(classes_names)):
            if conf_matrix[i, j] > dark_threshold:
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center")
            else:
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
    error_rate = 1-(sum([conf_matrix[i, i] for i in range(conf_matrix.shape[0])]) / conf_matrix.sum())
    ax.set_title(f"Confusion Matrix\nError rate = {round(error_rate,3)}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    fig.tight_layout()
    plt.show()


def add_pic_plot(class_path, pic_name, pic_margin, class_subplot, start_position):
    """
    Adds a pic to parent Subplot
    :param class_path: path to pics of the class
    :param pic_name: name of the pic to add to the plot
    :param pic_margin: margin of the true classifier score from the best false class score
    :param class_subplot: parent subplot
    :param start_position: (i,j) - position of the bottom left corner in the parent subplot
    """
    fig = plt.gcf()
    box = class_subplot.get_position()
    height, width = box.height * 0.7, box.width * 0.45
    in_ax_position = class_subplot.transAxes.transform(start_position)
    transFigure = fig.transFigure.inverted()
    in_fig_position = transFigure.transform(in_ax_position)
    x = in_fig_position[0]
    y = in_fig_position[1]
    sub_ax = fig.add_axes([x, y, width, height])
    sub_ax.title.set_text("{}\nMargin:{:.2f}".format(pic_name, np.absolute(pic_margin)))
    sub_ax.title.set_fontsize(18)
    sub_ax.axis('off')
    img = cv2.imread(os.path.join(class_path, pic_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def plot_worst_pics(dict_worst_pics):
    """
    Plots a plot with the 2 worst predicted pics from each class(if any exists)
    :param dict_worst_pics:dict {class_name: dict{"pics": tuple(pic index,pic_name,pic margin)
                                                   "path":string of path to class pics folder}}
    """
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Two Pictures With The Worst Mistake From Each Class", fontsize=27)
    classes = list(dict_worst_pics)

    start_position = [[0, 0.1], [0.55, 0.1]]

    for c, i in zip(classes, np.arange(len(classes))):
        class_subplot = fig.add_subplot(5, 2, i + 1)
        class_subplot.title.set_text("{}".format(c))
        class_subplot.title.set_fontsize(20)
        class_subplot.axis('off')

        if len(dict_worst_pics[c]) > 0:
            path = dict_worst_pics[c]["path"]
            pics = dict_worst_pics[c]["pics"]
            for pic, j in zip(pics, np.arange(len(pics))):
                pic_name, pic_margin = pic[1], pic[2]
                add_pic_plot(path, pic_name, pic_margin, class_subplot, start_position[j])
        else:
            class_subplot.text(0.4, 0.5, "No Errors for this Class", fontsize=18)
    plt.show()


def save_result_to_pickle(object_to_save, pickle_file_path):
    """
    Saves result pickle file to Results folder. If there was an OS error in the get_results_pickles func
        the file will not be saved
    :param object_to_save:
    :param pickle_file_path:
    """
    if pickle_file_path != 'OSError':
        pickle_out = open(pickle_file_path, "wb")
        pickle.dump(object_to_save, pickle_out)
        pickle_out.close()


def print_classes_info(classes_info):
    """
    :param classes_info:dict{class_name:dict{"label":label of the class,
                                             "train": size of train-set of the class,
                                             "test": size of test-set of the class}
    """
    classes_names = list(classes_info)

    print("####Classes Report####")

    for class_n in classes_names:
        print("Class name: {}, Class label: {}, Train-data size: {}, Test-data size: {}".format(class_n,
                                                                                                classes_info[class_n][
                                                                                                    "label"],
                                                                                                classes_info[class_n][
                                                                                                    "train"],
                                                                                                classes_info[class_n][
                                                                                                    "test"]))


def ReportResults(summary, params):
    """
    Assembles results from experiment, prints them to screen and saves to memory
    :param summary: dict{"confusion_matrix":(10,10) shape,
                           "error_rate": float 0-1. Represents the portion of mistake of the prediction,
                           "worst_pics":dict {class_name: dict{"pics": tuple(pic index,pic_name,pic margin)
                                                   "path":string of path to class pics folder}}
    :param params: dict{"pickle_file_path":A string of the path to the future pickle file to be saved to memory,
                        "classes_info": dict{class_name:dict{"label":label of the class,
                                                              "train": size of train-set of the class,
                                                                "test:: size of test-set of the class}}
    """
    conf_matrix = summary["confusion_matrix"]
    error_rate = summary["error_rate"]
    worst_pics = summary["worst_pics"]
    classes_names = list(worst_pics)
    pickle_file_path = params["pickle_file_path"]
    classes_info = params["classes_info"]

    object_to_save = {"classes": classes_names,
                      "error_rate": error_rate,
                      "confusion_matrix": conf_matrix,
                      "classes_info": classes_info}

    print_classes_info(classes_info)
    print("Final Test Error Result: {}".format(error_rate))

    plot_confusion_matrix(conf_matrix, classes_names)
    plot_worst_pics(worst_pics)
    save_result_to_pickle(object_to_save, pickle_file_path)


def plot_pics_hog_tuning_heatmap():
    """
    function not in the pipe. After running on for loops on all reasonable combinations for picture and hog and
    committing tuning with TrainWithTuning function, the best classifiers from each combination were colected
    and put into an object - pic_hog_tuning. THe parameters of the best combination were chosen as picture(size)
    and Hog(pixels per cell, bins, cells per block). The for loops were removed from main function.
    """

    pic_hog_tuning = {
        "(64,64)": {"(8,8)": [[1 - 0.5142040816326531, 1 - 0.5862040816326531, 1 - 0.602204081632653],
                              [1 - 0.5423673469387755, 1 - 0.5901224489795919, 1 - 0.594204081632653],
                              [1 - 0.5502857142857144, 1 - 0.582204081632653, 1 - 0.610204081632653]],
                    "(16,16)": [[1 - 0.5221224489795919, 1 - 0.5540408163265307, 1 - 0.5139591836734694],
                                [1 - 0.5262040816326531, 1 - 0.5341224489795919, 1 - 0.5340408163265307],
                                [1 - 0.5221224489795919, 1 - 0.5582857142857144, 1 - 0.5220408163265307]]},
        "(128,128)": {"(8,8)": [[1 - 0.5101224489795919, 1 - 0.5183673469387755, 1 - 0.5424489795918367],
                                [1 - 0.5140408163265306, 1 - 0.5142040816326531, 1 - 0.5342857142857144],
                                [1 - 0.5180408163265307, 1 - 0.5302857142857142, 1 - 0.5543673469387755]],
                      "(16,16)": [[1 - 0.5583673469387755, 1 - 0.570204081632653, 1 - 0.5942857142857143],
                                  [1 - 0.5662857142857142, 1 - 0.5702857142857143, 1 - 0.5902857142857142],
                                  [1 - 0.5744489795918367, 1 - 0.5822040816326531, 1 - 0.6022857142857143]]}}

    pic_sizes = list(pic_hog_tuning)
    pixels_per_cell = list(pic_hog_tuning[pic_sizes[0]])
    bins = [8, 9, 10]
    cells_per_block = [(2, 2), (3, 3), (4, 4)]

    min_score = min(min([min(pic_hog_tuning[i][j]) for i in pic_sizes for j in pixels_per_cell]))
    max_score = max(max([max(pic_hog_tuning[i][j]) for i in pic_sizes for j in pixels_per_cell]))
    dark_threshold = min_score + (max_score - min_score) / 3

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Tuning Picture size, Pixels per cell, Bins and Cells per block", fontsize=27)
    helper = 1
    for size, i in zip(pic_sizes, np.arange(len(pic_sizes))):
        for pixl, j in zip(pixels_per_cell, np.arange(len(pixels_per_cell))):
            class_subplot = fig.add_subplot(2, 2, i + j + helper)
            class_subplot.title.set_text("Image size:{}, Pixels per Cell: {}".format(size, pixl))
            class_subplot.title.set_fontsize(10)
            plt.imshow(pic_hog_tuning[size][pixl], interpolation='nearest', cmap=plt.cm.hot, vmin=min_score,
                       vmax=max_score)
            plt.xlabel('Cells per block')
            plt.ylabel('Bins')
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 10
            cbar.ax.set_ylabel('Error', rotation=270)
            plt.xticks(np.arange(len(cells_per_block)), cells_per_block)
            plt.yticks(np.arange(len(bins)), bins)
            for k in range(3):
                for m in range(3):
                    if pic_hog_tuning[size][pixl][k][m] > dark_threshold:
                        plt.text(m, k, '{:.3f}'.format(pic_hog_tuning[size][pixl][k][m]), ha="center", va="center")
                    else:
                        plt.text(m, k, '{:.3f}'.format(pic_hog_tuning[size][pixl][k][m]), ha="center", va="center",
                                 color="w")
        helper = 2

    plt.show()


def main():
    class_indices = [0, 6, 10, 60, 65, 67, 81, 83, 86, 90]

    Params = GetDefaultParameters()
    Params["Fold"]["3"] = class_indices
    Params["Phase"] = "TuningPhase"  # tuning/training/assignment -># "TuningPhase" # "ModelTrainPhase" #"Assignment"
    np.random.seed(0)

    DandL = GetData(Params)
    SplitData = TrainTestSplit(DandL["Data"], DandL["Labels"], Params["Split"])
    TrainDataRep = Prepare(SplitData["Train"]["Data"], Params["Prepare"])
    if Params["Phase"] == "TuningPhase":
        TrainWithTuning(TrainDataRep, SplitData["Train"]["Labels"], Params["Tuning"])
    else:
        Model = Train(TrainDataRep, SplitData["Train"]["Labels"], Params["Train"])
        TestDataRep = Prepare(SplitData["Test"]["Data"], Params["Prepare"])
        Results = Test(Model, TestDataRep)
        Summary = Evaluate(Results, SplitData["Test"]["Labels"], Params["Summary"])
        ReportResults(Summary, Params["Report"])


main()
