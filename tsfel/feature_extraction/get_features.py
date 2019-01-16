import pandas as pd
import numpy as np
import pandas_profiling
#import src.feature_extraction.utils.read_json as rj
from TSFEL.tsfel.utils.read_json import feat_extract

def extract_features(sig, label, cfg, segment=True, window_size=5):
    feat_val = None
    labels = None
    features = []
    row_idx = []
    #header = np.array(pd.read_csv('TSFEL/tests/input_signal/Signal.txt', delimiter=',', header=None))[0, 1:]
    if segment:
        sig = [sig[i:i + window_size] for i in range(0, len(sig), window_size)]
    for wind_idx, wind_sig in enumerate(sig):
        if len(sig.shape) >= 3:
            for i in range(sig.shape[1]):
                _row_idx, labels = feat_extract(cfg, wind_sig[i], label)
                row_idx.append(_row_idx)
        else:
            row_idx, labels = feat_extract(cfg, wind_sig, label)
        if wind_idx == 0:
            feat_val = row_idx
        else:
            feat_val = np.vstack((feat_val, row_idx))
    feat_val = np.array(feat_val)
    d = {str(lab): feat_val[:,idx] for idx, lab in enumerate(labels)}
    df = pd.DataFrame(data=d)
    df.to_csv('TSFEL/tsfel/utils/Features.csv', sep=',', encoding='utf-8', index_label="Sample")

    return df

def correlation_report(df):
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="CorrelationReport.html")
    inp = str(input('Do you wish to remove correlated features? Enter y/n: '))
    if inp == 'y':
        reject = profile.get_rejected_variables(threshold=0.9)
        if reject == []:
            print('No features to remove')
        for rej in reject:
            print('Removing ' + str(rej))
            df = df.drop(rej, axis=1)
    return df

def FSE(X_train, X_test, y_train, y_test, labs, classifier):
    #classifier is the classifier to use
    #X_train and X_test are the features values
    #y_train and y_test are the labels associated to each X_train and X_test
    #labs is the set of features retrieved from google sheets
    total_acc, FS_lab, acc_list = [], [], []
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    for feat_idx, feat_name in enumerate(labs):
        classifier.fit(X_train[:,feat_idx].reshape(-1,1), y_train)
        y_test_predict = classifier.predict(X_test[:,feat_idx].reshape(-1,1))
        acc_list.append(accuracy_score(y_test, y_test_predict))

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(labs[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    FS_X_train = X_train[:,curr_acc_idx]
    FS_X_test = X_test[:,curr_acc_idx]
    total_acc.append(last_acc)

    counter = 1
    while 1:
        acc_list = []
        for feat_idx, feat_name in enumerate(labs):
            if feat_name not in FS_lab:
                curr_train = np.column_stack((FS_X_train, X_train[:, feat_idx]))
                curr_test = np.column_stack((FS_X_test, X_test[:, feat_idx]))
                classifier.fit(curr_train, y_train)
                y_test_predict = classifier.predict(curr_test)
                acc_list.append(accuracy_score(y_test, y_test_predict))
            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc < acc_list[curr_acc_idx]:
            FS_lab.append(labs[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)

            FS_X_train = np.column_stack((FS_X_train, X_train[:, curr_acc_idx]))
            FS_X_test = np.column_stack((FS_X_test, X_test[:, curr_acc_idx]))
        else:
            print("FINAL Features: " + str(FS_lab))
            print("Number of features", len(FS_lab))
            print("Acc: ", str(total_acc))
            print("From ", str(len(X_train[0])), "to ", str(len(FS_lab)))

            break
        counter += 1
        print(counter)
    return np.array(FS_X_train), np.array(FS_X_test), np.array(FS_lab)
