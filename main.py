import pandas as pd

from sklearn import tree, metrics, neighbors, ensemble, svm, linear_model
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


# k-номер человека от 1 до 40
def get_train_and_test(df, k):
    te = data[data['Subject_index'] == k]
    tr = data[data['Subject_index'] != k]
    return tr, te


def fit_and_evaluate(clf, inner_train, inner_test, inner_features):
    clf.fit(inner_train[inner_features], inner_train["Activity"])

    predictions = clf.predict(inner_test[inner_features])

    # c_matrix = metrics.confusion_matrix(inner_test["Activity"], predictions)
    score = metrics.accuracy_score(inner_test["Activity"], predictions)
    # report = metrics.classification_report(inner_test["Activity"], predictions)
    return score


labels = []
with open('sensors-19-05524-s001_new/labels.txt') as labels_file:
    for line in labels_file:
        labels.append(line[:-1])

data = pd.read_csv('sensors-19-05524-s001_new/data.txt', header=None, names=labels)

data['id_block'] = (data['Subject_index'] - 1) * 4 + data['Activity'] - 1
# test_final = data[data['id_block'] > 140]
# data = data[data['id_block'] <= 140]


features1 = [
    "ECG_original_mean",
    "IT_Original_mean_time",
    "IT_PSD_mean_freq",
    "EDA_Original_mean_arm",
    "EDA_Original_mean_hand"
]

features = [
    "IT_BR_mean_time",
    "IT_BRV_min_time",
    "IT_HF_LF_freq",
    "IT_HF_TF_freq",
    "IT_LF_MF_HF_freq",
    "ECG_amplitude_RR_geomean(abs)",
    "ECG_Logstd",
    "ECG_hrv_geomean(abs)",
    "EDA_p_samples_hand",
    "ECG_original_skewness",
    "ECG_amplitude_RR_mean",
    "ECG_HR_min_div_mean",
    "ECG_original_baseline",
    "ECG_RR_window_baseline",
    "ECG_amplitude_RR_baseline",
    "ECG_HR_min_div_baseline",
    "ECG_hrv_baseline",
    "ECG_PSD_baseline",
    "ECG_p_VFL_baseline",
    "ECG_p_LF_baseline",
    "ECG_p_MF_baseline",
    "ECG_p_HF_baseline",
    "ECG_p_total_LF_baseline",
    "IT_Original_baseline_time",
    "IT_LF_baseline_time",
    "IT_RF_baseline_time",
    "IT_BRV_baseline_time",
    "IT_PSD_baseline_freq",
    "IT_VLF_baseline_freq",
    "IT_LF_baseline_freq",
    "IT_MF_baseline_freq",
    "IT_HF_baseline_freq",
    "IT_p_Total_baseline_freq",
    "EDA_Original_baseline_arm",
    "EDA_processed_baseline_arm",
    "EDA_Filt1_baseline_arm",
    "EDA_Filt2_baseline_arm",
    "EDA_Functionals_power_Originalbaseline_arm",
    "EDA_Functionals_power_Fil12baseline_arm",
    "EDA_Functionals_power_Filt2baseline_arm",
    "EDA_Original_baseline_hand",
    "EDA_processed_baseline_hand",
    "EDA_Filt1_baseline_hand",
    "EDA_Filt2_baseline_hand",
    "EDA_Functionals_power_Originalbaseline_hand",
    "EDA_Functionals_power_Fil12baseline_hand",
    "EDA_Functionals_power_Filt2baseline_hand",
]


clfs = [
    # tree.DecisionTreeClassifier(criterion="entropy"),
    # tree.DecisionTreeClassifier(criterion="gini"),
    # neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance'),
    # neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance'),
    # neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance'),
    # ensemble.RandomForestClassifier(criterion="entropy"),
    # ensemble.RandomForestClassifier(criterion="gini"),
    # linear_model.SGDClassifier(penalty='elasticnet', l1_ratio=0.7),
    # linear_model.SGDClassifier(penalty='elasticnet', l1_ratio=0.7, loss='log'),
    # linear_model.SGDClassifier(penalty='elasticnet', l1_ratio=0.7, loss='modified_huber'),
    # linear_model.SGDClassifier(penalty='elasticnet', l1_ratio=0.7, loss='squared_hinge'),
    # linear_model.SGDClassifier(penalty='elasticnet', l1_ratio=0.7, loss='perceptron'),
    # xgb.XGBClassifier(verbosity=0),
    # lgb.LGBMClassifier(),
    # CatBoostClassifier(verbose=False, loss_function='MultiClass', langevin=True),
    # CatBoostClassifier(verbose=False, loss_function='MultiClass'),
    CatBoostClassifier(verbose=False, langevin=True),
    CatBoostClassifier(verbose=False),
    CatBoostClassifier(verbose=False, learning_rate=0.001, loss_function='MultiClass', langevin=True),
]

for clf in clfs:
    sc = 0
    runs = 0
    for idx in range(40):
        # print(f"Run for person {idx + 1} for classifier {clf}")
        train, test = get_train_and_test(data, idx + 1)
        score = fit_and_evaluate(clf, train, test, features)
        # print(f"Got score {score}")
        sc += score
        runs += 1
    sc /= runs
    print(f"Score for classifier {clf} is {sc}")
    print("----------------")

