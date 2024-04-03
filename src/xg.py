from pathlib import Path
import xgboost as xgb
import numpy as np
from time import time
import json
from time import time
import collections

from utils.plots import plot_roc
from sklearn.model_selection import RandomizedSearchCV

from utils.utils import calc_all, NumpyEncoder
from utils.plots import plot_confusion_matrix


def xg_main(train, test, trials, save='xg'):
    # Runs XGBoost model with hyperparameteroptimization
    # Evaluation on csecicids2018
    name = '../experiments/' + save + '/best/'
    Path(name).mkdir(parents=True, exist_ok=True)

    params = {
        'num_rounds': 100,
        'max_depth': 8,
        'max_leaves': 2**8,
        'alpha': 0.9,
        'eta': 0.1,
        'gamma': 0.1,
        'subsample': 1,
        'reg_lambda': 1,
        'scale_pos_weight': 2,
        'objective': 'binary:logistic',
        'verbose': True,
        'gpu_id': 0,
        'tree_method': 'gpu_hist'
    }
    hyperparameter_grid = {
        'max_depth': [3, 6, 9],
        'eta': list(np.linspace(0.1, 0.6, 6)),
        'gamma': [int(x) for x in np.linspace(0, 10, 10)]
    }

    bst = xgb.XGBClassifier(**params)
    clf = RandomizedSearchCV(
        bst,
        hyperparameter_grid,
        random_state=0,
        n_iter=trials)

    start = time()
    model = clf.fit(train.x, train.y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (
        time() - start, len(clf.cv_results_["params"])))
    print(model.best_params_)

    metrics_train, cm_train, cm_norm_train, preds_train = calc_all(
        model, train)
    plot_confusion_matrix(cm_train, savefile=name + 'cm_train.pdf', name=save)
    plot_confusion_matrix(
        cm_norm_train,
        savefile=name +
        'cm_normalized_train.pdf',
        name=save)
    plot_roc(
        metrics_train['TPR'],
        metrics_train['FPR'],
        metrics_train['AUC'],
        name + save + '_roc_train.pdf',
        name=save)

    metrics, cm, cm_norm, preds = calc_all(model, test)
    plot_confusion_matrix(cm, savefile=name + 'cm.pdf', name=save)
    plot_confusion_matrix(
        cm_norm,
        savefile=name +
        'cm_normalized.pdf',
        name=save)
    plot_roc(
        metrics['TPR'],
        metrics['FPR'],
        metrics['AUC'],
        name +
        save +
        '_roc.pdf',
        name=save)

    normals = collections.Counter(train.y)[0]
    anomalies = collections.Counter(train.y)[1]
    anomalies_percentage = anomalies / (normals + anomalies)
    results = {
        'Anomalies Percentage': anomalies_percentage,
        'Metrics train': metrics_train,
        'Metrics test': metrics,
        'Best hyperparameters': model.best_params_
    }

    numpy_preds = {
        'Preds test': preds,
        'Y_true test': test.y.astype(int),
        'CM test': cm,
        'CM test norm': cm_norm,

        'Preds train': preds_train,
        'Y_true train': train.y.astype(int),
        'CM train': cm_train,
        'CM train norm': cm_norm_train,
    }
    print('End results', results)
    dumped = json.dumps(numpy_preds, cls=NumpyEncoder)
    with open('../experiments/' + save + '/best/' + save + '_best_model.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open('../experiments/' + save + '/best/' + save + '_best_model_preds.json', 'w', encoding='utf-8') as f:
        json.dump(dumped, f)

    model.best_estimator_.save_model('models/' + save + '.model')
    return model.best_estimator_
