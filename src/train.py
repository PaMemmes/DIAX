from pathlib import Path
import json
import shutil

from hyperopt import hyperopt
from utils.utils import test_model, calc_all_nn, get_preds, open_config, NumpyEncoder
from utils.wasserstein import HyperWGAN
from utils.gan import HyperGAN
from utils.plots import plot_confusion_matrix, plot_roc


def train(
        model_name,
        data,
        trials=1,
        num_retraining=1,
        epochs=1,
        save=False):
    # Train function for wgan or gan
    # Use hyperoptimization for hyperparameter and trains it on the train set anew
    # Evaluation on fragmented data and on csecicids2018 test data
    # Saves results and plots
    experiment = '../experiments/' + save + '/all/experiment'
    Path('../experiments/' + save + '/best/').mkdir(parents=True, exist_ok=True)

    train, test = data.train_sqc, data.test_sqc

    anomalies_percentage = data.anomalies_percentage
    config = open_config(model_name)

    best_hp = hyperopt(model_name, config, train, test, trials)

    num_features = train.x.shape[1]

    saves = []
    all_preds = []
    models = []
    for i in range(num_retraining):
        print('Starting experiment:', i)
        name = experiment + str(i) + '_tuner'
        Path(name).mkdir(parents=True, exist_ok=True)

        if model_name == 'wgan':
            hypermodel = HyperWGAN(
                num_features,
                config,
                discriminator_extra_steps=3,
                gp_weight=5.0)
        elif model_name == 'gan':
            hypermodel = HyperGAN(num_features, config)
        model = hypermodel.build(best_hp)

        hypermodel.fit(best_hp, model, train, epochs=epochs)
        models.append(hypermodel)

        results_df, results = test_model(hypermodel, test)
        y_pred, probas, per, anomalies_percentage = get_preds(
            results, train, data.anomalies_percentage)
        metrics, cm, cm_norm = calc_all_nn(test, y_pred, probas)
        plot_confusion_matrix(cm, name + '/cm.pdf', model_name)
        plot_confusion_matrix(cm_norm, name + '/cm_normalized.pdf', model_name)
        plot_roc(
            metrics['TPR'],
            metrics['FPR'],
            metrics['AUC'],
            name + '/roc.pdf',
            model_name)

        results = {
            'Discriminator Anomalies percentage': anomalies_percentage,
            'Anomalies Percentage': data.anomalies_percentage,
            'Nomal percentage': 1-data.anomalies_percentage,
            'Cutoff': per,
            'Accuracy': metrics['ACC'],
            'Mean Score for normal packets': results_df.loc[results_df['y_test'] == 0, 'results'].mean(),
            'Mean Score for anomalous packets': results_df.loc[results_df['y_test'] == 1, 'results'].mean(),
            'Best HP': best_hp['Dropout'],
            'Metrics test': metrics,
            'I': i
        }
        preds = {
            'Preds': y_pred,
            'Y_true': test.y.astype(int),
            'CM': cm
        }

        with open(name + '/' + save + '.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        saves.append(results)
        all_preds.append(preds)

    best_res = sorted(saves, key=lambda d: d['Accuracy'])[-1]
    numpy_preds = all_preds[best_res['I']]
    print('Best result: ', best_res)

    shutil.copy(experiment +
                str(best_res['I']) +
                '_tuner' +
                '/cm.pdf', '../experiments/' +
                save +
                '/best/cm.pdf')
    shutil.copy(experiment +
                str(best_res['I']) +
                '_tuner' +
                '/cm_normalized.pdf', '../experiments/' +
                save +
                '/best/cm_normalized.pdf')
    shutil.copy(experiment +
                str(best_res['I']) +
                '_tuner' +
                '/roc.pdf', '../experiments/' +
                save +
                '/best/roc.pdf')

    if save is not False:
        dumped = json.dumps(numpy_preds, cls=NumpyEncoder)
        with open('../experiments/' + save + '/best/best_model_wgan.json', 'w', encoding='utf-8') as f:
            json.dump(best_res, f, ensure_ascii=False, indent=4)
        with open('../experiments/' + save + '/best/best_model_preds.json', 'w', encoding='utf-8') as f:
            json.dump(dumped, f)
    return models[best_res['I']]
