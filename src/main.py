import argparse
from train import train
from xg import xg_main
from preprocess import DataFrame
from interpretable import interpret_tree
import tensorflow as tf
import collections
from math import floor

BATCH_SIZE = 256


def run_xg(trials):
    # Runs xg boost model and interprets it
    name = 'xg'
    data = DataFrame()
    data.preprocess(filename=None, kind=None, frags=name, scale=False)
    model = xg_main(
        train=data.train_sqc,
        test=data.test_sqc,
        frags=data.test_frag_sqc,
        trials=trials,
        save=name)
    interpret_tree(model, data, save=name)


def run_nn(model_name, retrain, trials, epochs):
    # Runs neural network: model_name = [gan, wgan]
    data = DataFrame()
    data.preprocess(filename=None, kind='normal', add=None)
    train(
        model_name=model_name,
        data=data,
        frags=data.test_frag_sqc,
        trials=trials,
        num_retraining=retrain,
        epochs=epochs,
        save=model_name)


def run_combined(retrain, trials, epochs):
    # Runs combined model: 1. WGAN 2. XGBoost
    data = DataFrame()
    data.preprocess(filename=None, kind='anomaly', frags=False)
    model = train(
        'wgan',
        data=data,
        frags=None,
        trials=trials,
        num_retraining=retrain,
        epochs=epochs,
        save='combined_wgan')
    gan_data = []
    for i in range(floor(len(data.train_sqc.x) / BATCH_SIZE / 10)):
        noise = tf.random.normal(shape=(BATCH_SIZE, data.train_sqc.x.shape[1]))
        fake_x = model.generator(noise, training=False)
        gan_data.append(fake_x)
    print('Additional data: ', len(gan_data) * BATCH_SIZE)
    data = DataFrame()
    data.preprocess(
        filename=None,
        kind=None,
        add=gan_data,
        scale=False)
    model = xg_main(
        train=data.train_sqc,
        test=data.test_sqc,
        frags=data.test_frag_sqc,
        trials=args.trials,
        save='combined')
    interpret_tree(model, data, save='combined')


if __name__ == '__main__':
    # python main.py 1 1 1
    parser = argparse.ArgumentParser('python3 main.py')
    parser.add_argument(
        'trials',
        help='Number of trials to hyperopt: [1, inf]',
        type=int)
    parser.add_argument(
        'retraining',
        help='Number of times the hyperoptimized model should be retrained [1, inf]',
        type=int)
    parser.add_argument(
        'epochs',
        help='Number of epochs to train: [1, inf]',
        type=int)
    args = parser.parse_args()

    run_nn('gan', args.retraining, args.trials, args.epochs)
    run_nn('wgan', args.retraining, args.trials, args.epochs)
    run_xg(args.trials)
    run_combined(args.retraining, args.trials, args.epochs)

