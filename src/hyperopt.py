import keras_tuner

from utils.wasserstein import HyperWGAN
from utils.gan import HyperGAN


def hyperopt(model, config, train, val, num_trials):
    num_features = train.x.shape[1]

    if model == 'wgan':
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=HyperWGAN(
                num_features,
                config,
                discriminator_extra_steps=5,
                gp_weight=10.0),
            max_trials=num_trials,
            overwrite=True,
            directory="./hyperopt",
            project_name="HyperWGAN",
        )
    else:
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=HyperGAN(num_features, config),
            max_trials=num_trials,
            overwrite=True,
            directory="./hyperopt",
            project_name="HyperGAN",
        )

    tuner.search(
        train,
        validation_data=(val.x, val.y)
    )

    tuner.results_summary()

    return tuner.get_best_hyperparameters(5)[0]
