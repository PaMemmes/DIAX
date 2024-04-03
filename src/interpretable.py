import shap
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

import pandas as pd

from utils.utils import NumpyEncoder


def make_local_plots(
        explainer,
        shap_values,
        test_x,
        preds,
        test_y,
        df_cols,
        name):
    a = 0
    b = 0
    c = 0
    d = 0
    for j in range(len(test_x)):
        if preds[j] == 1 and preds[j] == test_y[j]:
            if a == 15:
                continue
            feature_names = [
                a + ": " + str(b) for a,
                b in zip(
                    df_cols,
                    np.abs(shap_values).mean(0).round(2))]
            shap.force_plot(
                explainer.expected_value,
                shap_values[j, :],
                test_x.iloc[j, :].values,
                feature_names=feature_names,
                matplotlib=True,
                show=False)
            plt.savefig(
                name +
                'force_plot_pred(1)_truth(1)_' +
                str(a) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            shap.decision_plot(explainer.expected_value,
                               shap_values[j,
                                           :],
                               test_x.iloc[j,
                                           :],
                               feature_names=feature_names,
                               link='logit',
                               highlight=0,
                               show=False)
            plt.savefig(
                name +
                'decision_plot_pred(1)_truth(1)_' +
                str(a) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            a += 1

        elif preds[j] == 0 and preds[j] == test_y[j]:
            if b == 15:
                continue
            feature_names = [
                a + ": " + str(b) for a,
                b in zip(
                    df_cols,
                    np.abs(shap_values).mean(0).round(2))]
            shap.force_plot(
                explainer.expected_value,
                shap_values[j, :],
                test_x.iloc[j, :].values,
                feature_names=feature_names,
                matplotlib=True,
                show=False)
            plt.savefig(
                name +
                'force_plot_pred(0)_truth(0)_' +
                str(b) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            shap.decision_plot(explainer.expected_value,
                               shap_values[j,
                                           :],
                               test_x.iloc[j,
                                           :],
                               feature_names=feature_names,
                               link='logit',
                               highlight=0,
                               show=False)
            plt.savefig(
                name +
                'decision_plot_pred(0)_truth(0)_' +
                str(b) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            b += 1

        elif preds[j] == 1 and preds[j] != test_y[j]:
            if c == 15:
                continue
            feature_names = [
                a + ": " + str(b) for a,
                b in zip(
                    df_cols,
                    np.abs(shap_values).mean(0).round(2))]
            shap.force_plot(
                explainer.expected_value,
                shap_values[j, :],
                test_x.iloc[j, :].values,
                feature_names=feature_names,
                matplotlib=True,
                show=False)
            plt.savefig(
                name +
                'force_plot_pred(1)_truth(0)_' +
                str(c) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            shap.decision_plot(explainer.expected_value,
                               shap_values[j,
                                           :],
                               test_x.iloc[j,
                                           :],
                               feature_names=feature_names,
                               link='logit',
                               highlight=0,
                               show=False)
            plt.savefig(
                name +
                'decision_plot_pred(1)_truth(0)_' +
                str(c) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            c += 1

        elif preds[j] == 0 and preds[j] != test_y[j]:
            if d == 15:
                continue
            feature_names = [
                a + ": " + str(b) for a,
                b in zip(
                    df_cols,
                    np.abs(shap_values).mean(0).round(2))]
            shap.force_plot(
                explainer.expected_value,
                shap_values[j, :],
                test_x.iloc[j, :].values,
                feature_names=feature_names,
                matplotlib=True,
                show=False)
            plt.savefig(
                name +
                'force_plot_pred(0)_truth(1)_' +
                str(d) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            shap.decision_plot(explainer.expected_value,
                               shap_values[j,
                                           :],
                               test_x.iloc[j,
                                           :],
                               feature_names=feature_names,
                               link='logit',
                               highlight=0,
                               show=False)
            plt.savefig(
                name +
                'decision_plot_pred(0)_truth(1)_' +
                str(d) +
                '.pdf',
                bbox_inches='tight',
                dpi=300)
            plt.close()
            d += 1

        if a == 15 and b == 15 and c == 15 and d == 15:
            break


def make_interpret_plots(
        explainer,
        shap_values,
        test_x,
        preds,
        test_y,
        df_cols,
        name):
    # Makes plots for interpretation
    # Creates for the first 15 instances force and decision plots
    # Creates summary and summary bar plots for the first 20 (default) features
    # Creates summary and summary bar plots for all features
    # Creates a dependence plot for each feature, dependence is choosing
    # automatically by SHAP
    if len(shap_values) > 2000:
        shap_values = shap_values[:2000]
        test_x = test_x[:2000]
    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        test_x,
        feature_names=df_cols,
        link='logit',
        show=False)
    shap.save_html(name + 'force_plot.htm', f)
    plt.close()

    feature_names = [
        a + ": " + str(b) for a,
        b in zip(
            df_cols,
            np.abs(shap_values).mean(0).round(2))]
    shap.summary_plot(
        shap_values,
        test_x,
        plot_type="bar",
        feature_names=feature_names,
        show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_bar.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values,
        test_x,
        plot_type="bar",
        feature_names=feature_names,
        max_display=len(df_cols),
        show=False
    )
    f = plt.gcf()
    f.savefig(name + 'summary_bar_all.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values,
        test_x,
        feature_names=feature_names,
        show=False)
    f = plt.gcf()
    f.savefig(name + 'summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values,
        test_x,
        feature_names=feature_names,
        max_display=len(df_cols),
        show=False)
    f = plt.gcf()
    f.savefig(name + 'summary_all.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    make_local_plots(
        explainer,
        shap_values,
        test_x,
        preds,
        test_y,
        df_cols,
        name)

    for col in df_cols:
        print('Column', col)
        shap.dependence_plot(col, shap_values, test_x, show=False)
        f = plt.gcf()
        f.savefig(
            name +
            col.replace(
                '/',
                '_') +
            '_dependence.pdf',
            bbox_inches='tight',
            dpi=300)
        plt.close()

    shapleys = {
        'Expected_value': explainer.expected_value,
        'Shapleys': shap_values,
        'text_x': test_x.to_numpy()
    }

    dumped = json.dumps(shapleys, cls=NumpyEncoder)
    with open(name + 'shapley.json', 'w', encoding='utf-8') as f:
        json.dump(dumped, f)


def feature_importance(model, df_cols, importance_type):
    # Calculates feature importance of XGBoost model
    weight = model.get_booster().get_score(importance_type=importance_type)
    sorted_idx = np.argsort(list(weight.values()))
    weight = np.sort(list(weight.values()))
    y = df_cols[sorted_idx]
    return weight, y


def plot_importance(model, name, df_cols, importance_type):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    weight, y = feature_importance(model, df_cols, importance_type)
    if (len(weight) > 25):
        weight = weight[:25]
        y = y[:25]
    ax.barh(y=y, width=weight)
    ax.set_xlabel(importance_type.capitalize() + ' Score')
    ax.set_ylabel('Feature')
    fig.savefig(
        name +
        importance_type +
        '_importance.pdf',
        bbox_inches='tight',
        dpi=300)
    plt.close()


def interpret_tree(model, data, save):
    # Interprets XGBoost model
    name = '../experiments/' + save + '/best/'

    print('Starting interpreting...')

    preds = model.predict(data.test_sqc.x)
    df_cols = data.df_cols[:-1]
    test_df = pd.DataFrame(data.test_sqc.x, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    make_interpret_plots(
        explainer,
        shap_values,
        test_df,
        preds,
        data.test_sqc.y,
        df_cols,
        name)

    plot_importance(model, name, df_cols, 'gain')
    plot_importance(model, name, df_cols, 'weight')

    data.seperate_dfs(filename=None)

    for label, test_sqc in data.seperate_tests.items():
        name = '../experiments/' + save + '/best/' + label + '/'
        Path(name).mkdir(parents=True, exist_ok=True)
        preds = model.predict(test_sqc.x)
        df = pd.DataFrame(test_sqc.x, columns=df_cols)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df, check_additivity=False)
        make_interpret_plots(
            explainer,
            shap_values,
            df,
            preds,
            data.test_sqc.y,
            df_cols,
            name)
