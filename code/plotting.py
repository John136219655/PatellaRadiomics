import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from seaborn import color_palette
from sklearn.metrics import auc, confusion_matrix
from statannotations.Annotator import Annotator


def radscore_performance_comparison(modeling_directory, radscore_folders, metric, export_directory):
    combined_performance = []
    for roi, radscore_folder in radscore_folders.items():
        performances = pd.read_csv(os.path.join(modeling_directory, radscore_folder, 'model_performance.csv'),
                                   index_col=0)
        performances.index.name = 'Cohort'
        performances = performances.reset_index()
        performances = performances[['Cohort', metric]]
        performances['ROI'] = roi
        combined_performance.append(performances)
    combined_performance = pd.concat(combined_performance, axis=0)
    g = sns.catplot(x="ROI", y=metric,
                    row="Cohort",
                    data=combined_performance, kind="bar",
                    height=3, aspect=2)
    plt.ylim(0.5, 0.85)
    # plt.subplots_adjust(bottom=0.1, left=0.2)
    combined_performance.to_csv(os.path.join(export_directory, metric + '_radscore_comparison.csv'))
    plt.savefig(os.path.join(export_directory, metric + '_radscore_comparison.png'), dpi=300)
    plt.savefig(os.path.join(export_directory, metric + '_radscore_comparison.pdf'), dpi=300)
    plt.show()
    plt.clf()


def risk_score_distributions(stratifications, baseline_title, progression_title, export_directory,
                             filename='', baseline_order=None, progression_order=None):
    stratifications = stratifications[[baseline_title, progression_title, 'Risk score']].dropna()
    stratifications = stratifications.rename({'Risk score': 'KR score'}, axis=1)
    if progression_order is None:
        progression_values = stratifications[progression_title].unique()
    else:
        progression_values = progression_order

    if baseline_order is None:
        baseline_values = stratifications[baseline_title].unique()
    else:
        baseline_values = baseline_order

    plt.figure(figsize=(3.5, 3.5))
    stats_pairs = []
    stats_pairs.append(((baseline_values[0], progression_values[1]), (baseline_values[1], progression_values[1])))
    for i in range(len(progression_values) - 1):
        stats_pairs.append(
            ((baseline_values[0], progression_values[i]), (baseline_values[0], progression_values[i + 1])))
        if i == 1:
            stats_pairs.append(
                ((baseline_values[1], progression_values[i]), (baseline_values[1], progression_values[i + 1])))

    cmap = plt.get_cmap('Set1')
    colors = [cmap(2), cmap(4), cmap(0)]
    palette = {x: y for x, y in zip(progression_values, colors)}
    fig_args = dict(x=baseline_title, y='KR score', hue=progression_title,
                    data=stratifications, palette=palette,
                    hue_order=progression_values)
    ax = sns.boxplot(**fig_args)
    # plt.ylim(-4,6.5)
    ax.get_legend().remove()
    # plt.legend(loc='lower left', ncol=3, title=progression_title, frameon=False)
    annotator = Annotator(ax=ax, pairs=stats_pairs, **fig_args)
    annotator.configure(test='Mann-Whitney', text_format="simple", show_test_name=False,
                        loc='inside')
    annotator.apply_and_annotate()
    plt.axhline(y=10, color='black', linestyle='--')
    plt.axhline(y=10, color='black', linestyle='--')
    plt.ylim(-1.5, 9.5)
    export_filename = filename + '_KR_risk_score_comparisons'

    plt.tight_layout()
    plt.savefig(os.path.join(export_directory, export_filename + '.pdf'), dpi=300)
    plt.savefig(os.path.join(export_directory, export_filename + '.png'), dpi=300)
    plt.show()
    plt.clf()


def confusion_matrix_plotting(stratification_record, cohort, baseline_title, time_point, export_directory):
    # stratification_record = pd.read_csv(
    #     os.path.join(model_directory, 'Stratification', '{0}_predictions.csv'.format(cohort)), index_col=0)
    predictions = stratification_record['Prediction'].replace({
        'Low risk': 0,
        'Medium risk': 1,
        'High risk': 2
    }).dropna()
    baseline_stratifications = stratification_record[baseline_title].replace({
        'PFOA-': 0,
        'PFOA+': 1,
    }).dropna()

    pfoa_outcome = stratification_record.loc[predictions.index, time_point + ' Progression'].replace({
        'PFOA-': 0,
        'PFOA+': 1,
        'KR+': 2
    }).dropna()
    # pfoa_outcome = ~(baseline_negative_record[time_point+' Progression']=='PFOA-')
    matrix = confusion_matrix(pfoa_outcome.values, predictions[pfoa_outcome.index].values)
    matrix = pd.DataFrame(matrix)
    matrix.index.name = 'Ground truth'
    matrix.columns.name = 'Prediction'
    matrix.to_csv(os.path.join(export_directory, time_point + '_' + cohort + '_confusion_matrix.csv'))
    plt.figure(figsize=(3, 3))
    g = sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', square=True, cbar_kws={"shrink": .75}, vmin=0, vmax=750)
    plt.tight_layout()
    plt.savefig(os.path.join(export_directory, time_point + '_' + cohort + '_confusion_matrix.png'), dpi=300)
    plt.savefig(os.path.join(export_directory, time_point + '_' + cohort + '_confusion_matrix.pdf'), dpi=300)
    plt.show()
    plt.clf()

    pfoa_outcome = stratification_record.loc[baseline_stratifications.index, time_point + ' Progression'].replace({
        'PFOA-': 0,
        'PFOA+': 1,
        'KR+': 2
    }).dropna()
    # pfoa_outcome = ~(baseline_negative_record[time_point+' Progression']=='PFOA-')
    matrix = confusion_matrix(pfoa_outcome.values, baseline_stratifications[pfoa_outcome.index].values)
    matrix = pd.DataFrame(matrix)
    matrix.index.name = 'Ground truth'
    matrix.columns.name = 'Prediction'
    matrix.to_csv(
        os.path.join(export_directory, time_point + '_' + cohort + '_baseline_confusion_matrix.csv'))
    plt.figure(figsize=(3, 3))
    g = sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', square=True, cbar_kws={"shrink": .75}, vmin=0,
                    vmax=750)
    plt.tight_layout()
    plt.savefig(
        os.path.join(export_directory, time_point + '_' + cohort + '_baseline_confusion_matrix.png'),
        dpi=300)
    plt.savefig(
        os.path.join(export_directory, time_point + '_' + cohort + '_baseline_confusion_matrix.pdf'),
        dpi=300)
    plt.show()
    plt.clf()


def roc_curves(model_directory, model_names, roc_filename, export_directory, filename='', time_point=None, query=None,
               fig_size=(3.5, 3.5)):
    fig, ax = plt.subplots(figsize=fig_size)
    for model_label, model_name in model_names.items():
        roc_table = pd.read_csv(os.path.join(model_directory, model_name, roc_filename), index_col=0)
        if query is not None:
            roc_table = roc_table.query(query)
        if time_point is not None:
            tpr = roc_table[time_point + '_TPR'].dropna().values
            fpr = roc_table[time_point + '_FPR'].dropna().values
        else:
            tpr = roc_table['TPR'].dropna().values
            fpr = roc_table['FPR'].dropna().values
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=model_label + ", AUC={:.3f}".format(auc_score))

    ax.plot([0, 1], [0, 1], linestyle='dashed', color='gray', linewidth=0.5)
    ax.legend(frameon=False)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    plt.tight_layout()
    export_filename = '{0}_{1}_roc'.format(filename, time_point)
    plt.savefig(os.path.join(export_directory, export_filename + '.pdf'), dpi=300)
    plt.savefig(os.path.join(export_directory, export_filename + '.png'), dpi=300)
    plt.show()
    plt.clf()


def km_analysis(data, group_title, duration_title, event_title, export_directory, group_order=None, x_ticks=None,
                x_range=None, rows_to_show=None, colors=None, filename=''):
    if x_ticks is None:
        x_ticks = np.arange(8) * 12
    if x_range is None:
        x_range = (0, 84)
    # if rows_to_show is None:
    #     rows_to_show = ['At risk']
    if group_order is None:
        group_labels = data[group_title].unique()
    else:
        group_labels = group_order
    durations = data[duration_title].values
    data[duration_title] = durations + 0.1

    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    kmfs = []
    if colors is not None:
        colors = color_palette(colors, desat=0.75)

    index = 0
    for group_label in group_labels:
        kmf = KaplanMeierFitter()
        group_index = data[group_title] == group_label
        color = None
        if colors is not None:
            color = colors[index]
        ax = kmf.fit(data.loc[group_index, duration_title].values,
                     data.loc[group_index, event_title].values,
                     label=str(group_label)).plot_survival_function(ax=ax, ci_show=True, color=color)

        kmfs.append(kmf)
        index += 1
    ax.set_xticks(x_ticks)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('KRFS rate')
    ax.set_ylim([0, 1])
    ax.set_xlim(x_range)
    if rows_to_show is not None:
        add_at_risk_counts(*kmfs, xticks=x_ticks, ax=ax, rows_to_show=rows_to_show)
    plt.tight_layout()

    export_filename = filename + '_km_curves'
    plt.savefig(os.path.join(export_directory, export_filename + '.png'), dpi=300)
    plt.savefig(os.path.join(export_directory, export_filename + '.pdf'))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    modeling_directory = r"../modeling"
    export_directory = r"../plotting"
    stratification_directory = os.path.join(modeling_directory, 'Stratification')
    baseline_title = 'Baseline'
    progression_title = '60m Progression'
    progression_names = [0, 1, 2]
    # baseline_order = ['KLG-','KLG+']
    baseline_order = ['PFOA-', 'PFOA+']
    # progression_order = ['KLG-','KLG+','KR+']
    progression_order = ['PFOA-', 'PFOA+', 'KR+']
    group_title = 'Prediction'
    duration_title = 'LKR duration'
    event_title = 'LKR event'
    group_order = ['Low risk', 'Medium risk', 'High risk']
    cmap = plt.get_cmap('Set1')
    colors = [cmap(2), cmap(4), cmap(0)]
    rows_to_show = ['At risk']
    model_names = {
        'PFOA': 'PFOA',
        'RadScore': 'RadScore',
        'KL grade': 'KLG',
        'KR risk score': 'KRRiskScore'
    }
    time_points = ['30m', '60m', '84m']

    roi_folders = {
        'ROI00': 'BBLocal00',
        'ROI01': 'BBLocal01',
        'ROI02': 'BBLocal02',
        'ROI10': 'BBLocal10',
        'ROI11': 'BBLocal11',
        'ROI12': 'BBLocal12',
        'ROI20': 'BBLocal20',
        'ROI21': 'BBLocal21',
        'ROI22': 'BBLocal22',
    }
    metric = 'AUC'
    radscore_performance_comparison(modeling_directory, roi_folders, metric, export_directory)

    training_progressions = pd.read_csv(os.path.join(stratification_directory, 'training_predictions.csv'), index_col=0)
    risk_score_distributions(training_progressions, baseline_title, progression_title, export_directory,
                             baseline_order=baseline_order,
                             progression_order=progression_order,
                             filename='Training')
    confusion_matrix_plotting(training_progressions, 'Training', baseline_title, '60m', export_directory)
    km_analysis(training_progressions, group_title, duration_title, event_title, export_directory,
                group_order=group_order,
                rows_to_show=['At risk'], filename='Training', colors=colors)
    testing_progressions = pd.read_csv(os.path.join(stratification_directory, 'testing_predictions.csv'), index_col=0)
    risk_score_distributions(testing_progressions, baseline_title, progression_title,
                             export_directory,
                             baseline_order=baseline_order,
                             progression_order=progression_order,
                             filename='Testing')
    confusion_matrix_plotting(testing_progressions, 'Testing', baseline_title, '60m', export_directory)
    km_analysis(testing_progressions, group_title, duration_title, event_title, export_directory,
                group_order=group_order, rows_to_show=['At risk'],
                filename='Testing', colors=colors)

    for time_point in time_points:
        roc_export_directory = os.path.join(export_directory, 'roc_' + time_point)
        if not os.path.exists(roc_export_directory):
            os.mkdir(roc_export_directory)

        roc_filename = 'training_KR_roc_curves.csv'
        roc_curves(modeling_directory, model_names, roc_filename, roc_export_directory, filename='Training_KR',
                   time_point=time_point)

        roc_filename = 'testing_KR_roc_curves.csv'
        roc_curves(modeling_directory, model_names, roc_filename, roc_export_directory, filename='Testing_KR',
                   time_point=time_point)
