import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_curve, auc
from sksurv.metrics import concordance_index_censored


def patient_characteristics_test(left_data_table, right_data_table, group_methods, left_data_name='Left',
                                 right_data_name='Right'):
    summary = dict()
    for title, group_method in group_methods.items():
        if group_method == 'Mean':
            left_group_value = left_data_table[title].dropna().mean()
            right_group_value = right_data_table[title].dropna().mean()
            left_min = left_data_table[title].dropna().min()
            left_max = left_data_table[title].dropna().max()
            right_min = right_data_table[title].dropna().min()
            right_max = right_data_table[title].dropna().max()
            statistics, pvalue = ttest_ind(left_data_table[title].dropna().values,
                                           right_data_table[title].dropna().values)
            combined_value = pd.DataFrame([[left_group_value, right_group_value],
                                           [left_min, right_min],
                                           [left_max, right_max],
                                           [pvalue, pvalue]],
                                          index=[group_method, 'Min', 'Max', 'P value'],
                                          columns=[left_data_name, right_data_name])
        elif group_method == 'Median':
            left_group_value = left_data_table[title].dropna().median()
            right_group_value = right_data_table[title].dropna().median()
            left_min = left_data_table[title].dropna().min()
            left_max = left_data_table[title].dropna().max()
            right_min = right_data_table[title].dropna().min()
            right_max = right_data_table[title].dropna().max()
            statistics, pvalue = mannwhitneyu(left_data_table[title].dropna().values,
                                              right_data_table[title].dropna().values)
            combined_value = pd.DataFrame([[left_group_value, right_group_value],
                                           [left_min, right_min],
                                           [left_max, right_max],
                                           [pvalue, pvalue]],
                                          index=[group_method, 'Min', 'Max', 'P value'],
                                          columns=[left_data_name, right_data_name])
        elif group_method == 'Count':
            values, counts = np.unique(left_data_table[title].values, return_counts=True)
            left_group_value = pd.Series(counts, index=values, name=left_data_name)
            values, counts = np.unique(right_data_table[title].values, return_counts=True)
            right_group_value = pd.Series(counts, index=values, name=right_data_name)
            combined_value = pd.concat([left_group_value, right_group_value], axis=1).T
            _, p, _, _ = chi2_contingency(combined_value.loc[:,combined_value.columns.notnull()].values)
            combined_value['P value'] = [p, p]
            combined_value = combined_value.T
        else:
            continue

        summary[title] = combined_value

    summary = pd.concat(summary, axis=0)
    return summary


def progression_classification_performance(model_directory, model_name, cohort, time_points, negative_label,
                                           positive_label, keyword,
                                           baseline_label=None, prediction_title='Risk score'):
    stratification_record = pd.read_csv(
        os.path.join(model_directory, 'Stratification', '{0}_predictions.csv'.format(cohort)), index_col=0)
    if baseline_label is not None:
        if isinstance(baseline_label, str):
            stratification_record = stratification_record[stratification_record['Baseline'] == baseline_label]
        else:
            stratification_record = stratification_record.loc[stratification_record.index.intersection(baseline_label),
                                    :]
    model_scores = \
        pd.read_csv(os.path.join(model_directory, model_name, '{0}_predictions.csv'.format(cohort)), index_col=0)[
            prediction_title].dropna()
    rocs = []
    aucs = {}
    for time_point in time_points:
        progression = stratification_record[time_point + ' Progression'].dropna()
        if negative_label is None:
            negative_progression = progression[progression != positive_label]
        else:
            negative_progression = progression[progression == negative_label]
        positive_progression = progression[progression == positive_label]
        binary_progression = pd.concat(
            [pd.Series([0] * negative_progression.shape[0], index=negative_progression.index),
             pd.Series([1] * positive_progression.shape[0], index=positive_progression.index)])
        common_index = binary_progression.index.intersection(model_scores.index)
        scores = model_scores[common_index]
        binary_progression = binary_progression[common_index]
        fpr, tpr, thresholds = roc_curve(binary_progression.values, scores.values)
        roc = pd.DataFrame([fpr, tpr, thresholds],
                           index=[time_point + '_FPR', time_point + '_TPR', time_point + '_Threshold']).T
        rocs.append(roc)
        aucs[time_point] = auc(fpr, tpr)
    rocs = pd.concat(rocs, axis=1)
    aucs = pd.Series(aucs)
    rocs.to_csv(os.path.join(model_directory, model_name, '{0}_{1}_roc_curves.csv'.format(cohort, keyword)))
    return aucs


def covariate_analysis(preprocessing_directory, model_directory, clinical_factors, cohort, selected_regions,
                       duration_title, event_title):
    training_features = []
    training_clinical = pd.read_csv(os.path.join(preprocessing_directory, '{0}_clinical.csv'.format(cohort)),
                                    index_col=0)
    for selected_region in selected_regions:
        rad_score_filepath = os.path.join(model_directory, selected_region, '{0}_predictions.csv'.format(cohort))
        rad_score = pd.read_csv(rad_score_filepath, index_col=0)['Probability']
        rad_score = (rad_score - rad_score.mean()) / rad_score.std()
        rad_score.name = selected_region
        training_features.append(rad_score.reindex(training_clinical.index))
    training_features.append(training_clinical[clinical_factors])
    training_features = pd.concat(training_features, axis=1).dropna()
    training_clinical = training_clinical.reindex(training_features.index)

    lls_cph = CoxPHFitter()
    univariate_performance = []
    for feature_name in training_features.columns:
        training_table = pd.concat(
            [training_features[[feature_name]], training_clinical[[event_title, duration_title]]], axis=1)
        lls_cph.fit(training_table.copy(), duration_col=duration_title, event_col=event_title)
        model_description = lls_cph.summary.copy()
        univariate_performance.append(model_description)
    univariate_performance = pd.concat(univariate_performance, axis=0)

    training_table = pd.concat([training_features, training_clinical[[event_title, duration_title]]], axis=1)
    lls_cph.fit(training_table.copy(), duration_col=duration_title, event_col=event_title)
    multivariate_performance = lls_cph.summary.copy()

    columns = ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
    performance = pd.concat([univariate_performance[columns], multivariate_performance[columns]], axis=1,
                            keys=['Univariate', 'Multivariate'])
    return performance


def c_index_bs_evaluation(model_directory, model_name, cohort, event_title, duration_title, keyword,
                          baseline_label=None, bootstrap_iter=100):
    stratification_record = pd.read_csv(
        os.path.join(model_directory, 'Stratification', '{0}_predictions.csv'.format(cohort)), index_col=0)
    if baseline_label is not None:
        if isinstance(baseline_label, str):
            stratification_record = stratification_record[stratification_record['Baseline'] == baseline_label]
        else:
            stratification_record = stratification_record.loc[stratification_record.index.intersection(baseline_label),
                                    :]
    predictions = \
        pd.read_csv(os.path.join(model_directory, model_name, '{0}_predictions.csv'.format(cohort)), index_col=0)
    common_patients = predictions.index.intersection(stratification_record.index)
    predictions = predictions.loc[common_patients, :]
    c_index = concordance_index_censored(predictions[event_title].values.astype(bool),
                                         predictions[duration_title].values,
                                         predictions['Risk score'].values)
    original_results = pd.Series(c_index, index=['C-index', 'Concordance', 'Discordant', 'Tied risk', 'Tied time'])
    bootstrap_results = []
    random_state = np.random.RandomState(100)
    sample_number = predictions.shape[0]
    for i in range(bootstrap_iter):
        bootstrap_index = random_state.choice(np.arange(sample_number), sample_number, replace=True)
        c_index = concordance_index_censored(predictions.iloc[bootstrap_index, :][event_title].values.astype(bool),
                                             predictions.iloc[bootstrap_index, :][duration_title].values,
                                             predictions.iloc[bootstrap_index, :]['Risk score'].values)

        bootstrap_results.append(
            pd.Series(c_index, index=['C-index', 'Concordance', 'Discordant', 'Tied risk', 'Tied time']))
    bootstrap_results = pd.concat(bootstrap_results, axis=1).T
    lower_ci = bootstrap_results.quantile(q=0.025, axis=0)
    higher_ci = bootstrap_results.quantile(q=0.975, axis=0)
    summary = pd.concat([original_results, lower_ci, higher_ci], axis=1, ignore_index=True)
    summary.columns = ['Original', 'Lower 95CI', 'Higher 95CI']
    summary = summary.T
    bootstrap_results.to_csv(
        os.path.join(model_directory, model_name, '{0}_{1}_bs_c_index.csv'.format(cohort, keyword)))
    return summary


def c_index_simplified(data, **kwargs):
    # y: 0-event, 1-duration
    c_index = concordance_index_censored(data['Event'].values.astype(bool), data['Duration'].values,
                                         data['Prediction'].values)
    return c_index[0]


def permutation_p_value(measurement_function, left_testing_data, right_testing_data, permutation_iteration=100,
                        random_seed=100,
                        left_training_data=None, right_training_data=None, **kwargs):
    random_state = np.random.RandomState(random_seed)
    left_measurement = measurement_function(left_testing_data, training_data=left_training_data, **kwargs)
    right_measurement = measurement_function(right_testing_data, training_data=right_training_data, **kwargs)
    original_measurement_diff = np.abs(left_measurement - right_measurement)
    diff_distribution = []
    for i in range(permutation_iteration):
        left_choice_mask = random_state.randint(2, size=left_testing_data.shape[0]).astype(bool)
        left_data_shuffles = pd.concat(
            [left_testing_data.loc[left_choice_mask, :], right_testing_data.loc[~left_choice_mask, :]], axis=0)
        right_data_shuffles = pd.concat(
            [left_testing_data.loc[~left_choice_mask, :], right_testing_data.loc[left_choice_mask, :]], axis=0)
        left_measurement = measurement_function(left_data_shuffles, training_data=left_training_data, **kwargs)
        right_measurement = measurement_function(right_data_shuffles, training_data=right_training_data, **kwargs)
        measurement_diff = np.abs(left_measurement - right_measurement)
        diff_distribution.append(measurement_diff)
    diff_distribution = np.array(diff_distribution)
    rejection_rate = np.sum(diff_distribution > original_measurement_diff) + np.sum(
        diff_distribution < -original_measurement_diff)
    rejection_rate = rejection_rate / len(diff_distribution)
    return rejection_rate, original_measurement_diff, diff_distribution


def c_index_permuatation_comparison(model_directory, left_model, right_models, export_directory):
    p_values_combined = {}
    for right_model in right_models:
        p_values = dict()
        for cohort in ['training', 'testing']:
            left_predictions = pd.read_csv(
                os.path.join(model_directory, left_model, '{0}_predictions.csv'.format(cohort)), index_col=0)
            right_predictions = pd.read_csv(
                os.path.join(model_directory, right_model, '{0}_predictions.csv'.format(cohort)),
                index_col=0)
            left_training_data = left_predictions[['LKR event', 'LKR duration', 'Risk score']].rename(
                {'LKR event': 'Event',
                 'LKR duration': 'Duration',
                 'Risk score': 'Prediction'},
                axis=1).dropna()
            right_training_data = right_predictions[['LKR event', 'LKR duration', 'Risk score']].rename(
                {'LKR event': 'Event',
                 'LKR duration': 'Duration',
                 'Risk score': 'Prediction'},
                axis=1).dropna()
            common_index = left_training_data.index.intersection(right_training_data.index)
            left_training_data = left_training_data.loc[common_index, :]
            right_training_data = right_training_data.loc[common_index, :]
            c_index_p_value, c_index_diff, c_index_distribution = permutation_p_value(c_index_simplified,
                                                                                      left_training_data,
                                                                                      right_training_data,
                                                                                      permutation_iteration=5000,
                                                                                      random_seed=100,
                                                                                      left_training_data=left_training_data,
                                                                                      right_training_data=right_training_data)
            p_values[cohort] = c_index_p_value
        p_values = pd.Series(p_values)
        p_values_combined[right_model] = p_values
    p_values_combined = pd.concat(p_values_combined, axis=1)
    p_values_combined.to_csv(os.path.join(export_directory, left_model + '_c_index_p_values.csv'))


if __name__ == '__main__':
    preprocessing_feature_directory = r"../preprocessed_features"
    model_names = ['PFOA', 'RadScore', 'KLG', 'KRRiskScore']
    time_points = ['30m', '60m', '84m']
    modeling_directory = r"../modeling"
    export_directory = r"../statistical_analysis"
    clinical_factors = ['AGE', 'SEX', 'V0BMI', 'V0WOPNKL', 'V0XLKL', 'V0XLPFROA']
    duration_title = 'LKR duration'
    event_title = 'LKR event'

    left_data_table = pd.read_csv(os.path.join(preprocessing_feature_directory, 'training_clinical.csv'), index_col=0)
    right_data_table = pd.read_csv(os.path.join(preprocessing_feature_directory, 'testing_clinical.csv'), index_col=0)
    group_methods = {
        'AGE': 'Mean',
        'SEX': 'Count',
        'V0BMI': 'Mean',
        'V0WOPNKL': 'Median',
        'V0XLPFROA': 'Count',
        'V0XLKL': 'Count',
        'LKR duration': 'Mean',
        'LKR event': 'Count'
    }
    summary = patient_characteristics_test(left_data_table, right_data_table, group_methods, left_data_name='Training',
                                           right_data_name='Testing')
    summary.to_csv(os.path.join(export_directory, 'patient_characteristics.csv'))

    selected_regions = ['BBLocal20']
    for cohort in ['training', 'testing']:
        performance = covariate_analysis(preprocessing_feature_directory, modeling_directory, clinical_factors, cohort,
                                         selected_regions,
                                         duration_title, event_title)
        performance.to_csv(os.path.join(export_directory, '{0}_covariate_analysis.csv'.format(cohort)))

    negative_label = None
    positive_label = 'KR+'
    baseline_label = None
    keyword = 'KR'
    auc_summary = []
    for model_name in model_names:
        for cohort in ['training', 'testing']:
            aucs = progression_classification_performance(modeling_directory, model_name, cohort, time_points,
                                                          negative_label, positive_label,
                                                          keyword,
                                                          baseline_label=baseline_label)
            aucs.name = (model_name, cohort)
            auc_summary.append(aucs)
    auc_summary = pd.concat(auc_summary, axis=1)
    auc_summary.to_csv(os.path.join(export_directory, keyword + '_aucs.csv'))
    print(auc_summary)

    event_title = 'LKR event'
    duration_title = 'LKR duration'
    keyword = 'KR'
    bootstrap_iter = 1000
    baseline_label = None
    c_indexs = {}
    for model_name in model_names:
        for cohort in ['training', 'testing']:
            c_index = c_index_bs_evaluation(modeling_directory, model_name, cohort, event_title, duration_title,
                                            keyword,
                                            baseline_label=baseline_label, bootstrap_iter=bootstrap_iter)
            c_indexs[model_name + '_' + cohort] = c_index
    c_indexs = pd.concat(c_indexs, axis=0)
    c_indexs.to_csv(os.path.join(export_directory, keyword + '_c_index_statistics.csv'))

    reference_models = ['KLG', 'PFOA']
    right_models = ['PFOA', 'KLG', 'RadScore', 'KRRiskScore']
    for reference_model in reference_models:
        c_index_permuatation_comparison(modeling_directory, reference_model, right_models, export_directory)
