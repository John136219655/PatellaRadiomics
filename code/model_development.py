import json
import os
import warnings

import matplotlib.pyplot as plt
import mrmr
import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.under_sampling import RandomUnderSampler
from lifelines import CoxPHFitter, KaplanMeierFitter, CRCSplineFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import CensoringType
from sklearn.feature_selection import f_classif, r_regression, VarianceThreshold, SelectFdr, SelectKBest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score, \
    _check_estimate_2d
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import check_y_survival


def confounding_correlation_reduction(feature_table, confounding_factor_table):
    common_index = feature_table.dropna().index.intersection(confounding_factor_table.dropna().index)
    for confounding_factor in confounding_factor_table.columns:
        unique_confounding_values = confounding_factor_table[confounding_factor].unique()
        if len(unique_confounding_values) < 5:
            chi2_stats, p_values = f_classif(feature_table.loc[common_index, :].values,
                                             confounding_factor_table.loc[common_index, confounding_factor].values)
            feature_selection_mask = p_values > 0.05
        else:
            r = r_regression(feature_table.loc[common_index, :].values,
                             confounding_factor_table.loc[common_index, confounding_factor].values)
            feature_selection_mask = np.abs(r) < 0.65
        feature_table = feature_table.loc[:, feature_selection_mask]
        # print(pearsonr(feature_table['original_firstorder_Energy'].values, confounding_factor_table[confounding_factor].values))
    return feature_table


def classification_model_performance(model, predictors, ground_truth, return_roc=False):
    scores = {}
    predictions_proba = model.decision_function(predictors)
    predictions_binary = model.predict(predictors)
    for metric_name in ['AUC', 'Accuracy', 'Precision', 'Recall']:
        if metric_name == 'AUC':
            score = roc_auc_score(ground_truth, predictions_proba)
        elif metric_name == 'Accuracy':
            score = accuracy_score(ground_truth, predictions_binary)
        elif metric_name == 'Precision':
            score = precision_score(ground_truth, predictions_binary)
        elif metric_name == 'Recall':
            score = recall_score(ground_truth, predictions_binary)
        else:
            continue
        scores[metric_name] = score
    scores = pd.Series(scores)
    if return_roc:
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions_proba)
        roc = pd.DataFrame([fpr, tpr, thresholds], index=['FPR', 'TPR', 'Threshold']).T
        return scores, roc
    else:
        return scores


class RadScoreDevelopment:
    def __init__(self, preprocessing_feature_directory, region_name, duration_title, event_title, survival_timepoint,
                 feature_selection_parameters=dict(), patients=None):
        self.preprocessing_feature_directory = preprocessing_feature_directory
        self.region_name = region_name
        self.duration_title = duration_title
        self.event_title = event_title
        self.survival_timepoint = survival_timepoint
        self.feature_selection_parameters = feature_selection_parameters
        self.patients = patients
        self.training_features = None
        self.training_clinical = None
        self.training_outcome = None
        self.testing_features = None
        self.testing_clinical = None
        self.testing_outcome = None
        self.scaler = None
        self.model = None

    def load_training_data(self):
        training_feature_filepath = os.path.join(self.preprocessing_feature_directory,
                                                 "training_{0}.csv".format(self.region_name))
        self.training_features = pd.read_csv(training_feature_filepath, index_col=0)
        training_clinical_filepath = os.path.join(self.preprocessing_feature_directory, 'training_clinical.csv')
        self.training_clinical = pd.read_csv(training_clinical_filepath, index_col=0)
        training_outcome = self.training_clinical[self.duration_title].values < self.survival_timepoint
        validity = ~np.all([training_outcome, self.training_clinical[self.event_title] == 0], axis=0)
        if self.survival_timepoint == 84:
            training_outcome = self.training_clinical[self.event_title]
        else:
            training_outcome = self.training_clinical[self.duration_title] <= self.survival_timepoint
        self.training_outcome = training_outcome.loc[validity]
        if self.patients is not None:
            common_patients = self.patients.intersection(self.training_outcome.index)
            self.training_outcome = self.training_outcome[common_patients]
            self.training_clinical = self.training_clinical.loc[common_patients, :]
            self.training_features = self.training_features.loc[common_patients, :]

    def load_testing_data(self):
        testing_feature_filepath = os.path.join(self.preprocessing_feature_directory,
                                                "testing_{0}.csv".format(self.region_name))
        self.testing_features = pd.read_csv(testing_feature_filepath, index_col=0)
        testing_clinical_filepath = os.path.join(self.preprocessing_feature_directory, 'testing_clinical.csv')
        self.testing_clinical = pd.read_csv(testing_clinical_filepath, index_col=0)
        testing_outcome = self.testing_clinical[self.duration_title].values < self.survival_timepoint
        validity = ~np.all([testing_outcome, self.testing_clinical[self.event_title] == 0], axis=0)
        if self.survival_timepoint == 84:
            testing_outcome = self.testing_clinical[self.event_title]
        else:
            testing_outcome = self.testing_clinical[self.duration_title] <= self.survival_timepoint
        self.testing_outcome = testing_outcome.loc[validity]
        if self.patients is not None:
            common_patients = self.patients.intersection(self.testing_outcome.index)
            self.testing_outcome = self.testing_outcome[common_patients]
            self.testing_clinical = self.testing_clinical.loc[common_patients, :]
            self.testing_features = self.testing_features.loc[common_patients, :]

    def feature_selection_pipeline(self, training_features=None, training_clinical=None, training_outcome=None,
                                   inplace=True):
        if training_features is None:
            training_features = self.training_features
        if training_clinical is None:
            training_clinical = self.training_clinical
        if training_outcome is None:
            training_outcome = self.training_outcome

        selected_training_features = training_features

        including_keywords = self.feature_selection_parameters.get('Including keywords')
        if including_keywords is not None:
            selected_feature_names = []
            for feature_name in selected_training_features.columns.values:
                if np.all([keyword in feature_name for keyword in including_keywords]):
                    selected_feature_names.append(feature_name)
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after keyword inclusion filtering.'.format(selected_training_features.shape[1]))

        excluding_keywords = self.feature_selection_parameters.get('Excluding keywords')
        if excluding_keywords is not None:
            selected_feature_names = []
            for feature_name in selected_training_features.columns.values:
                if np.any([keyword in feature_name for keyword in excluding_keywords]):
                    continue
                selected_feature_names.append(feature_name)
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after keyword exclusion filtering.'.format(selected_training_features.shape[1]))

        confounding_factors = self.feature_selection_parameters.get('Confounding factors')
        if confounding_factors is not None:
            confounding_factor_table = []
            for confounding_factor in confounding_factors:
                if confounding_factor in training_features.columns:
                    confounding_factor_table.append(training_features[confounding_factor])
                elif confounding_factor in training_clinical.columns:
                    confounding_factor_table.append(training_clinical[confounding_factor])
                else:
                    continue
            confounding_factor_table = pd.concat(confounding_factor_table, axis=1)
            selected_training_features = confounding_correlation_reduction(selected_training_features,
                                                                           confounding_factor_table)
            print('{0} features remain after confounding factor filtering.'.format(selected_training_features.shape[1]))

        variance_threshold = self.feature_selection_parameters.get('Variance threshold')
        if variance_threshold is not None:
            sel = VarianceThreshold(threshold=variance_threshold)
            # filter the original MR radiomics feature table array by the variance threshold, the filtered feature array is returned
            sel.fit(selected_training_features.values)
            # get the selected feature names by inputting the original feature names
            selected_training_features = selected_training_features.iloc[:, sel.get_support()]
            print('{0} features remain after variance filtering.'.format(selected_training_features.shape[1]))

        # scaler = StandardScaler()
        # standardized_feature_array = scaler.fit_transform(selected_training_features.values)
        # standardized_training_features = pd.DataFrame(standardized_feature_array, columns=selected_training_features.columns,
        #                                       index = selected_training_features.index)
        p_value_threshold = self.feature_selection_parameters.get('P value')
        if p_value_threshold is not None:
            feature_selector = SelectFdr(alpha=p_value_threshold)
            feature_selector.fit(selected_training_features.loc[self.training_outcome.index, :],
                                 self.training_outcome)
            selected_feature_names = selected_training_features.columns.values[feature_selector.get_support()]
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after FDR filtering.'.format(selected_training_features.shape[1]))

        final_feature_number = self.feature_selection_parameters.get('Feature number')
        if final_feature_number is not None:
            feature_selector = SelectKBest(k=final_feature_number)
            feature_selector.fit(selected_training_features.loc[self.training_outcome.index, :],
                                 self.training_outcome)
            selected_feature_names = selected_training_features.columns.values[feature_selector.get_support()]
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after K best filtering.'.format(selected_training_features.shape[1]))

        final_feature_number = self.feature_selection_parameters.get('Downsample feature number')
        if final_feature_number is not None:
            rus = RandomUnderSampler(random_state=42)
            frequency = []
            for i in range(100):
                X_res, y_res = rus.fit_resample(selected_training_features.loc[self.training_outcome.index, :].values,
                                                self.training_outcome.values)
                feature_selector = SelectKBest(k=final_feature_number)
                feature_selector.fit(X_res, y_res)
                frequency.append(feature_selector.get_support())
            feature_selection_mask = np.flip(np.argsort(np.sum(frequency, axis=0)))[:final_feature_number]
            selected_feature_names = selected_training_features.columns.values[feature_selection_mask]
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after K best filtering.'.format(selected_training_features.shape[1]))

        final_feature_number = self.feature_selection_parameters.get('AUC')
        if final_feature_number is not None:
            aucs = []
            for feature_name in selected_training_features.columns.values:
                auc = roc_auc_score(self.training_outcome.values,
                                    selected_training_features.loc[self.training_outcome.index, feature_name])
                aucs.append(auc)
            feature_selection_mask = np.flip(np.argsort(aucs))[:final_feature_number]
            selected_feature_names = selected_training_features.columns.values[feature_selection_mask]
            selected_training_features = selected_training_features[selected_feature_names]
            print('{0} features remain after K best filtering.'.format(selected_training_features.shape[1]))

        mrmr_feature_number = self.feature_selection_parameters.get('MRMR feature number')
        if mrmr_feature_number is not None:
            selected_feature_names = mrmr.mrmr_classif(X=selected_training_features.loc[training_outcome.index, :],
                                                       y=training_outcome, K=mrmr_feature_number)
            selected_training_features = selected_training_features.loc[:, selected_feature_names]
            print('{0} features remain after MRMR filtering.'.format(len(selected_feature_names)))

        if inplace:
            self.selected_features = list(selected_training_features.columns)
        else:
            return list(selected_training_features.columns)

    def model_training(self, training_features=None, training_outcome=None,
                       selected_features=None, inplace=True, export_directory=None):
        if training_features is None:
            training_features = self.training_features
        if training_outcome is None:
            training_outcome = self.training_outcome
        if selected_features is None:
            selected_features = self.selected_features

        training_features = training_features.loc[training_outcome.index, selected_features].values
        training_labels = training_outcome.values
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        model = EasyEnsembleClassifier(base_estimator=RidgeClassifierCV(), n_estimators=500)
        model.fit(training_features, training_labels)

        if export_directory is not None:
            model_parameters = self._model_export(model=model, scaler=scaler)
            model_parameters.to_csv(os.path.join(export_directory, 'model_parameters.csv'))

        if inplace:
            self.model = model
            self.scaler = scaler
        else:
            return scaler, model

    def _model_export(self, model=None, scaler=None):
        if model is not None:
            self.model = model
        if scaler is not None:
            self.scaler = scaler
        if self.model is None:
            print('The model has not been trained.')
            return
        ensemble_model_parameters = []
        for sub_model in self.model.estimators_:
            sub_model = sub_model['classifier']
            # print(sub_model)
            model_parameters = list(sub_model.coef_.flatten()) + list(sub_model.intercept_)
            model_parameters = pd.Series(model_parameters, index=list(self.selected_features) + ['intercept'])
            ensemble_model_parameters.append(model_parameters)
        ensemble_model_parameters.append(
            pd.Series(self.scaler.scale_, name='Scale', index=list(self.selected_features)))
        ensemble_model_parameters.append(pd.Series(self.scaler.mean_, name='Mean', index=list(self.selected_features)))
        ensemble_model_parameters.append(pd.Series(self.scaler.var_, name='Var', index=list(self.selected_features)))
        ensemble_model_parameters = pd.concat(ensemble_model_parameters, axis=1)
        return ensemble_model_parameters

    def model_performance(self, export_directory=None):
        training_features = self.training_features.loc[self.training_outcome.index, self.selected_features].values
        training_labels = self.training_outcome.values
        training_features = self.scaler.transform(training_features)
        training_scores, training_roc = classification_model_performance(self.model, training_features, training_labels,
                                                                         return_roc=True)

        testing_features = self.testing_features.loc[self.testing_outcome.index, self.selected_features].values
        testing_labels = self.testing_outcome.values
        testing_features = self.scaler.transform(testing_features)
        testing_scores, testing_roc = classification_model_performance(self.model, testing_features, testing_labels,
                                                                       return_roc=True)

        scores = pd.concat([training_scores, testing_scores], axis=1, keys=['Training', 'Testing']).T
        training_roc['Cohort'] = ['Training'] * training_roc.shape[0]
        testing_roc['Cohort'] = ['Testing'] * testing_roc.shape[0]
        rocs = pd.concat([training_roc, testing_roc], axis=0)

        if export_directory is not None:
            training_features_transformed = self.scaler.transform(self.training_features[self.selected_features].values)
            training_probabilities = self.model.decision_function(training_features_transformed)
            training_predictions = self.model.predict(training_features_transformed)
            self.training_outcome.name = 'Ground truth'
            training_predictions = pd.concat([self.training_features[self.selected_features],
                                              pd.Series(training_probabilities, index=self.training_features.index,
                                                        name='Probability'),
                                              pd.Series(training_predictions, index=self.training_features.index,
                                                        name='Prediction'),
                                              self.training_outcome],
                                             axis=1)
            training_predictions.to_csv(os.path.join(export_directory, 'training_predictions.csv'))

            testing_features_transformed = self.scaler.transform(self.testing_features[self.selected_features].values)
            testing_probabilities = self.model.decision_function(testing_features_transformed)
            testing_predictions = self.model.predict(testing_features_transformed)
            self.testing_outcome.name = 'Ground truth'
            testing_predictions = pd.concat([self.testing_features[self.selected_features],
                                             pd.Series(testing_probabilities, index=self.testing_features.index,
                                                       name='Probability'),
                                             pd.Series(testing_predictions, index=self.testing_features.index,
                                                       name='Prediction'),
                                             self.testing_outcome],
                                            axis=1)
            testing_predictions.to_csv(os.path.join(export_directory, 'testing_predictions.csv'))
        return scores, rocs

    def execute(self, export_directory):
        self.load_training_data()
        self.load_testing_data()
        self.feature_selection_pipeline()
        print('{0} feature selected.'.format(len(self.selected_features)))
        self.model_training(export_directory=export_directory)
        scores, rocs = self.model_performance(export_directory=export_directory)
        print('Model performance:')
        print(scores.loc[:, 'AUC'])
        scores.to_csv(os.path.join(export_directory, 'model_performance.csv'))
        rocs.to_csv(os.path.join(export_directory, 'rocs.csv'))


def ccl(p):
    return np.log(-np.log(1 - p))


def survival_roc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times)

    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    # fit and transform IPCW
    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)

    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.infty, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    rocs = []
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            rocs.append((tp_no_ties, fp_no_ties))
    return rocs


class SurvivalAnalysis:
    def __init__(self):
        self.model_description = None
        self.model = None
        self.duration_title = 'duration'
        self.event_title = 'event_title'
        self.cutoffs = None
        self.group_names = None

    def fit(self, features, outcomes, duration_title='duration',
            event_title='event'):
        self.duration_title = duration_title
        self.event_title = event_title
        training_table = pd.concat([features, outcomes[[event_title, duration_title]]], axis=1)
        training_x = features.values
        training_y = outcomes[[event_title, duration_title]].astype(
            {event_title: 'bool', duration_title: 'int32'}).to_records(index=False)

        lls_cph = CoxPHFitter()
        sksurv_cph = CoxPHSurvivalAnalysis()
        lls_cph.fit(training_table.copy(), duration_col=duration_title, event_col=event_title)
        sksurv_cph.fit(training_x, training_y)

        self.model_description = lls_cph.summary.copy()
        self.model = (lls_cph, sksurv_cph, training_x, training_y)
        return self.model_description

    def predict(self, features):
        if self.model is None:
            print('The survival model has not been trained.')
            return
        lls_cph, sksurv_cph, training_x, training_y = self.model
        testing_x = features.values
        testing_risk_scores = sksurv_cph.predict(testing_x)
        return pd.Series(testing_risk_scores, index=features.index, name='Risk score')

    def performance(self, features, outcomes, parameters, duration_title=None, event_title=None):
        predictions = self.predict(features)
        # print(predictions.shape)
        if predictions is None:
            print('Cannot get risk predictions.')
            return
        if duration_title is None:
            duration_title = self.duration_title
        if event_title is None:
            event_title = self.event_title
        lls_cph, sksurv_cph, training_x, training_y = self.model
        testing_x = features.values
        testing_y = outcomes[[event_title, duration_title]].astype(
            {event_title: 'bool', duration_title: 'int32'}).to_records(index=False)
        results = []
        # print(testing_x.shape)
        # print(testing_y.shape)
        for performance_metric, metric_parameters in parameters.items():
            try:
                if performance_metric == 'c-index':
                    # Concordance-index
                    c_index = concordance_index_censored(testing_y[event_title],
                                                         testing_y[duration_title],
                                                         predictions.values)
                    result = pd.Series(c_index,
                                       index=['C-index', 'Concordance', 'Discordant', 'Tied risk', 'Tied time'])
                elif performance_metric == 'auc':
                    # the default time points for AUC calculations are 1 to 5 years with 1-year interval
                    times = metric_parameters.get('times', np.arange(1, 6) * 12)
                    # Time dependent AUC evaluation
                    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
                        training_y, testing_y, predictions.values, times
                    )
                    result = pd.Series(cph_auc, index=['{0}m AUC'.format(x) for x in times])
                    result['Mean AUC'] = cph_mean_auc
                elif performance_metric == 'ibs':
                    # the default time points for AUC calculations are 1 to 5 years with 1-year interval
                    times = metric_parameters.get('times', np.arange(1, 6) * 12)
                    # Integrated Brier score evaluation
                    survs = sksurv_cph.predict_survival_function(testing_x)
                    probs = np.asarray([[fn(t) for t in times] for fn in survs])
                    ibs = integrated_brier_score(training_y, testing_y, probs, times)
                    result = pd.Series([ibs], index=['IBS'])
                else:
                    continue
            except Exception as e:
                print(e)
                continue
            results.append(result)
        results = pd.concat(results, axis=0)
        return results

    def bootstrap_performance(self, features, outcomes, parameters, random_seed=None, bootstrap_iter=1000,
                              duration_title=None, event_title=None):
        if duration_title is None:
            duration_title = self.duration_title
        if event_title is None:
            event_title = self.event_title
        random_state = np.random.RandomState(random_seed)
        sample_number = features.shape[0]
        bs_performance = [self.performance(features, outcomes,
                                           parameters, duration_title=duration_title, event_title=event_title)]
        for i in range(bootstrap_iter):
            bootstrap_index = random_state.choice(np.arange(sample_number), sample_number, replace=True)
            performance = self.performance(features.iloc[bootstrap_index, :], outcomes.iloc[bootstrap_index, :],
                                           parameters, duration_title=duration_title, event_title=event_title)
            bs_performance.append(performance)
        return pd.concat(bs_performance, axis=1).T

    def roc_curves(self, features, outcomes, times, duration_title=None, event_title=None):
        if duration_title is None:
            duration_title = self.duration_title
        if event_title is None:
            event_title = self.event_title
        predictions = self.predict(features)
        if predictions is None:
            print('Cannot get risk predictions.')
            return
        lls_cph, sksurv_cph, training_x, training_y = self.model
        testing_y = outcomes[[event_title, duration_title]].astype(
            {event_title: 'bool', duration_title: 'int32'}).to_records(index=False)

        rocs = survival_roc(training_y, testing_y, predictions.values, times, tied_tol=1e-8)
        roc_dataframe = dict()
        for time, roc in zip(times, rocs):
            roc_dataframe[str(time) + 'm_TPR'] = pd.Series(roc[0])
            roc_dataframe[str(time) + 'm_FPR'] = pd.Series(roc[1])
        roc_dataframe = pd.DataFrame(roc_dataframe)
        return roc_dataframe

    def calibration_curve(self, features, outcomes, time_point, duration_title=None, event_title=None):
        if duration_title is None:
            duration_title = self.duration_title
        if event_title is None:
            event_title = self.event_title
        lls_cph, sksurv_cph, training_x, training_y = self.model
        predictions_at_t0 = np.clip(1 - lls_cph.predict_survival_function(features, times=[time_point]).T.squeeze(),
                                    1e-10, 1 - 1e-10)
        prediction_df = pd.DataFrame(
            {"ccl_at_%d" % time_point: ccl(predictions_at_t0), duration_title: outcomes[duration_title],
             event_title: outcomes[event_title]})
        knots = 3
        regressors = {"beta_": ["ccl_at_%d" % time_point], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

        # this model is from examples/royson_crowther_clements_splines.py
        crc = CRCSplineFitter(knots, penalizer=0.000001)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if CensoringType.is_right_censoring(lls_cph):
                crc.fit_right_censoring(prediction_df, duration_title, event_title, regressors=regressors)
            elif CensoringType.is_left_censoring(lls_cph):
                crc.fit_left_censoring(prediction_df, duration_title, event_title, regressors=regressors)
            elif CensoringType.is_interval_censoring(lls_cph):
                crc.fit_interval_censoring(prediction_df, duration_title, event_title, regressors=regressors)

        observed_df = 1 - crc.predict_survival_function(prediction_df, times=[time_point]).T.squeeze()

        # # predict new model at values 0 to 1, but remember to ccl it!
        x = np.linspace(np.clip(predictions_at_t0.min() - 0.01, 0, 1), np.clip(predictions_at_t0.max() + 0.01, 0, 1),
                        100)
        y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % time_point: ccl(x)}),
                                              times=[time_point]).T.squeeze()

        deltas = (observed_df - predictions_at_t0).abs()

        sorting_index = np.argsort(predictions_at_t0)
        bins, bin_edges = np.histogram(observed_df, bins=10)

        results = pd.concat([
            pd.Series(predictions_at_t0[sorting_index], name='Predicted probability').reset_index(drop=True),
            pd.Series(observed_df[sorting_index], name='Observed probability').reset_index(drop=True),
            pd.Series(bins, name='Observed probability histogram').reset_index(drop=True),
            pd.Series(bin_edges, name='Observed probability bin edges').reset_index(drop=True),
            pd.Series(deltas, name='Delta').reset_index(drop=True),
            pd.Series(x, name='Predicted probability interp').reset_index(drop=True),
            pd.Series(y, name='Observed probability interp').reset_index(drop=True),

        ], axis=1)

        return results


class RiskScoreDevelopment:
    def __init__(self, preprocessing_directory, model_directory, selected_regions, clinical_factors, duration_title,
                 event_title,
                 performance_parameters=dict(), patients=None):
        self.preprocessing_directory = preprocessing_directory
        self.model_directory = model_directory
        self.selected_regions = selected_regions
        self.clinical_factors = clinical_factors
        self.duration_title = duration_title
        self.event_title = event_title
        self.performance_parameters = performance_parameters
        self.patients = patients

        self.training_features = None
        self.training_clinical = None
        self.testing_features = None
        self.testing_clinical = None

    def load_training_data(self):
        self.training_features = []
        self.training_clinical = pd.read_csv(os.path.join(self.preprocessing_directory, 'training_clinical.csv'),
                                             index_col=0)
        for selected_region in self.selected_regions:
            rad_score_filepath = os.path.join(self.model_directory, selected_region, 'training_predictions.csv')
            rad_score = pd.read_csv(rad_score_filepath, index_col=0)['Probability']
            rad_score = (rad_score - rad_score.mean()) / rad_score.std()
            rad_score.name = selected_region
            self.training_features.append(rad_score.reindex(self.training_clinical.index))
        self.training_features.append(self.training_clinical[self.clinical_factors])
        self.training_features = pd.concat(self.training_features, axis=1).dropna()
        common_patients = self.training_clinical.index.intersection(self.training_features.index)
        self.training_clinical = self.training_clinical.loc[common_patients, :]
        self.training_features = self.training_features.loc[common_patients, :]
        durations = self.training_clinical[self.duration_title].values
        durations[durations > 95] = 95
        self.training_clinical[self.duration_title] = durations

    def load_testing_data(self):
        self.testing_features = []
        self.testing_clinical = pd.read_csv(os.path.join(self.preprocessing_directory, 'testing_clinical.csv'),
                                            index_col=0)
        for selected_region in self.selected_regions:
            rad_score_filepath = os.path.join(self.model_directory, selected_region, 'testing_predictions.csv')
            rad_score = pd.read_csv(rad_score_filepath, index_col=0)['Probability']
            rad_score = (rad_score - rad_score.mean()) / rad_score.std()
            rad_score.name = selected_region
            self.testing_features.append(rad_score.reindex(self.testing_clinical.index))
        self.testing_features.append(self.testing_clinical[self.clinical_factors])
        self.testing_features = pd.concat(self.testing_features, axis=1).dropna()
        common_patients = self.testing_clinical.index.intersection(self.testing_features.index)
        self.testing_clinical = self.testing_clinical.loc[common_patients, :]
        self.testing_features = self.testing_features.loc[common_patients, :]
        durations = self.testing_clinical[self.duration_title].values
        durations[durations > 95] = 95
        self.testing_clinical[self.duration_title] = durations

    def model_training(self, export_directory=None):
        self.model = SurvivalAnalysis()
        model_description = self.model.fit(self.training_features,
                                           self.training_clinical[[self.event_title, self.duration_title]],
                                           duration_title=self.duration_title, event_title=self.event_title)
        if export_directory is not None:
            model_description.to_csv(os.path.join(export_directory, 'model_descriptions.csv'))

    def model_performance(self, export_directory=None):
        training_outcomes = self.training_clinical[[self.event_title, self.duration_title]]
        testing_outcomes = self.testing_clinical[[self.event_title, self.duration_title]]
        training_performance = self.model.performance(self.training_features,
                                                      training_outcomes,
                                                      self.performance_parameters,
                                                      duration_title=duration_title,
                                                      event_title=event_title)
        testing_performance = self.model.performance(self.testing_features,
                                                     testing_outcomes,
                                                     self.performance_parameters,
                                                     duration_title=duration_title,
                                                     event_title=event_title)
        if export_directory is not None:
            training_predictions = self.model.predict(self.training_features)
            training_predictions = pd.concat([self.training_features, training_predictions, training_outcomes], axis=1)
            training_predictions.to_csv(os.path.join(export_directory, 'training_predictions.csv'))

            testing_predictions = self.model.predict(self.testing_features)
            testing_predictions = pd.concat([self.testing_features, testing_predictions, testing_outcomes],
                                            axis=1)
            testing_predictions.to_csv(os.path.join(export_directory, 'testing_predictions.csv'))

        representative_time_points = self.performance_parameters.get('Representative time points')
        if representative_time_points is not None:
            for time_point in representative_time_points:
                training_calibration_curve = self.model.calibration_curve(self.training_features, training_outcomes,
                                                                          time_point,
                                                                          duration_title=duration_title,
                                                                          event_title=event_title)
                testing_calibration_curve = self.model.calibration_curve(self.testing_features,
                                                                         testing_outcomes, time_point,
                                                                         duration_title=duration_title,
                                                                         event_title=event_title)
                if export_directory is not None:
                    training_calibration_curve.to_csv(
                        os.path.join(export_directory, '{0}m_training_calibration_curve.csv'.format(time_point)))
                    testing_calibration_curve.to_csv(
                        os.path.join(export_directory, '{0}m_testing_calibration_curve.csv'.format(time_point)))
        return training_performance, testing_performance

    def execute(self, export_directory=None):
        self.load_training_data()
        self.load_testing_data()
        self.model_training(export_directory=export_directory)
        training_performance, testing_performance = self.model_performance(export_directory=export_directory)
        performance = pd.concat([training_performance, testing_performance], axis=1, keys=['Training', 'Testing'])
        print(performance.loc['C-index', :])
        performance.to_csv(os.path.join(export_directory, 'performance.csv'))


def optimum_binary_cutoff_search(risk_scores, outcomes):
    print(roc_auc_score(outcomes.values, risk_scores.values))
    fpr, tpr, thresholds = roc_curve(outcomes.values, risk_scores.values)
    J = tpr - fpr
    ix = np.argmax(J)
    final_threshold = thresholds[ix]
    return final_threshold


def progression_label_generation(clinical, baseline_factor, duration_title, event_title, survival_timepoints,
                                 secondary_timepoints, baseline_threshold=0, secondary_endpoint_threshold=0):
    outcomes = []
    if baseline_threshold is not None:
        baseline_outcome = (clinical[baseline_factor].dropna() > baseline_threshold).astype('Int8')
        baseline_outcome.name = 'Baseline'
        outcomes.append(baseline_outcome)
    for timepoint in survival_timepoints:
        validity = ~np.all([clinical[duration_title].values < timepoint,
                            clinical[event_title] == 0], axis=0)

        if timepoint == 84:
            primary_outcome = clinical[event_title] == 1
        else:
            primary_outcome = np.all([clinical[duration_title].values <= timepoint,
                                      clinical[event_title] == 1], axis=0)
        primary_outcome = pd.Series(primary_outcome, index=clinical.index).astype(int)[validity] * 2
        negative_primary_patients = primary_outcome.index[primary_outcome == 0]

        secondary_endpoint = secondary_timepoints[timepoint]
        secondary_outcome = (clinical.loc[
                                 negative_primary_patients, secondary_endpoint].dropna() > secondary_endpoint_threshold).astype(
            'Int8')
        secondary_outcome = secondary_outcome.reindex(negative_primary_patients)
        primary_outcome[negative_primary_patients] = secondary_outcome
        primary_outcome.name = str(timepoint) + 'm Progression'
        outcomes.append(primary_outcome)
    outcomes.append(clinical[[duration_title, event_title]])
    outcomes = pd.concat(outcomes, axis=1)
    return outcomes


class SurvivalStratification:
    def __init__(self, preprocessed_feature_directory, model_directory, baseline_factor, stratification_timepoint,
                 duration_title, event_title, survival_timepoints, secondary_timepoints, group_names=None):
        self.preprocessed_feature_directory = preprocessed_feature_directory
        self.model_directory = model_directory
        self.baseline_factor = baseline_factor
        self.duration_title = duration_title
        self.event_title = event_title
        self.survival_timepoints = survival_timepoints
        self.secondary_endpoints = secondary_timepoints
        self.stratification_timepoint = stratification_timepoint
        self.group_names = group_names
        self.training_scores = None
        self.training_outcomes = None
        self.testing_scores = None
        self.testing_outcomes = None
        self.model = None

    def load_training_data(self):
        training_prediction_directory = os.path.join(self.model_directory, 'training_predictions.csv')
        self.training_scores = pd.read_csv(training_prediction_directory, index_col=0)['Risk score']
        training_clinical = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'training_clinical.csv'),
                                        index_col=0)
        self.training_outcomes = progression_label_generation(training_clinical, self.baseline_factor,
                                                              self.duration_title, self.event_title,
                                                              self.survival_timepoints, self.secondary_endpoints,
                                                              baseline_threshold=0)

    def load_testing_data(self):
        testing_prediction_directory = os.path.join(self.model_directory, 'testing_predictions.csv')
        self.testing_scores = pd.read_csv(testing_prediction_directory, index_col=0)['Risk score']
        testing_clinical = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'testing_clinical.csv'),
                                       index_col=0)
        self.testing_outcomes = progression_label_generation(testing_clinical, self.baseline_factor,
                                                             self.duration_title, self.event_title,
                                                             self.survival_timepoints, self.secondary_endpoints,
                                                             baseline_threshold=0)

    def model_training(self, export_directory=None):
        baseline_positive_outcomes = self.training_outcomes[self.training_outcomes['Baseline'] == 1]
        baseline_negative_outcomes = self.training_outcomes[self.training_outcomes['Baseline'] == 0]
        progression_title = str(self.stratification_timepoint) + 'm Progression'
        baseline_positive_progression = baseline_positive_outcomes[progression_title].dropna()
        baseline_negative_progression = baseline_negative_outcomes[progression_title].dropna()

        scores = self.training_scores.loc[baseline_positive_progression.index]
        baseline_positive_primary = (baseline_positive_progression == 2).astype(int)
        primary_cutoff = optimum_binary_cutoff_search(scores, baseline_positive_primary)
        print('Baseline positive primary cutoff: {0}'.format(primary_cutoff))

        scores = self.training_scores.loc[baseline_negative_progression.index]
        baseline_negative_secondary = (baseline_negative_progression > 0).astype(int)
        secondary_cutoff = optimum_binary_cutoff_search(scores, baseline_negative_secondary)
        print('Baseline negative secondary cutoff: {0}'.format(secondary_cutoff))

        if self.group_names is None:
            group_names = [0, 1, 2]
        else:
            group_names = self.group_names
        self.model = {
            group_names[0]: [[(secondary_cutoff, True)]],
            group_names[1]: [[(secondary_cutoff, False)],
                             [(primary_cutoff, True)]],
            group_names[2]: [[(primary_cutoff, False)]]
        }
        if export_directory is not None:
            model_export = json.dumps(self.model)
            # open file for writing, "w"
            f = open(os.path.join(export_directory, "model.json"), "w")
            # write json object to file
            f.write(model_export)
            # close file
            f.close()

    def model_prediction(self, scores):
        predictions = pd.Series(index=scores.index, dtype=object, name='Prediction')
        for group_name, criteria_group in self.model.items():
            print(group_name)
            selected_scores_outcomes = scores
            for criteria in criteria_group:
                for threshold, less in criteria:
                    if less:
                        selected_scores_outcomes = selected_scores_outcomes[selected_scores_outcomes < threshold]
                    else:
                        selected_scores_outcomes = selected_scores_outcomes[selected_scores_outcomes >= threshold]
            print(len(selected_scores_outcomes.index))
            predictions[selected_scores_outcomes.index] = group_name
        print(predictions.unique())
        return predictions

    def model_performance(self, export_directory=None, baseline_renames=None, progression_renames=None):
        training_predictions = self.model_prediction(self.training_scores)
        training_predictions = pd.concat([self.training_outcomes, self.training_scores, training_predictions], axis=1)
        if baseline_renames is not None:
            training_predictions['Baseline'] = training_predictions['Baseline'].astype(str).replace(baseline_renames)
        if progression_renames is not None:
            progression_titles = [str(x) + 'm Progression' for x in self.survival_timepoints]
            training_predictions[progression_titles] = training_predictions[progression_titles].astype(str).replace(
                progression_renames)
        if export_directory is not None:
            training_predictions.to_csv(os.path.join(export_directory, 'training_predictions.csv'))

        testing_predictions = self.model_prediction(self.testing_scores)
        testing_predictions = pd.concat([self.testing_outcomes, self.testing_scores, testing_predictions],
                                        axis=1)
        if baseline_renames is not None:
            testing_predictions['Baseline'] = testing_predictions['Baseline'].astype(str).replace(
                baseline_renames)
        if progression_renames is not None:
            progression_titles = [str(x) + 'm Progression' for x in self.survival_timepoints]
            testing_predictions[progression_titles] = testing_predictions[progression_titles].astype(str).replace(
                progression_renames)
        if export_directory is not None:
            testing_predictions.to_csv(os.path.join(export_directory, 'testing_predictions.csv'))

        x_ticks = np.arange(8) * 12
        x_range = (0, 84)
        rows_to_show = ['At risk']

        plt.figure(figsize=(4, 4))
        ax = plt.subplot(111)
        kmfs = []

        for group_label in self.model.keys():
            kmf = KaplanMeierFitter()
            group_index = training_predictions['Prediction'] == group_label
            ax = kmf.fit(training_predictions.loc[group_index, self.duration_title].values,
                         training_predictions.loc[group_index, self.event_title].values,
                         label=group_label).plot_survival_function(ax=ax, ci_show=True)
            kmfs.append(kmf)
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('KRFS rate')
        ax.set_ylim([0, 1])
        ax.set_xlim(x_range)
        add_at_risk_counts(*kmfs, xticks=x_ticks, ax=ax, rows_to_show=rows_to_show)
        plt.tight_layout()

        if export_directory is not None:
            plt.savefig(os.path.join(export_directory, 'training_km_curves.png'), dpi=300)
            plt.savefig(os.path.join(export_directory, 'training_km_curves.pdf'))
        plt.show()
        plt.clf()

        plt.figure(figsize=(4, 4))
        ax = plt.subplot(111)
        kmfs = []
        for group_label in self.model.keys():
            kmf = KaplanMeierFitter()
            group_index = testing_predictions['Prediction'] == group_label
            ax = kmf.fit(testing_predictions.loc[group_index, self.duration_title].values,
                         testing_predictions.loc[group_index, self.event_title].values,
                         label=group_label).plot_survival_function(ax=ax, ci_show=True)
            kmfs.append(kmf)
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('KRFS rate')
        ax.set_ylim([0, 1])
        ax.set_xlim(x_range)
        add_at_risk_counts(*kmfs, xticks=x_ticks, ax=ax, rows_to_show=rows_to_show)
        plt.tight_layout()

        if export_directory is not None:
            plt.savefig(os.path.join(export_directory, 'testing_km_curves.png'), dpi=300)
            plt.savefig(os.path.join(export_directory, 'testing_km_curves.pdf'))
        plt.show()
        plt.clf()

    def execute(self, export_directory, baseline_renames=None, progression_renames=None):
        self.load_training_data()
        self.load_testing_data()
        self.training_outcomes.to_csv(os.path.join(export_directory, 'training_outcomes.csv'))
        self.testing_outcomes.to_csv(os.path.join(export_directory, 'testing_outcomes.csv'))
        # print(self.training_outcomes)
        self.model_training(export_directory)
        self.model_performance(export_directory, baseline_renames=baseline_renames,
                               progression_renames=progression_renames)


if __name__ == '__main__':
    duration_title = 'LKR duration'
    event_title = 'LKR event'
    survival_timepoint = 84
    feature_selection_parameters = {
        'Confounding factors': ['original_shape2D_PixelSurface'],
        'Excluding keywords': ['shape'],
        'MRMR feature number': 10
    }
    preprocessing_feature_directory = r"../preprocessed_features"
    export_directory = r"../modeling"
    for i in range(3):
        for j in range(3):
            region_name = 'BBLocal{0}{1}'.format(i, j)
            # if region_name != 'BBLocal20':
            #     continue
            regional_export_directory = os.path.join(export_directory, region_name)
            if not os.path.exists(regional_export_directory):
                os.mkdir(regional_export_directory)
            print('-------------------------{0}--------------------------'.format(region_name))
            regional_modeling = RadScoreDevelopment(preprocessing_feature_directory, region_name, duration_title,
                                                    event_title, survival_timepoint,
                                                    feature_selection_parameters=feature_selection_parameters)
            regional_modeling.execute(regional_export_directory)

    region_names = ['BBLocal20']
    performance_parameters = {
        'c-index': None,
        'auc': {'times': np.array([15, 30, 60, 84])},
        'ibs': {'times': np.array([15, 30, 60, 84])},
        'Representative time points': [30, 60, 84]
    }

    clinical_factors = []
    final_model_export_directory = os.path.join(export_directory, 'RadScore')
    if not os.path.exists(final_model_export_directory):
        os.mkdir(final_model_export_directory)
    final_modeling = RiskScoreDevelopment(preprocessing_feature_directory, export_directory, region_names,
                                          clinical_factors, duration_title, event_title,
                                          performance_parameters)
    final_modeling.execute(final_model_export_directory)

    clinical_factors = ['V0XLPFROA']
    final_model_export_directory = os.path.join(export_directory, 'PFOA')
    if not os.path.exists(final_model_export_directory):
        os.mkdir(final_model_export_directory)
    final_modeling = RiskScoreDevelopment(preprocessing_feature_directory, export_directory, [],
                                          clinical_factors, duration_title, event_title,
                                          performance_parameters)
    final_modeling.execute(final_model_export_directory)

    clinical_factors = ['V0WOPNKL']
    final_model_export_directory = os.path.join(export_directory, 'KLG')
    if not os.path.exists(final_model_export_directory):
        os.mkdir(final_model_export_directory)
    final_modeling = RiskScoreDevelopment(preprocessing_feature_directory, export_directory, [],
                                          clinical_factors, duration_title, event_title,
                                          performance_parameters)
    final_modeling.execute(final_model_export_directory)

    clinical_factors = ['AGE', 'SEX', 'V0BMI', 'V0WOPNKL', 'V0XLKL']
    final_model_export_directory = os.path.join(export_directory, 'KRRiskScore')
    if not os.path.exists(final_model_export_directory):
        os.mkdir(final_model_export_directory)
    final_modeling = RiskScoreDevelopment(preprocessing_feature_directory, export_directory, region_names,
                                          clinical_factors, duration_title, event_title,
                                          performance_parameters)
    final_modeling.execute(final_model_export_directory)

    model_directory = os.path.join(export_directory, 'KRRiskScore')
    primary_endpoint = '60m KR'
    secondary_endpoint = '60m PFOA'
    survival_timepoints = [30, 60, 84]
    secondary_timepoints = {
        30: 'V2XLPFROA',
        60: 'V3XLPFROA',
        84: 'V5XLPFROA'
    }
    clinical_renames = {
        'V0XLPFROA': '0m PFOA',
        'V2XLPFROA': '30m PFOA',
        'V3XLPFROA': '60m PFOA',
        'V5XLPFROA': '84m PFOA',

    }
    baseline_factor = 'V0XLPFROA'
    baseline_renames = {
        '0': 'PFOA-',
        '1': 'PFOA+'
    }
    progression_renames = {
        '0': 'PFOA-',
        '1': 'PFOA+',
        '2': 'KR+'
    }
    group_names = ['Low risk', 'Medium risk', 'High risk']
    stratification_timepoint = 60
    stratification_export_directory = os.path.join(export_directory, 'Stratification')
    if not os.path.exists(stratification_export_directory):
        os.mkdir(stratification_export_directory)
    survival_stratification = SurvivalStratification(preprocessing_feature_directory, model_directory, baseline_factor,
                                                     stratification_timepoint,
                                                     duration_title, event_title, survival_timepoints,
                                                     secondary_timepoints, group_names=group_names)
    survival_stratification.execute(stratification_export_directory, baseline_renames=baseline_renames,
                                    progression_renames=progression_renames)
