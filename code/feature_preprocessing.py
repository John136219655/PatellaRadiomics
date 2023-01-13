import os

import pandas as pd


def institution_train_test_split(clinical_feature_table, radiomics_feature_tables, export_directory,
                                 validity_column=None):
    if validity_column is not None:
        common_patients = clinical_feature_table[validity_column].dropna().index
    else:
        common_patients = clinical_feature_table.dropna().index
    for radiomics_feature_table in radiomics_feature_tables.values():
        common_patients = common_patients.intersection(radiomics_feature_table.index)
    print('{0} common patients found.'.format(len(common_patients)))
    clinical_feature_table = clinical_feature_table.reindex(common_patients)

    training_patients = clinical_feature_table.index[clinical_feature_table['SITE'] == 2]
    testing_patients = clinical_feature_table.index[clinical_feature_table['SITE'] == 1]

    training_clinical = clinical_feature_table.loc[training_patients, :]
    testing_clinical = clinical_feature_table.loc[testing_patients, :]
    training_clinical.to_csv(os.path.join(export_directory, 'training_clinical.csv'))
    testing_clinical.to_csv(os.path.join(export_directory, 'testing_clinical.csv'))

    for feature_table_name, radiomics_feature_table in radiomics_feature_tables.items():
        training_radiomics_table = radiomics_feature_table.reindex(training_patients)
        training_radiomics_table.to_csv(os.path.join(export_directory, 'training_' + feature_table_name + '.csv'))
        testing_radiomics_table = radiomics_feature_table.reindex(testing_patients)
        testing_radiomics_table.to_csv(os.path.join(export_directory, 'testing_' + feature_table_name + '.csv'))


def feature_preprocessing_pipeline(feature_extraction_directory, feature_preprocessing_directory, feature_names,
                                   clinical_filename, validity_title=None):
    clinical_feature_filepath = os.path.join(feature_preprocessing_directory, clinical_filename)
    clinical_feature_table = pd.read_csv(clinical_feature_filepath, index_col=0)
    radiomcis_feature_tables = dict()
    for name in feature_names:
        feature_table_filepath = os.path.join(feature_extraction_directory, name + '.csv')
        if not os.path.isfile(feature_table_filepath):  # or not '.csv' in name:
            continue
        radiomics_table = pd.read_csv(feature_table_filepath, index_col=0)
        radiomcis_feature_tables[name] = radiomics_table
    institution_train_test_split(clinical_feature_table, radiomcis_feature_tables, feature_preprocessing_directory,
                                 validity_column=validity_title)


if __name__ == '__main__':
    feature_extraction_directory = r"../feature_extraction"
    feature_preprocessing_directory = r"../preprocessed_features"
    clinical_filename = 'clinical_complete.csv'
    feature_names = []
    for i in range(3):
        for j in range(3):
            feature_names.append('BBLocal{0}{1}'.format(i, j))
    feature_preprocessing_pipeline(feature_extraction_directory, feature_preprocessing_directory, feature_names,
                                   clinical_filename,
                                   validity_title=['LKR duration'])
