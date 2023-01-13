import concurrent.futures as cf
import os
import timeit

import pandas as pd
import yaml
from radiomics import featureextractor
from tqdm import tqdm


def parse_from_yaml(filename):
    if not os.path.exists(filename):
        return
    with open(filename, 'r') as stream:
        try:
            result = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        return result


def radiomics_feature_extraction(preprocessed_data_directory, export_filepath, feature_extraction_parameter_filepath,
                                 cache_frequency=50, image_name=None, mask_name=None):
    image_feature_extraction_parameters = parse_from_yaml(feature_extraction_parameter_filepath)
    extractor = featureextractor.RadiomicsFeatureExtractor(image_feature_extraction_parameters)
    if mask_name is None:
        mask_name = 'mask'
    if image_name is None:
        image_name = 'image'
    feature_table = []
    existing_patient_ids = []
    patient_ids = [patient_id for patient_id in os.listdir(preprocessed_data_directory) if
                   patient_id not in existing_patient_ids]
    patient_num = len(patient_ids)
    starttime = timeit.default_timer()
    patient_index = 0
    failed_patients = []
    with tqdm(total=patient_num) as pbar:
        while patient_index < patient_num:
            patient_ending_index = patient_index + cache_frequency
            if patient_ending_index > patient_num:
                patient_ending_index = patient_num

            with cf.ProcessPoolExecutor(2) as executor:
                futures = dict()
                for patient_id in patient_ids[patient_index:patient_ending_index]:
                    patient_directory = os.path.join(preprocessed_data_directory, patient_id)
                    if not os.path.isdir(patient_directory):
                        continue
                    image_filepath = os.path.join(patient_directory, image_name + '.mha')
                    mask_filepath = os.path.join(patient_directory, mask_name + '.mha')
                    if not (os.path.exists(image_filepath) and os.path.exists(mask_filepath)):
                        continue
                    try:
                        future = executor.submit(extractor.execute, image_filepath, mask_filepath)
                        futures[future] = patient_id
                    except Exception as e:
                        print('Feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
                        failed_patients.append(patient_id)
                        pbar.update(1)
                for future in cf.as_completed(futures):
                    try:
                        feature_values = future.result()
                        patient_id = futures[future]
                        feature_series = pd.Series(feature_values, name=patient_id).astype(float)
                        feature_table.append(feature_series)
                    except Exception as e:
                        print('Feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
                        failed_patients.append(patient_id)
                    finally:
                        pbar.update(1)
            patient_index += cache_frequency
    print("The time difference is :", timeit.default_timer() - starttime)
    feature_table = pd.concat(feature_table, axis=1).T
    feature_table.to_csv(export_filepath)
    print('Finally, the following patients failed for radiomics feature extraction: {0}'.format(failed_patients))


if __name__ == '__main__':
    preprocessing_directory = r"../preprocessing"
    feature_extraction_directory = r"../feature_extraction"
    patient_batch_size = 2
    feature_extraction_parameter_filepath = os.path.join(feature_extraction_directory,
                                                         'image_feature_extraction_parameters.yaml')
    for i in range(3):
        for j in range(3):
            extracted_feature_filepath = os.path.join(feature_extraction_directory, 'BBLocal{0}{1}.csv'.format(i, j))
            radiomics_feature_extraction(preprocessing_directory, extracted_feature_filepath,
                                         feature_extraction_parameter_filepath,
                                         patient_batch_size, mask_name='ROI{0}{1}_mask'.format(i, j))
