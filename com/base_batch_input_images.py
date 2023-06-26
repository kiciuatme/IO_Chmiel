import collections
import logging
import os

import radiomics
from radiomics import featureextractor

from loader import load_image, load_dummy_mask, read_file_list_from_csv, update_output_file
import utils

PROJECT_ROOT = utils.get_project_root(__file__)

COM_ROOT = PROJECT_ROOT + r"\com"
PARAMS_FILE = COM_ROOT + r'\params.yaml'
BASE_DIR = PROJECT_ROOT + r"\ISIC2018_Task3_Training_Input"
inputCSV = PROJECT_ROOT + r"\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"

outputFilepath = os.path.join(COM_ROOT, 'radiomics_features.csv')
progress_filename = os.path.join(COM_ROOT, 'pyrad_log.txt')

VERBOSE = True


def main():
    # Configure logging
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger = logging.getLogger('radiomics')
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')
    if VERBOSE:
        radiomics.setVerbosity(logging.INFO)
    logger.info('Loading CSV')

    flists = read_file_list_from_csv(inputCSV, logger)

    logger.info('Loading Done')
    logger.info('Nof: %d', len(flists))

    ###############################################################
    if not os.path.isfile(PARAMS_FILE):
        exit(-1)

    extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS_FILE)

    # log info about extractor
    logger.info('input images types: %s', extractor.enabledImagetypes)
    logger.info('features: %s', extractor.enabledFeatures)
    logger.info('settings: %s', extractor.settings)

    write_headers = True
    for idx, entry in enumerate(flists, start=1):
        logger.info("(%d/%d) Processing Patient (Image: %s)", idx, len(flists), entry['image'])

        image_filepath = BASE_DIR + "\\" + entry['image'] + ".jpg"
        if not os.path.isfile(image_filepath):
            logger.warning('%s - not a file', image_filepath)
            continue

        # remove unused labels
        entry.pop('image')
        entry.pop('BCC')
        entry.pop('AKIEC')
        entry.pop('BKL')
        entry.pop('DF')
        entry.pop('VASC')

        try:
            if float(entry['NV']) == 0. and float(entry['MEL']) == 0.:
                logger.warning('%s not NV and not MEL', image_filepath)
                continue
        except ValueError:
            continue

        image = load_image(image_filepath)
        mask = load_dummy_mask()
        label = entry.get('Label', None)

        if str(label).isdigit():
            label = int(label)
        else:
            label = None

        feature_vector = collections.OrderedDict(entry)
        feature_vector['Image'] = os.path.basename(image_filepath)
        feature_vector['Mask'] = os.path.basename('dummy_mask')

        try:
            feature_vector.update(extractor.execute(image, mask, label))

            update_output_file(feature_vector, outputFilepath, write_headers)
            if write_headers:
                write_headers = False
        except Exception:
            logger.error('Feature extraction failed', exc_info=True)


if __name__ == '__main__':
    main()
