import logging
import os

import radiomics

from loader import read_file_list_from_csv, update_output_file
import utils

###############################################################
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# PROJECT_ROOT = r"C:\Users\glosk\Desktop\AGH\Stopien II\Semestr_10\IO"

COM_ROOT = PROJECT_ROOT + r"\com"
csv_res = COM_ROOT + r"\radiomics_features.csv"
output_filepath = os.path.join(COM_ROOT, 'radiomics_features_selected.csv')
progress_filename = os.path.join(COM_ROOT, f'pyrad_{os.path.basename(__file__)}.txt')

VERBOSE = True
###############################################################
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
###############################################################


def main():
    # load preselected data
    crs = read_file_list_from_csv(csv_res, logger)

    print("Len in:", len(crs))

    new_entries = []
    for idx, entry in enumerate(crs, 1):
        # print(f"Processing: {idx}/{len(crs)}")  # for debug purpose

        entry = utils.pop_obsolete_entries(entry)

        del entry['Mask']  # no longer needed

        try:
            if float(entry['NV']) == 0. and float(entry['MEL']) == 0.:
                # print("Skipping not NV nor MEL")
                continue
        except ValueError:
            continue
        new_entries.append(entry)

    print("Len out:", len(new_entries))

    # save fine-tuned data
    is_header = True
    for entry in new_entries:
        update_output_file(entry, output_filepath, write_headers=is_header)
        if is_header:
            is_header = False


if __name__ == '__main__':
    main()
