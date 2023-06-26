import SimpleITK as sitk
import csv


def load_image(image_path):
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)

    return sitk.JoinSeries(image)


def load_dummy_mask(mask_path=r"C:\Users\glosk\Desktop\AGH\Stopien II\Semestr_10\IO\mask-1.png"):
    mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)

    return sitk.JoinSeries(mask)  # 3D image needed for pyradiomics


def read_file_list_from_csv(input_csv):
    flist = []
    try:
        with open(input_csv, 'r') as inFile:
            cr = csv.DictReader(inFile, lineterminator='\n')
            flist = [row for row in cr]
    except Exception:
        pass

    return flist


def update_output_file(feature_vector, output_filepath, write_headers=False):
    with open(output_filepath, 'a') as outputFile:
        writer = csv.writer(outputFile, lineterminator='\n')
        headers = list(feature_vector.keys())
        if write_headers:
            writer.writerow(headers)

        row = []
        for h in headers:
            row.append(feature_vector.get(h, "N/A"))
        writer.writerow(row)
