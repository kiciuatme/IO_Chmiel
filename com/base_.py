import six
from radiomics import featureextractor

from loader import load_image, load_dummy_mask


def main():
    """Example how to use custom input data such as image, mask and parameters file"""
    base_path = r"C:\Users\glosk\Desktop\AGH\Stopien II\Semestr_10\IO"

    path = base_path + r"\ISIC2018_Task3_Training_Input\ISIC_0024324.jpg"
    image3d = load_image(path)

    mask_name = base_path + r"\mask-1.png"
    mask3d = load_dummy_mask(mask_name)

    params = "./params.yaml"

    extractor = featureextractor.RadiomicsFeatureExtractor(params, verbose=True)

    result = extractor.execute(image3d, mask3d)
    for key, val in six.iteritems(result):
        print(f"\t{key}: {val}")


if __name__ == '__main__':
    main()
