from PIL import Image
from pathlib import Path
import numpy as np

from feature_extractor4096 import FeatureExtractor4096

if __name__ == '__main__':
    fe = FeatureExtractor4096()

    for img_path in sorted(Path("./static/CitraBatik_dataset/train").glob("*.jpg")):
        print(img_path)
        #Extract a depp feature here



        feature = fe.extract(img=Image.open(img_path))
        # print(type(feature), feature.shape)

        feature_path = Path("./static/feature4096")/ (img_path.stem + ".npy")
        print(feature_path)

        #Save the feature
        np.save(feature_path,feature)