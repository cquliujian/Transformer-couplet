import argparse
import json
import os
import tensorflow as tf

hparams_path = r"./data/hparams.json"

class HParams:
    def __init__(self):
        """
        Args:
            model_dir: Name of the folder storing the hparams.json file.
        """
        self.hparams_path = r"./data/hparams.json"
        self.hparams = self.load_hparams(self.hparams_path)

    @staticmethod
    def load_hparams(model_dir):
        """Load hparams from an existing directory."""
        hparams_file = model_dir
        if os.path.exists(hparams_file):  # 判断该文件是否存在
            # print("# Loading hparams from {} ...".format(hparams_file))
            with open(model_dir) as f:
                try:
                    hparams_values = json.load(f)
                    hparams = tf.contrib.training.HParams(**hparams_values)
                except ValueError:
                    print("Error loading hparams file.")
                    return None
            return hparams
        else:
            return None


if __name__ == "__main__":

    hp = HParams()
    print(hp.hparams.mode)



