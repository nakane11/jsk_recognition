import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), './torchreid/torchreid'))

import models
build_model = models.build_model
