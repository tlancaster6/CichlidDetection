import os
from Modules.DataPreppers import TrainDataPrepper, DetectDataPrepper
from Modules.FileManager import FileManager
from Modules.Trainer import Trainer
from Modules.Detector import Detector, ContinuousDetector
from Modules.Annotator import Annotator


class Runner:
    """user-friendly class for accessing the majority of module's functionality."""
    def __init__(self, training=False, pids=None):
        """initiate the Runner class"""
        self.fm = FileManager(training=training)
        if training:
            self.dp = TrainDataPrepper()
        else:
            self.dp = DetectDataPrepper(pids)
            self.pids = self.dp.pids
        self.pfms = self.dp.pfms
        self.tr = None
        self.de = None
        self.cde = None
        self.ann = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def download_all(self):
        """download all required data."""
        self.dp.download_all()

    def annotate(self, dry=False):
        for pid, pfm in self.pfms.items():
            self.ann = Annotator(pfm)
            if not dry:
                pfm.upload()

    def prep(self):
        """prep downloaded data"""
        self.dp.prep()

    def train(self, num_epochs, upload_results=True):
        """initiate a Trainer object and train the model.

        Args:
            num_epochs (int): number of epochs to train
            upload_results(bool): if True, automatically upload the results (weights, logs, etc.) after training
        """
        self.tr = Trainer(num_epochs, upload_results)
        self.tr.train()

    def sync(self):
        self.fm.sync_training_dir()
        self.fm.sync_model_dir()

    def detect(self):
        for pid in self.pids:
            self.de = Detector(pid)
            self.de.detect()

    def continuous_detect(self):
        self.cde = ContinuousDetector(self.pids)
