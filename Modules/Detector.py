import os
import pandas as pd
import numpy as np
import torch
import torchvision
from Modules.DataSet import DataSet, DetectDataSet
from Modules.FileManager import ProjectFileManager
from Modules.MLutils import collate_fn, Compose, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


class Detector:

    def __init__(self, pid):
        # initialize detector
        self.pid = pid
        self.pfm = ProjectFileManager(self.pid)
        self._initialize_model()
        self.detections = self._open_detections_csv()

    # def test(self, n_imgs):
    #     """run detection on a random set of n images from the test set.
    #
    #     Args:
    #     num: number of images to select
    #
    #     """
    #     dataset = DataSet(Compose([ToTensor()]), 'test')
    #     indices = list(range(len(dataset)))
    #     np.random.shuffle(indices)
    #     idx = indices[:n_imgs]
    #     sampler = SubsetRandomSampler(idx)
    #     loader = DataLoader(dataset, sampler=sampler, batch_size=n_imgs, collate_fn=collate_fn)
    #     self.evaluate(loader)

    def detect(self):
        """run detection on the images contained in img_dir

        Args:
            img_dir (str): path to the image directory, relative to data_dir (see FileManager)
        """
        img_dir = self.pfm.local_paths['image_dir']
        assert os.path.exists(img_dir)
        img_files = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)]
        dataset = DetectDataSet(Compose([ToTensor()]), img_files)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                collate_fn=collate_fn)
        self.evaluate(dataloader)

    def _open_detections_csv(self):
        if os.path.exists(self.pfm.local_paths['detections_csv']):
            df = pd.read_csv(self.pfm.local_paths['detections_csv'])

        else:
            return pd.DataFrame(columns=['Framefile', 'boxes', 'labels', 'scores']).set_index('Framefile')

    def _initialize_model(self):
        """initiate the model, optimizer, and scheduler."""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.load_state_dict(torch.load(self.pfm.local_paths['weights_file']))
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.model.load_state_dict(torch.load(self.pfm.local_paths['weights_file'], map_location=self.device))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        """evaluate the model on the detect set of images"""
        cpu_device = torch.device("cpu")
        self.model.eval()
        results = {}
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        df = pd.DataFrame.from_dict(results, orient='index')
        index_list = df.index.tolist()
        detect_framefiles = []
        for i in index_list:
            detect_framefiles.append(dataloader.dataset.img_files[i])
        df['Framefile'] = [os.path.basename(path) for path in detect_framefiles]
        df = df[['Framefile', 'boxes', 'labels', 'scores']].set_index('Framefile')
        self.detections.append(df)
        self.detections.to_csv(self.pfm.local_paths['detections_csv'])


class ContinuousDetector:

    def __init__(self, pids):
        self.pids = pids
        self.detectors = {pid: Detector(pid) for pid in pids}
