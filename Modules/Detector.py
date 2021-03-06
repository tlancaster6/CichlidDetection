import datetime
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import cv2
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

    def img_detect(self):
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

    def vid_detect(self, video_ids=None, sampling_rate=None, save_images=True, frame_range=None):
        """run detection on an mp4 video
        Args:
            video_ids: list of ints, corresponding to the videos to be analyzed. For example, [1] corresponds to
                    '0001_vid.mp4'. Pass None (default) to analyze all videos in the video_dir
            sampling_rate: int, corresponding to the number of frames per second analyze. Set to None (default) to
                    analyze all frames. This is equivalent to setting sampling_rate equal to the video frame-rate.
            save_images: if True (default) save each frame as a .jpg file to the images directory
            frame_range: iterable of two ints, specifying the frame numbers of the first and last frame to be analyzed. in
                    each video. Set to 'None' (default) to analyze each video in its entirety.
        """
        # find the paths to the videos to analyze and confirm the files exist
        start = datetime.datetime.now()
        available_videos = os.listdir(self.pfm.local_paths['video_dir'])
        available_videos = [f for f in available_videos if f.endswith('.mp4')]
        if video_ids is None:
            video_paths = available_videos
        else:
            video_ids = video_ids if type(video_ids) is list else [video_ids]
            video_paths = ['{:04}_vid.mp4'.format(x) for x in video_ids]
        video_paths = [os.path.join(self.pfm.local_paths['video_dir'], x) for x in video_paths]
        for path in video_paths:
            assert os.path.exists(path)

        for path in video_paths:
            cap = cv2.VideoCapture(path)
            fps = cap.get(5)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            inc = 1 if sampling_rate is None else int(round(fps/sampling_rate))
            count = 0 if frame_range is None else frame_range[0]
            count_max = frame_count if frame_range is None else frame_range[1]
            n_analyzed = 0
            fnames = []
            frames = []
            while cap.isOpened() and (count <= count_max):
                ret, frame = cap.read()
                if ret:
                    fname = '_'.join([self.pid, os.path.basename(path).split('.')[0], '{:07}'.format(count)]) + '.jpg'
                    fname = os.path.join(self.pfm.local_paths['image_dir'], fname)
                    if save_images:
                        cv2.imwrite(fname, frame)
                    fnames.append(fname)
                    frames.append(frame)
                    if len(frames) == 150:
                        dataset = DetectDataSet(Compose([ToTensor()]), fnames, frames)
                        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                            collate_fn=collate_fn)
                        self.evaluate(dataloader, save_detections=False)
                        fnames = []
                        frames = []
                        n_analyzed += 150
                    count += inc
                    cap.set(1, count)
                else:
                    if len(frames) > 0:
                        dataset = DetectDataSet(Compose([ToTensor()]), fnames, frames)
                        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                            collate_fn=collate_fn)
                        self.evaluate(dataloader, save_detections=False)
                        n_analyzed += len(frames)
                    cap.release()
                    break
        self.save_detections()
        t_elapsed = (datetime.datetime.now() - start).total_seconds()
        frames_per_second = n_analyzed / t_elapsed
        print('detection time = {} seconds = {} frames per second'.format(t_elapsed, frames_per_second))

    def _open_detections_csv(self):
        if os.path.exists(self.pfm.local_paths['detections_csv']):
            return pd.read_csv(self.pfm.local_paths['detections_csv'], index_col='Framefile')

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
    def evaluate(self, dataloader: DataLoader, save_detections=True):
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
        self.detections = self.detections.append(df)
        if save_detections:
            self.save_detections()

    def save_detections(self):
        self.detections = self.detections[~self.detections.index.duplicated()]
        self.detections.sort_index(inplace=True)
        self.detections.to_csv(self.pfm.local_paths['detections_csv'])
