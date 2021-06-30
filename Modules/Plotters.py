import os

import cv2

from Modules.FileManager import FileManager, ProjectFileManager
from Modules.Utils import xyminmax_to_xywh
from os.path import join, exists
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps, partial
from PIL import Image
import numpy as np
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def plotter_decorator(plotter_method=None, save=True):
    """decorator used for automatic set-up and clean-up when making figures with methods from Plotter derived classes"""
    if plotter_method is None:
        return partial(plotter_decorator, save=save)

    @wraps(plotter_method)
    def wrapper(plotter, fig=None, *args, **kwargs):
        method_name = plotter_method.__name__
        fig = plt.figure(*args, **kwargs) if fig is None else fig
        plotter_method(plotter, fig)
        if save:
            plotter.save_fig(fig, method_name)
    return wrapper


class Plotter:

    def __init__(self):
        self.fm = FileManager()
        self.training_figure_dir = self.fm.local_paths['training_figure_dir']
        self.training_figure_data_dir = join(self.training_figure_dir, 'training_figure_data_dir')


    def save_fig(self, fig: Figure, file_stub: str):
        """save the figure as a pdf and close it

        Notes:
            saves the figure to the figure_dir specified in the FileManager object

        Args:
            fig (Figure): figure to save
            file_stub (str): name to use for the file. Don't include '.pdf'
        """
        fig.savefig(join(self.training_figure_dir, '{}.pdf'.format(file_stub)))
        plt.close('all')


class TrainPlotter(Plotter):

    def __init__(self):
        super().__init__()
        self._load_data()

    def plot_all(self):
        """create pdf's of every plot this class can produce"""
        self.total_loss_vs_epoch()
        self.n_boxes_vs_epoch()
        self.animated_learning()
        self.iou_vs_epoch()
        self.final_epoch_eval()

    @plotter_decorator
    def total_loss_vs_epoch(self, fig: Figure):
        """plot the training loss vs epoch and save as loss_vs_epoch.pdf

        Args:
            fig (Figure): matplotlib Figure object into which to plot
        """
        ax = fig.add_subplot(111)
        ax.set(xlabel='epoch', ylabel='total loss', title='Training Loss vs. Epoch')
        sns.lineplot(data=self.train_log.loss_total, ax=ax)
        self.train_log.loc[:, ['loss_total']].to_csv(join(self.training_figure_data_dir, 'total_loss_vs_epoch.csv'))

    @plotter_decorator
    def n_boxes_vs_epoch(self, fig: Figure):
        """plot the average number of boxes predicted per frame vs the epoch"""
        predicted = pd.Series([df.boxes.apply(len).agg('mean') for df in self.epoch_predictions])
        actual = pd.Series([self.ground_truth.boxes.apply(len).agg('mean')] * len(predicted))
        ax = fig.add_subplot(111)
        ax.set(xlabel='epoch', ylabel='avg # detections', title='Average Number of Detections vs. Epoch')
        sns.lineplot(data=predicted, ax=ax, label='predicted')
        sns.lineplot(data=actual, ax=ax, label='actual')
        df = pd.DataFrame({'predicted': predicted, 'actual': actual})
        df.to_csv(join(self.training_figure_data_dir, 'n_boxes_vs_epoch.csv'))

    @plotter_decorator(save=False)
    def animated_learning(self, fig: Figure):
        """for a single frame, successively plot the predicted boxes and labels at each epoch to create an animation"""

        # find a frame with a good balance of number of fish and final-epoch score for each box, and load that image
        final_epoch = self.epoch_predictions[-1].copy()
        final_epoch['n_detections'] = final_epoch['labels'].apply(len)
        final_epoch['min_score'] = final_epoch['scores'].apply(lambda x: 0 if len(x) is 0 else min(x))
        final_epoch = final_epoch[final_epoch.min_score > 0.95]
        frame = final_epoch.sort_values(by=['n_detections', 'min_score'], ascending=False).iloc[0].name
        im = np.array(Image.open(join(self.fm.local_paths['test_image_dir'], frame)), dtype=np.uint8)

        # build up the animation
        max_detections = 5
        ax = fig.add_subplot(111)
        plt.xlim(0, im.shape[1])
        plt.ylim(im.shape[0], 0)
        boxes = [Rectangle((0, 0), 0, 0, fill=False) for _ in range(max_detections)]

        def init():
            for box in boxes:
                ax.add_patch(box)
            return boxes

        def animate(i):
            label_preds = self.epoch_predictions[i].loc[frame, 'labels']
            label_preds = (label_preds + ([0] * max_detections))[:5]
            box_preds = self.epoch_predictions[i].loc[frame, 'boxes']
            box_preds = [xyminmax_to_xywh(*p) for p in box_preds]
            box_preds = (box_preds + ([[0, 0, 0, 0]] * max_detections))[:5]
            color_lookup = {0: 'None', 1: '#FF1493', 2: '#00BFFF'}
            for j in range(5):
                boxes[j].set_xy(xy=(box_preds[j][0], box_preds[j][1]))
                boxes[j].set_width(box_preds[j][2])
                boxes[j].set_height(box_preds[j][3])
                boxes[j].set_edgecolor(color_lookup[label_preds[j]])
            return boxes

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.epoch_predictions), blit=True, interval=200,
                             repeat=False)
        ax.imshow(im, zorder=0)
        anim.save(join(self.training_figure_dir, 'animated_learning.gif'), writer='imagemagick')
        plt.close('all')

    @plotter_decorator
    def iou_vs_epoch(self, fig: Figure):
        ious = []
        for ep in range(len(self.epoch_predictions)):
            ious.append(self._calc_epoch_iou(ep))
        ax = fig.add_subplot(111)
        ax.set(xlabel='epoch', ylabel='average iou', title='IOU score vs. Epoch')
        sns.lineplot(data=pd.Series(ious), ax=ax)
        pd.DataFrame({'iou': ious}).to_csv(join(self.training_figure_data_dir, 'iou_vs_epoch.csv'))

    @plotter_decorator
    def final_epoch_eval(self, fig: Figure):
        fig.set_size_inches(11, 8.5)
        epoch_index = len(self.epoch_predictions) - 1
        df, summary = self._full_epoch_eval(epoch_index)
        ######
        df = df.reset_index()
        # No. of frames w/ error vs no. of frames w/o error
        no_err_val = df[df.n_boxes_predicted_error == 0].count()['Framefile']
        err_val = df[df.n_boxes_predicted_error != 0].count()['Framefile']
        # No. of frames w/ positive error & negative error
        pos = df[df.n_boxes_predicted_error > 0].count()['Framefile']
        neg = df[df.n_boxes_predicted_error < 0].count()['Framefile']
        # Distribution of no. of frames per error value
        df1 = df.groupby(df.n_boxes_predicted_error).count()['Framefile']
        ### graphs ###
        ax1 = fig.add_subplot(1, 2, 1)  # top and bottom left
        ax2 = fig.add_subplot(2, 2, 2)  # top right
        ax3 = fig.add_subplot(2, 2, 4)  # bottom right

        ax1.bar(x=df.n_boxes_predicted_error.unique(), height=df1, color='darkblue')
        ax1.set_xlabel('Error Score')
        ax1.set_ylabel('No. of Frames')
        ax1.set_title("Distribution of No. of Frames Per Error Value")

        ax2.bar(x=['No Error', 'Error'], height=[no_err_val, err_val], color=['green', 'red'], width=0.4)
        ax2.set_ylabel('No. of Framefiles')
        ax2.set_title('No. of Frames With Error vs Without Error')

        ax3.bar(x=['Overestimation', 'Underestimation'], height=[pos, neg], color='red', width=0.4)
        ax3.set_title('Analysis of Errors')
        ax3.set_ylabel('No. of Frames')

    def _load_data(self):
        """load and parse all relevant data. Automatically syncs training dir with cloud if any files are missing"""
        required_files = [self.fm.local_paths[x] for x in ['boxed_fish_csv', 'train_log']]
        required_files.append(join(self.fm.local_paths['predictions_dir'], '0.csv'))
        for f in required_files:
            if not exists(f):
                self.fm.sync_training_dir(exclude=['labels/**', 'train_images/**'])
                break
        self.train_log = self._parse_train_log()
        self.num_epochs = len(self.train_log)
        self.ground_truth = self._parse_epoch_csv()
        self.epoch_predictions = []
        for epoch in range(self.num_epochs):
            self.epoch_predictions.append(self._parse_epoch_csv(epoch))

    def _parse_train_log(self):
        """parse the logfile that tracked overall loss and learning rate at each epoch

        Returns:
            Pandas Dataframe of losses and learning rate, indexed by epoch number
        """
        return pd.read_csv(self.fm.local_paths['train_log'], sep='\t', index_col='epoch')

    def _parse_epoch_csv(self, epoch=-1):
        """parse the csv file of predictions produced when Trainer.train() is run with compare_annotations=True

        Notes:
            if the epoch arg is left at the default value of -1, this function will instead parse 'ground_truth.csv'

        Args:
            epoch(int): epoch number, where 0 refers to the first epoch. Defaults to -1, which parses the
                ground truth csv

        Returns:
            Pandas DataFrame of epoch data
        """
        if epoch == -1:
            path = self.fm.local_paths['ground_truth_csv']
            usecols = ['Framefile', 'boxes', 'labels']
        else:
            path = join(self.fm.local_paths['predictions_dir'], '{}.csv'.format(epoch))
            usecols = ['Framefile', 'boxes', 'labels', 'scores']
        return pd.read_csv(path, usecols=usecols).set_index('Framefile').applymap(lambda x: eval(x))

    def _full_epoch_eval(self, epoch):
        ep = self.epoch_predictions[epoch]
        gt = self.ground_truth
        df = gt.join(ep, lsuffix='_actual', rsuffix='_predicted')
        df['n_boxes_actual'] = df.boxes_actual.apply(len)
        df['n_boxes_predicted'] = df.boxes_predicted.apply(len)
        df['n_boxes_predicted_error'] = df.n_boxes_predicted - df.n_boxes_actual
        df['average_iou'], df['act_to_pred_map'] = zip(*df.apply(
            lambda x: self._calc_frame_iou(x.boxes_actual, x.boxes_predicted, map_boxes=True), axis=1))
        df['pred_to_act_map'] = df.apply(lambda x: self._flip_mapping(x.act_to_pred_map, x.n_boxes_predicted), axis=1)
        df['pred_accuracy'] = df.apply(
            lambda x: self._compare_labels(x.labels_actual, x.labels_predicted, x.pred_to_act_map), axis=1)
        df['avg_accuracy'] = df.pred_accuracy.apply(lambda x: sum(x) / len(x) if len(x) > 0 else 1.0)
        df.to_csv(join(self.training_figure_data_dir, 'epoch_{}_eval.csv'.format(epoch)))

        summary = pd.Series()
        summary['classification_accuracy'] = np.average(df.avg_accuracy, weights=df.n_boxes_predicted)
        summary['average_iou'] = np.average(df.average_iou, weights=df.n_boxes_predicted)
        summary['n_predictions'] = df.n_boxes_predicted.sum()
        summary['n_annotations'] = df.n_boxes_actual.sum()
        summary['n_frames'] = len(df)
        summary.to_csv(join(self.training_figure_data_dir, 'epoch_{}_eval_summary.csv'.format(epoch)))

        return df, summary

    def _compare_labels(self, labels_actual, labels_predicted, pred_to_act_map):
        """determine whether the label for each predicted box matches the label of the corresponding ground-truth box

        Args:
            labels_actual (list of ints): ground-truth label for each ground-truth box
            labels_predicted (list of ints): predicted label for each predicted box
            pred_to_act_map (list of ints): list mapping each predicted box to the ground truth box with the max iou

        Returns:
            list: outcomes, where outcomes[i] = 1 if labels_predicted[i] is correct, and 0 if it's incorrect
        """
        outcomes = []
        for i, pred in enumerate(labels_predicted):
            # append 0 if the predicted box does not overlap a ground truth box
            if pred_to_act_map[i] is None:
                outcomes.append(0)
            # else, append 1 if the predicted label was correct, or 0 if it was incorrect
            else:
                outcomes.append(1 if pred == labels_actual[pred_to_act_map[i]] else 0)
        return outcomes

    def _calc_precision(self):
        pass

    def _calc_recall(self):
        pass

    def _flip_mapping(self, a_to_b, len_b):
        if len_b == 0:
            return []
        else:
            mapping = []
            for i in range(len_b):
                try:
                    mapping.append(a_to_b.index(i))
                except ValueError:
                    mapping.append(None)
            return mapping

    def _calc_epoch_iou(self, epoch):
        """calculate the average iou across all test frames for a given epoch

        Args:
            epoch (int): epoch number

        Returns:
            float: average iou value per predicted box for the epoch
        """
        gt = self.ground_truth
        ep = self.epoch_predictions[epoch]
        combo = gt.join(ep, lsuffix='_gt', rsuffix='_ep')
        combo['frame_iou'] = combo.apply(lambda x: self._calc_frame_iou(x.boxes_gt, x.boxes_ep), axis=1)
        combo['n_boxes_ep'] = combo.boxes_ep.apply(len)
        return np.average(combo.frame_iou, weights=combo.n_boxes_ep)

    def _calc_frame_iou(self, actual_boxes, predicted_boxes, map_boxes=False):
        """calculate the average iou for a frame

        Args:
            actual_boxes (list of lists of 4 ints): list of ground truth bounding boxes
            predicted_boxes (list of lists of 4 ints): list of predicted bounding boxes
            map_boxes: if True, also return a list of ints mapping the ground truth boxes to the
                predicted box with the highest iou value

        Returns:
            if map_boxes is False, returns:
                float: mean iou value for the given frame
            if map_boxes is True, returns:
                float: mean iou value for the given frame
                list of ints: if map_boxes is True, also returns a list mapping the actual boxes to the
                    the predicted boxes, where actual_box[i] maps to predicted_boxes[mapping_list[i]]
                    (and predicted_box[j] maps to actual_boxes[mapping_list.index(j)]
                    if the actual_boxes list is empty, returns an empty list. for ground truth boxes that do not
                    intersect a predicted box, the list will contain a None element
        """
        a_bs = actual_boxes
        p_bs = predicted_boxes
        # if the model predicts no boxes for an empty frame, return a perfect score of 1.0 (and an empty mapping list)
        if (len(a_bs) == 0) and (len(p_bs) == 0):
            return (1.0, []) if map_boxes else 1.0

        # elif the model predicts no boxes for a frame with one or more fish, return a score of 0.0 (and mapping list of
        # null values with the same length as the number of actual boxes)
        elif (len(a_bs) > 0) and (len(p_bs) == 0):
            return (0.0, [None] * len(a_bs)) if map_boxes else 0.0

        # elif the model predicts 1 or more boxes for a frame with no fish, return a score of 0 (and an empty mapping
        # list)
        elif (len(a_bs) == 0) and (len(p_bs) > 0):
            return (0.0, []) if map_boxes else 0.0

        # elif the model predicted 1 or more boxes for a frame with 1 or more fish, calculate the iou for each
        # ground truth box with its best match, and average those values for the frame
        elif (len(a_bs) > 0) and (len(p_bs) > 0):
            ious = []
            mapping = []
            # for each actual box, find the largest iou score of that box with a one of the predicted boxes
            for a_b in a_bs:
                max_iou = 0.0
                max_iou_mapping = None
                for i, p_b in enumerate(p_bs):
                    iou = self._calc_iou(a_b, p_b)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_mapping = i
                ious.append(max_iou)
                mapping.append(max_iou_mapping)
            # if the model predicted more boxes than there are objects, penalize by scoring the remaining boxes 0.0
            if len(a_bs) < len(p_bs):
                ious.extend([0.0] * (len(p_bs) - len(a_bs)))
            # return the mean
            return (np.mean(ious), mapping) if map_boxes else np.mean(ious)

    def _calc_iou(self, box_a, box_b):
        """calculate the iou between box_a and box_b"""
        a = box_a
        b = box_b
        if (len(box_a) == 0) and (len(box_b) == 0):
            return 1.0
        # find area of the box formed by the intersection of box_a and box_b
        xa, ya, xb, yb = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        # if the boxes do not intersect, short-circuit and return 0.0
        if intersection == 0:
            return 0.0
        # else, calculate the area of the union of box_a and box_b, and return intersection/union
        else:
            a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
            b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
            union = float(a_area + b_area - intersection)
            return intersection / union


class DetectPlotter(Plotter):

    def __init__(self, pfm=None, pid=None):
        super().__init__()
        self.pfm = pfm if pfm is not None else ProjectFileManager(pid)
        self.image_dir = self.pfm.local_paths['image_dir']
        self.summary_dir = self.pfm.local_paths['summary_dir']
        self.detections = pd.read_csv(self.pfm.local_paths['detections_csv'], index_col='Framefile')

    def animated_detections(self, video_id, frame_range=None, framerate=None):
        image_paths = [join(self.image_dir, x) for x in os.listdir(self.image_dir)]
        image_paths = [x for x in image_paths if '{:04}_vid'.format(video_id) in x]
        if frame_range is not None:
            image_paths = [x for x in image_paths if int(x.split('_')[-1].split('.')[0]) >= frame_range[0]]
            image_paths = [x for x in image_paths if int(x.split('_')[-1].split('.')[0]) < frame_range[1]]
        image_paths = sorted(image_paths)
        image = cv2.imread(image_paths[0])
        height, width, _ = image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        framerate = 30 if framerate is None else framerate
        video_path = join(self.summary_dir, '{:04}_vid_detections.mp4'.format(video_id))
        video = cv2.VideoWriter(video_path, fourcc, framerate, (width, height))
        for p in image_paths:
            image = self.draw_boxes(p)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
        video.release()

    def draw_boxes(self, image_path, max_boxes=5, min_score=0.9):
        color_lookup = {0: 'None', 1: (255, 20, 147), 2: (0, 191, 255)}
        image = cv2.imread(image_path)
        framefile = os.path.basename(image_path)
        boxes, labels, scores = [eval(self.detections.loc[framefile, x]) for x in ['boxes', 'labels', 'scores']]
        boxes = [[int(np.round(coord)) for coord in box] for box in boxes]
        boxes = [x for i, x in enumerate(boxes) if scores[i] >= min_score]
        labels = [x for i, x in enumerate(labels) if scores[i] >= min_score]
        scores = [x for x in scores if x >= min_score]
        if len(boxes) > max_boxes:
            boxes, labels, scores = [x[:5] for x in [boxes, labels, scores]]
        for box, label in zip(boxes, labels):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color_lookup[label])
        return image


