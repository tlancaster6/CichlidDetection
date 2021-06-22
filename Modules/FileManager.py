import os, subprocess
from itertools import chain
from os.path import join
import pandas as pd
from Modules.Utils import run, make_dir


class FileManager:
    """Project non-specific class for handling local and cloud storage."""

    def __init__(self, training=False):
        """create an empty local_paths variable and run self._initialize()"""
        self.training = training
        self.local_paths = {}
        self.training_pids = None
        self._initialize()

    def sync_training_dir(self, exclude=None, quiet=False):
        """sync the training directory bidirectionally, keeping the newer version of each file.

        Args:
            exclude (list of str): files/directories to exclude. Accepts both explicit file/directory names and
                regular expressions. Expects a list, even if it's a list of length one. Default None.
            quiet: if True, suppress the output of rclone copy. Default False
        """
        print('syncing training directory')
        local_training_dir = self.local_paths['training_dir']
        cloud_training_dir = self._local_path_to_cloud_path(local_training_dir)
        down = ['rclone', 'copy', '-u', '-c', cloud_training_dir, local_training_dir]
        up = ['rclone', 'copy', '-u', '-c', local_training_dir, cloud_training_dir, '--exclude', '.*{/**,}']
        if not quiet:
            [com.insert(3, '-P') for com in [down, up]]
        if exclude is not None:
            [com.extend(list(chain.from_iterable(zip(['--exclude'] * len(exclude), exclude)))) for com in [down, up]]
        [run(com) for com in [down, up]]
        self.training_pids = pd.read_csv(self.local_paths['boxed_fish_csv'], index_col=0)['ProjectID'].unique()

    def sync_model_dir(self, exclude=None, quiet=False):
        """sync the machine learning directory bidirectionally, keeping the newer version of each file.

        Args:
            exclude (list of str): files/directories to exclude. Accepts both explicit file/directory names and
                regular expressions. Expects a list, even if it's a list of length one. Default None.
            quiet: if True, suppress the output of rclone copy. Default False
        """
        print('syncing training directory')
        local_model_dir = self.local_paths['model_dir']
        cloud_model_dir = self._local_path_to_cloud_path(local_model_dir)
        down = ['rclone', 'copy', '-u', '-c', cloud_model_dir, local_model_dir]
        up = ['rclone', 'copy', '-u', '-c', local_model_dir, cloud_model_dir, '--exclude', '.*{/**,}']
        if not quiet:
            [com.insert(3, '-P') for com in [down, up]]
        if exclude is not None:
            [com.extend(list(chain.from_iterable(zip(['--exclude'] * len(exclude), exclude)))) for com in [down, up]]
        [run(com) for com in [down, up]]

    def download(self, name, relative_path=None, overwrite=False):
        """use rclone to download a file, untar if it is a .tar file, and update self.local_paths with the path

                Args:
                    name: brief descriptor of the file or directory. Used as the key in the new self.local_paths entry
                    relative_path: path to file or directory, relative to the local_master / cloud_master directory
                    overwrite: if True, run rclone copy even if a local file with the intended name already exists

                Returns:
                    the full path the to the newly downloaded file or directory
                """
        if relative_path is None:
            try:
                relative_path = self._full_path_to_relative_path(self.local_paths[name])
            except KeyError:
                print('{} was not a valid key for the local_paths dictionary. '
                      'Please provide name and relative path instead'.format(name))
                return

        local_path = join(self.local_paths['master_dir'], relative_path)
        if local_path != self.local_paths[name]:
            print(
                '{0} is already a key for the local_paths dict, but the relative path provided ({1}) differs from the '
                'destination path previously defined ({2}). Either omit the relative_path keyword to download {2}, '
                'or choose a unique name.'.format(name, relative_path, self.local_paths[name]))
            return
        cloud_path = join(self.cloud_master_dir, relative_path)
        if not os.path.exists(local_path) or overwrite:
            run(['rclone', 'copyto', cloud_path, local_path])
            assert os.path.exists(local_path), "download failed\nsource: {}\ndestination_dir: {}".format(cloud_path,
                                                                                                         local_path)
        if os.path.splitext(local_path)[1] == '.tar':
            if not os.path.exists(os.path.splitext(local_path)[0]) or overwrite:
                run(['tar', '-xvf', local_path, '-C', os.path.dirname(local_path)])
            local_path = os.path.splitext(local_path)[0]
            assert os.path.exists(local_path), 'untarring failed for {}'.format(local_path)
        self.local_paths.update({name: local_path})
        os.remove(local_path + '.tar')
        return local_path

    def upload(self, name, tarred=False):
        """use rclone to upload a file

                Args:
                    name: brief descriptor of the file or directory. must be a key from self.local_paths

                Returns:
                    the full path the to the newly downloaded file or directory
                """
        try:
            local_path = self.local_paths[name]
        except KeyError:
            print('{} is not a valid key to the local_paths dict. Please use a valid key'.format(name))
            return

        cloud_path = self._local_path_to_cloud_path(local_path)

        if tarred:
            output = subprocess.run(
                ['tar', '-cvf', local_path + '.tar', '-C', local_path, os.path.split(local_path)[0]],
                capture_output=True, encoding='utf-8')
            if output.returncode != 0:
                print(output.stderr)
                raise Exception('Error in tarring ' + local_path)
            local_path += '.tar'

        output = subprocess.run(['rclone', 'copyto', local_path, cloud_path], capture_output=True, encoding='utf-8')
        if output.returncode != 0:
            raise Exception('Error in uploading file: ' + output.stderr)

    def _initialize(self):
        """create all required local directories and set the paths for files generated later."""
        # locate the cloud master directory
        self.cloud_master_dir = self._locate_cloud_master_dir()

        # create basic directory structure and define essential file-paths
        self._make_dir('master_dir',  join(os.getenv('HOME'), 'Temp', 'CichlidDetection'))
        self._make_dir('analysis_states_dir',
                       join(self.local_paths['master_dir'], '__AnalysisStates', 'CichlidDetection'))
        self._make_dir('data_dir', join(self.local_paths['master_dir'], '__ProjectData'))
        self._make_dir('model_dir',
                       join(self.local_paths['master_dir'], '__MachineLearningModels', 'FishDetectionModels'))
        self._make_dir('weights_dir', join(self.local_paths['model_dir'], 'weights'))
        self.local_paths.update({'weights_file': join(self.local_paths['weights_dir'], 'last.weights')})

        # if training, create training-specific directories and file-paths as well
        self._make_dir('training_dir', join(self.local_paths['master_dir'], '__TrainingData', 'CichlidDetection'))
        self._make_dir('train_image_dir', join(self.local_paths['training_dir'], 'train_images'))
        self._make_dir('test_image_dir', join(self.local_paths['training_dir'], 'test_images'))
        self._make_dir('label_dir', join(self.local_paths['training_dir'], 'labels'))
        self._make_dir('log_dir', join(self.local_paths['training_dir'], 'logs'))
        self._make_dir('predictions_dir', join(self.local_paths['training_dir'], 'predictions'))
        self._make_dir('training_figure_dir', join(self.local_paths['training_dir'], 'figures'))
        self._make_dir('training_figure_data_dir', join(self.local_paths['training_figure_dir'], 'figure_data'))
        self._make_dir('boxed_images_dir', join(self.local_paths['training_dir'], 'BoxedImages'))

        self.local_paths.update({'train_list': join(self.local_paths['training_dir'], 'train_list.txt')})
        self.local_paths.update({'test_list': join(self.local_paths['training_dir'], 'test_list.txt')})
        self.local_paths.update({'train_log': join(self.local_paths['log_dir'], 'train_log.txt')})
        self.local_paths.update({'val_log': join(self.local_paths['log_dir'], 'val_log.txt')})
        self.local_paths.update({'batch_log': join(self.local_paths['log_dir'], 'batch_log.txt')})
        self.local_paths.update({'ground_truth_csv': join(self.local_paths['training_dir'], 'ground_truth.csv')})
        self.local_paths.update({'boxed_fish_csv': join(self.local_paths['training_dir'], 'BoxedFish.csv')})

        # also sync the training directory and determine the unique project ID's from boxed_fish.csv
        if self.training:
            self.sync_training_dir()


    def _locate_cloud_master_dir(self):
        """locate the required files in Dropbox.

        Returns:
            string: cloud_master_dir, the outermost Dropbox directory that will be used henceforth
            dict: cloud_files, a dict of paths to remote files, keyed by brief descriptors
        """
        # establish the correct remote
        possible_remotes = run(['rclone', 'listremotes']).split()
        if len(possible_remotes) == 1:
            remote = possible_remotes[0]
        elif 'cichlidVideo:' in possible_remotes:
            remote = 'cichlidVideo:'
        elif 'd:' in possible_remotes:
            remote = 'd:'
        else:
            raise Exception('unable to establish rclone remote')

        # establish the correct path to the CichlidPiData directory
        root_dir = [r for r in run(['rclone', 'lsf', remote]).split() if 'McGrath' in r][0]
        cloud_master_dir = join(remote + root_dir, 'Apps', 'CichlidPiData')

        return cloud_master_dir

    def _local_path_to_cloud_path(self, local_path):
        return local_path.replace(self.local_paths['master_dir'], self.cloud_master_dir)

    def _cloud_path_to_local_path(self, cloud_path):
        return cloud_path.replace(self.cloud_master_dir, self.local_paths['master_dir'])

    def _full_path_to_relative_path(self, full_path):
        return full_path.replace(self.cloud_master_dir, '').replace(self.local_paths['master_dir'], '')

    def _make_dir(self, name, path):
        """update the self.local_paths dict with {name: path}, and create the directory if it does not exist

        Args:
            name (str): brief file descriptor, to be used as key in the local_paths dict
            path (str): local path of the directory to be created

        Returns:
            str: the path argument, unaltered
        """
        self.local_paths.update({name: make_dir(path)})
        return path


class ProjectFileManager(FileManager):
    """Project specific class for managing local and cloud storage, Inherits from FileManager"""

    def __init__(self, pid, file_manager=None, training=False):
        """initialize a new FileManager, unless an existing file manager was passed to the constructor to save time

        Args:
            pid (str): project id
            file_manager (FileManager): optional. pass a pre-existing FileManager object to improve performance when
                initiating numerous ProjectFileManagers
        """
        # initiate the FileManager parent class unless the optional file_manager argument is used
        self.training = training
        if file_manager is None:
            FileManager.__init__(self, training=training)
        # if the file_manager argument is used, manually inherit the required attributes
        else:
            self.local_paths = file_manager.local_paths.copy()
            self.cloud_master_dir = file_manager.cloud_master_dir
            self.training = training
        self.pid = pid
        # initialize project-specific directories
        self._initialize()

    def download_all(self, image_dir=True, video_dir=False):
        required_files = ['video_points_numpy', 'video_crop_numpy']
        for fname in required_files:
            self.download(fname)
        if image_dir:
            self.download('image_dir')
        if video_dir:
            self.download('video_dir')

    def update_annotations(self):
        self.download(self.local_paths['boxed_fish_csv'])


    def _initialize(self):
        """create project-specific directories and locate project-specific files in the cloud

        Overwrites FileManager._initialize() method
        """
        self._make_dir('project_dir', join(self.local_paths['data_dir'], self.pid))
        self._make_dir('master_analysis_dir', join(self.local_paths['project_dir'], 'MasterAnalysisFiles'))
        self._make_dir('summary_dir', join(self.local_paths['project_dir'], 'Summary'))
        self._make_dir('video_dir', join(self.local_paths['project_dir'], 'Videos'))
        self.local_paths.update({'video_points_numpy': join(self.local_paths['master_analysis_dir'], 'VideoPoints.npy')})
        self.local_paths.update({'video_crop_numpy': join(self.local_paths['master_analysis_dir'], 'VideoCrop.npy')})
        self.local_paths.update({'detections_csv': join(self.local_paths['master_analysis_dir'], 'Detections.csv')})
        self.local_paths.update({'labeled_frames_csv': join(self.local_paths['master_analysis_dir'], 'LabeledFrames.csv')})

        if self.training:
            self.local_paths.update({'image_dir':  join(self.local_paths['boxed_images_dir'], '{}.tar'.format(self.pid))})
        else:
            self._make_dir('image_dir', join(self.local_paths['project_dir'], 'Images'))


