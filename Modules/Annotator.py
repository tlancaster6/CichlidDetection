from Modules.FileManager import FileManager, ProjectFileManager
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt

# These disable some of the default key strokes that we will use
plt.rcParams['keymap.all_axes'] = ''  # a
plt.rcParams['keymap.back'] = ['left', 'backspace', 'MouseButton.BACK']  # c
plt.rcParams['keymap.pan'] = ''  # p
plt.rcParams['keymap.quit'] = ['ctrl+w', 'cmd+w']
# Import buttons that we will use to make this interactive
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Slider
# Import some patches that we will use to display the annotations
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

import pdb, datetime, os, subprocess, argparse, random
import pandas as pd


class Annotation:
    def __init__(self, other):
        self.other = other
        self.sex = ''
        self.coords = ()
        self.poses = ()
        self.rectangle = None

    def addRectangle(self):
        if self.rectangle is None:
            self.rectangle = Rectangle((self.coords[0], self.coords[1]), self.coords[2], self.coords[3],
                                       fill=False, edgecolor='green', linewidth=1.4, figure=self.other.fig)
            self.other.ax_image.add_patch(self.rectangle)
            self.other.cur_text.set_text('BB: ' + str(self.coords))
        else:
            self.other.error_text.set_text('Error: Rectangle already exists')

    def removePatches(self):
        if self.lastRectangle is None:
            self.other.error_text.set_text('Cant remove annotation. Reset frame instead')
            self.other.fig.canvas.draw()

            return False
        try:
            self.other.ax_image.patches.remove(self.lastRectangle)
        except ValueError:
            pass
        return True

    def retRow(self):
        if self.coords == ():
            return 'Must create bounding box before saving an annotation'
        return [self.other.pid, self.other.frames[self.other.frame_index], self.sex, self.coords, self.other.user,
                self.other.now]

    def reset(self):
        self.sex = ''
        self.coords = ()
        self.poses = ()
        self.lastRectangle = self.rectangle
        self.rectangle = None


class Annotator:
    def __init__(self, project_file_manager: ProjectFileManager, number=10, dry=False):

        self.pfm = project_file_manager
        self.image_dir = self.pfm.local_paths['image_dir']
        self.labeled_frames_csv = self.pfm.local_paths['labeled_frames_csv']
        self.boxed_fish_csv = self.pfm.local_paths['boxed_fish_csv']

        self.number = number
        self.pid = self.pfm.pid
        self.dry = dry

        self.frames = sorted([x for x in os.listdir(self.image_dir) if '.jpg' in x and '._' not in x])
        random.Random(4).shuffle(self.frames)
        # self.frames = sorted([x for x in os.listdir(self.frameDirectory) if '.jpg' in x and '._' not in x]) # remove annoying mac OSX files
        assert len(self.frames) > 0

        # Keep track of the frame we are on and how many we have annotated
        self.frame_index = 0
        self.annotated_frames = []

        # Intialize lists to hold annotated objects
        self.coords = ()

        # Create dataframe to hold annotations
        if os.path.exists(self.labeled_frames_csv):
            self.dt = pd.read_csv(self.labeled_frames_csv, index_col=0)
        else:
            self.dt = pd.DataFrame(columns=['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'User', 'DateTime'])
        self.f_dt = pd.DataFrame(columns=['ProjectID', 'Framefile', 'Sex', 'Box', 'User', 'DateTime'])

        # Get user and current time
        self.user = os.getenv('USER')
        self.now = datetime.datetime.now()

        # Create Annotation object
        self.annotation = Annotation(self)

        #
        self.annotation_text = ''

        # Start figure
        self._createFigure()

        # if not a dry run, upload results
        if not dry:
            self._upload_results()

    def _upload_results(self):
        labeled_frames_df = pd.read_csv(self.labeled_frames_csv, index_col=0)
        boxed_fish_df = pd.read_csv(self.boxed_fish_csv, index_col=0)

        boxed_fish_df = boxed_fish_df.append(labeled_frames_df, sort=True).drop_duplicates(
                        subset=['ProjectID', 'Framefile', 'User', 'Sex', 'Box'])
        boxed_fish_df.to_csv(self.boxed_fish_csv,
                             columns=['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'User', 'DateTime'])

        self.pfm.upload('labeled_frames_csv')
        self.pfm.upload('boxed_fish_csv')
        self.pfm.upload('image_dir')

    def _createFigure(self):
        # Create figure
        self.fig = fig = plt.figure(1, figsize=(10, 7))

        # Create image subplot
        self.ax_image = fig.add_axes([0.05, 0.2, .8, 0.75])
        while len(self.dt[(self.dt.Framefile == self.frames[self.frame_index]) & (self.dt.User == self.user)]) != 0:
            self.frame_index += 1
        # Create slider for saturation
        self.ax_saturation = fig.add_axes([0.1, 0.08, 0.2, 0.03])
        self.slid_saturation = Slider(self.ax_saturation, 'Saturation', 0, 10, valinit=1, valstep=.1)

        # Plot image
        self.img = Image.open(self.image_dir + self.frames[self.frame_index])
        # img = plt.imread(self.frameDirectory + self.frames[self.frame_index])
        # print(img.shape)
        self.converter = ImageEnhance.Color(self.img)
        img = self.converter.enhance(self.slid_saturation.val)

        self.image_obj = self.ax_image.imshow(img)
        self.ax_image.set_title('Frame ' + str(self.frame_index) + ': ' + self.frames[self.frame_index])

        # Create selectors for identifying bounding bos and body parts (nose, left eye, right eye, tail)

        self.RS = RectangleSelector(self.ax_image, self._grabBoundingBox,
                                    drawtype='box', useblit=True,
                                    button=[1, 3],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.RS.set_active(True)

        # Create radio buttons
        self.ax_radio = fig.add_axes([0.85, 0.85, 0.125, 0.1])
        self.radio_names = [r"$\bf{M}$" + 'ale', r"$\bf{F}$" + 'emale', r"$\bf{U}$" + 'nknown']
        self.bt_radio = RadioButtons(self.ax_radio, self.radio_names, active=0, activecolor='blue')

        # Create click buttons for adding annotations
        self.ax_boxAdd = fig.add_axes([0.85, 0.775, 0.125, 0.04])
        self.bt_boxAdd = Button(self.ax_boxAdd, r"$\bf{A}$" + 'dd Box')
        self.ax_boxClear = fig.add_axes([0.85, 0.725, 0.125, 0.04])
        self.bt_boxClear = Button(self.ax_boxClear, r"$\bf{C}$" + 'lear Box')

        # Create click buttons for saving frame annotations or starting over
        self.ax_frameClear = fig.add_axes([0.85, 0.375, 0.125, 0.04])
        self.bt_frameClear = Button(self.ax_frameClear, r"$\bf{R}$" + 'eset Frame')
        self.ax_frameAdd = fig.add_axes([0.85, 0.325, 0.125, 0.04])
        self.bt_frameAdd = Button(self.ax_frameAdd, r"$\bf{N}$" + 'ext Frame')
        self.ax_framePrevious = fig.add_axes([0.85, 0.275, 0.125, 0.04])
        self.bt_framePrevious = Button(self.ax_framePrevious, r"$\bf{P}$" + 'revious Frame')

        # Create click button for quitting annotations
        self.ax_quit = fig.add_axes([0.85, 0.175, 0.125, 0.04])
        self.bt_quit = Button(self.ax_quit, r"$\bf{Q}$" + 'uit and save')

        # Add text boxes to display info on annotations
        self.ax_cur_text = fig.add_axes([0.85, 0.575, 0.125, 0.14])
        self.ax_cur_text.set_axis_off()
        self.cur_text = self.ax_cur_text.text(0, 1, '', fontsize=8, verticalalignment='top')

        self.ax_all_text = fig.add_axes([0.85, 0.425, 0.125, 0.19])
        self.ax_all_text.set_axis_off()
        self.all_text = self.ax_all_text.text(0, 1, '', fontsize=9, verticalalignment='top')

        self.ax_error_text = fig.add_axes([0.3, 0.05, .6, 0.1])
        self.ax_error_text.set_axis_off()
        self.error_text = self.ax_error_text.text(0, 1, '', fontsize=14, color='red', verticalalignment='top')

        # Set buttons in active that shouldn't be pressed
        # self.bt_poses.set_active(False)

        # Turn on keypress events to speed things up
        self.fig.canvas.mpl_connect('key_press_event', self._keypress)

        # Turn off hover event for buttons (no idea why but this interferes with the image rectange remaining displayed)
        self.fig.canvas.mpl_disconnect(self.bt_boxAdd.cids[2])
        self.fig.canvas.mpl_disconnect(self.bt_boxClear.cids[2])
        self.fig.canvas.mpl_disconnect(self.bt_frameAdd.cids[2])
        self.fig.canvas.mpl_disconnect(self.bt_frameClear.cids[2])
        self.fig.canvas.mpl_disconnect(self.bt_framePrevious.cids[2])
        self.fig.canvas.mpl_disconnect(self.bt_quit.cids[2])

        # Connect buttons to specific functions
        self.bt_boxAdd.on_clicked(self._addBoundingBox)
        self.bt_boxClear.on_clicked(self._clearBoundingBox)
        self.bt_frameClear.on_clicked(self._clearFrame)
        self.bt_framePrevious.on_clicked(self._previousFrame)
        self.bt_frameAdd.on_clicked(self._nextFrame)
        self.bt_quit.on_clicked(self._quit)
        self.slid_saturation.on_changed(self._updateFrame)

        # Show figure
        plt.show()

    def _grabBoundingBox(self, eclick, erelease):
        self.error_text.set_text('')

        # Transform and store image coords
        image_coords = list(self.ax_image.transData.inverted().transform((eclick.x, eclick.y))) + list(
            self.ax_image.transData.inverted().transform((erelease.x, erelease.y)))

        # Convert to integers:
        image_coords = tuple([int(x) for x in image_coords])

        xy = (min(image_coords[0], image_coords[2]), min(image_coords[1], image_coords[3]))
        width = abs(image_coords[0] - image_coords[2])
        height = abs(image_coords[1] - image_coords[3])
        self.annotation.coords = xy + (width, height)

    def _keypress(self, event):
        if event.key in ['m', 'f', 'u']:
            self.bt_radio.set_active(['m', 'f', 'u'].index(event.key))
        # self.fig.canvas.draw()
        elif event.key == 'a':
            self._addBoundingBox(event)
        elif event.key == 'c':
            self._clearBoundingBox(event)
        elif event.key == 'n':
            self._nextFrame(event)
        elif event.key == 'r':
            self._clearFrame(event)
        elif event.key == 'p':
            self._previousFrame(event)
        elif event.key == 'q':
            self._quit(event)
        else:
            pass

    def _addBoundingBox(self, event):
        if self.annotation.coords == ():
            self.error_text.set_text('Error: Bounding box not set')
            self.fig.canvas.draw()
            return

        displayed_names = [r"$\bf{M}$" + 'ale', r"$\bf{F}$" + 'emale', r"$\bf{U}$" + 'nknown']
        stored_names = ['m', 'f', 'u']

        self.annotation.sex = stored_names[displayed_names.index(self.bt_radio.value_selected)]

        # Add new patch rectangle
        # colormap = {self.radio_names[0]:'blue', self.radio_names[1]:'pink', self.radio_names[2]: 'red', self.radio_names[3]: 'black'}
        # color = colormap[self.bt_radio.value_selected]
        self.annotation.addRectangle()

        outrow = self.annotation.retRow()

        if type(outrow) == str:
            self.error_text.set_text(outrow)
            self.fig.canvas.draw()
            return
        else:
            self.f_dt.loc[len(self.f_dt)] = outrow
            self.f_dt.drop_duplicates(subset=['ProjectID', 'Framefile', 'User', 'Sex', 'Box'])

        self.annotation_text += self.annotation.sex + ':' + str(self.annotation.coords) + '\n'
        # Add annotation to the temporary data frame
        self.cur_text.set_text(self.annotation_text)
        self.all_text.set_text('# Ann = ' + str(len(self.f_dt)))

        self.annotation.reset()

        self.fig.canvas.draw()

    def _clearBoundingBox(self, event):

        if not self.annotation.removePatches():
            return

        self.annotation_text = self.annotation_text.split(self.annotation_text.split('\n')[-2])[0]

        self.annotation.reset()

        self.f_dt.drop(self.f_dt.tail(1).index, inplace=True)

        self.cur_text.set_text(self.annotation_text)
        self.all_text.set_text('# Ann = ' + str(len(self.f_dt)))

        self.fig.canvas.draw()

    def _nextFrame(self, event):

        if self.annotation.coords != ():
            self.error_text.set_text('Save or clear (esc) current annotation before moving on')
            return

        if len(self.f_dt) == 0:
            self.f_dt.loc[0] = [self.pid, self.frames[self.frame_index], '', '', self.user, self.now]
            self.f_dt['Nfish'] = 0
        else:
            self.f_dt['Nfish'] = len(self.f_dt)
        self.dt = self.dt.append(self.f_dt, sort=True)
        # Save dataframe (in case user quits)
        self.dt.to_csv(self.labeled_frames_csv, sep=',',
                       columns=['ProjectID', 'Framefile', 'Nfish', 'Sex', 'Box', 'User', 'DateTime'])
        self.f_dt = pd.DataFrame(columns=['ProjectID', 'Framefile', 'Sex', 'Box', 'User', 'DateTime'])
        self.annotated_frames.append(self.frame_index)

        # Remove old patches
        self.ax_image.patches = []

        # Reset annotations
        self.annotation = Annotation(self)
        self.annotation_text = ''

        # Update frame index and determine if all images are annotated
        self.frame_index += 1
        while len(self.dt[(self.dt.Framefile == self.frames[self.frame_index]) & (self.dt.User == self.user)]) != 0:
            self.frame_index += 1

        if self.frame_index == len(self.frames) or len(self.annotated_frames) == self.number:
            # Disconnect connections and close figure
            plt.close(self.fig)

        self.cur_text.set_text('')
        self.all_text.set_text('')

        # Load new image and save it as the background
        self.img = Image.open(self.image_dir + self.frames[self.frame_index])
        # img = plt.imread(self.frameDirectory + self.frames[self.frame_index])
        # print(img.shape)
        self.converter = ImageEnhance.Color(self.img)
        img = self.converter.enhance(self.slid_saturation.val)

        self.image_obj.set_array(img)
        self.ax_image.set_title('Frame ' + str(self.frame_index) + ': ' + self.frames[self.frame_index])
        self.fig.canvas.draw()

    # self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _updateFrame(self, event):
        img = self.converter.enhance(self.slid_saturation.val)
        self.image_obj.set_array(img)
        self.fig.canvas.draw()

    def _clearFrame(self, event):
        print('Clearing')
        self.f_dt = pd.DataFrame(columns=['ProjectID', 'Framefile', 'Sex', 'Box', 'User', 'DateTime'])
        # Remove old patches
        self.ax_image.patches = []
        self.annotation_text = ''
        self.annotation = Annotation(self)

        self.cur_text.set_text(self.annotation_text)
        self.all_text.set_text('# Ann = ' + str(len(self.f_dt)))

        self.fig.canvas.draw()

        # Reset annotations
        self.annotation = Annotation(self)

    def _previousFrame(self, event):
        self.frame_index = self.annotated_frames.pop()
        self.dt = self.dt[self.dt.Framefile != self.frames[self.frame_index]]
        self._clearFrame(event)
        # Load new image and save it as the background
        self.img = Image.open(self.image_dir + self.frames[self.frame_index])
        self.converter = ImageEnhance.Color(self.img)
        img = self.converter.enhance(self.slid_saturation.val)

        # img = plt.imread(self.frameDirectory + self.frames[self.frame_index])
        self.image_obj.set_array(img)
        self.ax_image.set_title('Frame ' + str(self.frame_index) + ': ' + self.frames[self.frame_index])
        self.fig.canvas.draw()

    def _quit(self, event):
        plt.close(self.fig)
