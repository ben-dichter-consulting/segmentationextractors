import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from roiextractors.extraction_tools import _pixel_mask_extractor
import os
import shutil

class ExtractSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'EXTRACT\' ROI segmentation method.
    """
    extractor_name = 'ExtractSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.image_masks = self._image_mask_extractor_read()
        self._roi_response = self._trace_extractor_read()
        self._roi_response_fluorescence = self._roi_response
        self._raw_movie_file_location = self._raw_datafile_read()
        if self._tot_exptime_extractor_read():
            self._sampling_frequency = self._roi_response.shape[1]/self._tot_exptime_extractor_read()
        self._images_correlation = self._summary_image_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]['filters']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['traces'])
        return extracted_signals

    def _tot_exptime_extractor_read(self):
        total_time = self._dataset_file[self._group0[0]]['time'].get('totalTime', None)
        if total_time is None:
            return None
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['info'].get('summary_image', None)
        if summary_images_ is None:
            return None
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['file'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.no_rois))

    def get_rejected_list(self):
        return [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        no_ROIs = self.no_rois
        raw_images = self.image_masks
        roi_location = np.ndarray([2, no_ROIs], dtype='int')
        for i in range(no_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath, **kwargs):
        plane_no = kwargs.get('plane_no', 0)
        filename = os.path.basename(savepath)
        savepath_folder = os.path.join(os.path.dirname(savepath), f'Plane_{plane_no}')
        savepath = os.path.join(savepath_folder, filename)
        if not os.path.exists(savepath_folder):
            os.makedirs(savepath_folder)
        else:
            if os.path.exists(savepath):
                os.remove(savepath)
        if savepath.split('.')[-1] != 'mat':
            raise ValueError('filetype to save must be *.mat')
        with h5py.File(savepath, 'a') as f:
            # create base groups:
            _ = f.create_group('#refs#')
            main = f.create_group('extractAnalysisOutput')
            #create datasets:
            main.create_dataset('filters',data=segmentation_object.get_roi_image_masks().T)
            main.create_dataset('traces', data=segmentation_object.get_traces())
            info = main.create_group('info')
            if segmentation_object.get_images() is not None:
                info.create_dataset('summary_image', data=segmentation_object.get_images())
            main.create_dataset('file', data=[ord(i) for i in segmentation_object.get_movie_location()])
            time = main.create_group('time')
            if segmentation_object.get_sampling_frequency() is not None:
                time.create_dataset('totalTime', (1,1), data=segmentation_object.get_roi_image_masks().shape[1]/
                                                        segmentation_object.get_sampling_frequency())

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_num_rois(self):
        return self._roi_response.shape[0]

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self.roi_locations
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locations[:, roi_idx_]

    def get_num_frames(self):
        return self._roi_response.shape[1]

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = self.roi_ids
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in roi_idx_]).T

    def get_image_size(self):
        return self.image_masks.shape[0:2]
