import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from roiextractors.extraction_tools import _pixel_mask_extractor

class CnmfeSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    """
    extractor_name = 'CnmfeSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.image_masks = self._image_mask_extractor_read()
        self._roi_response = self._trace_extractor_read()
        self._roi_response_fluorescence = self._roi_response
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self._roi_response.shape[1]/self._tot_exptime_extractor_read()
        self._images_correlation = self._summary_image_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.file_path, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]['extractedImages']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['extractedSignals'])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['Cn']
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['movieList'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]

    def _calculate_roi_locations(self):
        roi_location = np.ndarray([2, self.get_num_rois()], dtype='int')
        for i in range(self.get_num_rois()):
            temp = np.where(self.image_masks[:, :, i] == np.amax(self.image_masks[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, save_path):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_num_rois(self):
        return self._roi_response.shape[0]

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self._calculate_roi_locations()
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self._calculate_roi_locations()[:, roi_idx_]

    def get_num_frames(self):
        return self._roi_response.shape[1]

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in roi_idx_]).T

    def get_image_size(self):
        return self.image_masks.shape[0:2]
