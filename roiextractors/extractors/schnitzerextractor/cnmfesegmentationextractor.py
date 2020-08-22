import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from roiextractors.extraction_tools import _pixel_mask_extractor
import os
import shutil
from scipy.sparse import csc_matrix

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
        self._roi_response_deconvolved = self._spikes_extract()
        self._raw_movie_file_location = self._raw_datafile_read()
        if self._dataset_file[self._group0[0]]['inputOptions'].get('Fs'):
            self._sampling_frequency = self._dataset_file[self._group0[0]]['inputOptions']['Fs']
        self._images_correlation = self._summary_image_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]['extractedImages']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['extractedSignals'])
        return extracted_signals.T

    def _spikes_extract(self):
        if self._dataset_file[self._group0[0]].get('extractedPeaks'):
            data = self._dataset_file[self._group0[0]]['extractedPeaks']['data']
            indices = self._dataset_file[self._group0[0]]['extractedPeaks']['ir']
            indptr = self._dataset_file[self._group0[0]]['extractedPeaks']['jc']
            spikes = csc_matrix((data, indices, indptr), shape=self._roi_response.shape).toarray()
            return spikes
        else:
            return None

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['Cn']
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['movieList'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.no_rois))

    def get_rejected_list(self):
        return [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        roi_location = np.ndarray([2, self.no_rois], dtype='int')
        for i in range(self.no_rois):
            temp = np.where(self.image_masks[:, :, i] == np.amax(self.image_masks[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath, **kwargs):
        savepath_folder = os.path.dirname(savepath)
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
            main = f.create_group('cnmfeAnalysisOutput')
            # create datasets:
            main.create_dataset('extractedImages', data=segmentation_object.get_roi_image_masks().T)
            main.create_dataset('extractedSignals', data=segmentation_object.get_traces().T)
            if segmentation_object.get_traces(name='deconvolved') is not None:
                image_mask_csc = csc_matrix(segmentation_object.get_traces(name='deconvolved'))
                main.create_dataset('extractedPeaks/data', data=image_mask_csc.data)
                main.create_dataset('extractedPeaks/ir', data=image_mask_csc.indices)
                main.create_dataset('extractedPeaks/jc', data=image_mask_csc.indptr)
            if segmentation_object.get_images() is not None:
                main.create_dataset('Cn', data=segmentation_object.get_images())
            main.create_dataset('movieList', data=[ord(i) for i in segmentation_object.get_movie_location()])
            inputoptions = main.create_group('inputOptions')
            if segmentation_object.get_sampling_frequency() is not None:
                inputoptions.create_dataset('Fs', data=segmentation_object.get_sampling_frequency())


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
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in roi_idx_]).T

    def get_image_size(self):
        return self.image_masks.shape[0:2]
