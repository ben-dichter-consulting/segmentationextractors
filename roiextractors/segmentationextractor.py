from abc import ABC, abstractmethod
from spikeextractors.baseextractor import BaseExtractor
import numpy as np
from typing import Union
from .extraction_tools import ArrayType, PathType


class SegmentationExtractor(ABC, BaseExtractor):
    '''An abstract class that contains all the meta-data and output data from
        the ROI segmentation operation when applied to the pre-processed data.
        It also contains methods to read from and write to various data formats
        ouput from the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.
        All the methods with @abstract decorator have to be defined by the
        format specific classes that inherit from this.
    '''

    def __init__(self):
        BaseExtractor.__init__(self)
        self._sampling_frequency = None

    @property
    def image_size(self):
        '''
        Returns
        -------
        image_dims: list
            The width X height of the image.
        '''
        return self.get_image_size()

    @property
    def no_rois(self):
        '''
        The number of Independent sources(neurons) indentified after the
        segmentation operation. The regions of interest for which fluorescence
        traces will be extracted downstream.

        Returns
        -------
        no_rois: int
            The number of rois
        '''
        return len(self.get_num_rois())

    @property
    def roi_ids(self):
        '''
        Integer label given to each region of interest (neuron).

        Returns
        -------
        roi_idx: list
            list of integers of the ROIs. Listed in the order in which the ROIs
            occur in the image_masks (2nd dimention)
        '''
        return self.get_roi_ids()



    @property
    def roi_locations(self):
        '''
        The x and y pixel location of the ROIs. The location where the pixel
        value is maximum in the image mask.

        Returns
        -------
        roi_locs: np.array
            Array with the first column representing the x (width) and second representing
            the y (height) coordinates of the ROI.
        '''
        return self.get_roi_locations()

    @property
    def num_frames(self):
        '''
        Total number of images in the image sequence across time.

        Returns
        -------
        num_of_frames: int
            Same as the -1 dimention of the dF/F trace(roi_response).
        '''
        return self.get_num_frames()

    @property
    def sampling_frequency(self):
        '''
        Returns
        -------
        samp_freq: int
            Sampling frequency of the dF/F trace.
        '''
        return self._sampling_frequency

    @abstractmethod
    def get_accepted_list(self) -> list:
        '''
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        '''
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        '''
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of rejected ROIs
        '''
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        '''This function returns the number of frames in the recording.

        Returns
        -------
        num_of_frames: int
            Number of frames in the recording (duration of recording).
        '''
        pass

    @abstractmethod
    def get_roi_locations(self, roi_ids=None) -> np.array:
        '''
        Returns the locations of the Regions of Interest

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        ------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        '''
        pass

    @abstractmethod
    def get_roi_ids(self) -> list:
        '''Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list.
        '''
        pass


    @abstractmethod
    def get_roi_image_masks(self, roi_ids=None) -> np.array:
        '''Returns the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(ROI_ids)
        '''
        pass

    @abstractmethod
    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        '''Returns the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        pixel_masks: numpy.ndarray
            3-D array with weight for each pixel of the rroi: image_height X image_width X length(ROI_ids)
        '''
        pass

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        '''Frame size of movie ( x and y size of image).

        Returns
        -------
        no_rois: array_like
            2-D array: image y x image x
        '''
        pass

    def get_sampling_frequency(self):
        '''This function returns the sampling frequency in units of Hz.

        Returns
        -------
        samp_freq: float
            Sampling frequency of the recordings in Hz.
        '''
        return self._sampling_frequency

    def get_num_rois(self):
        '''Returns total number of Regions of Interest in the acquired images.

        Returns
        -------
        no_rois: int
            integer number of ROIs extracted.
        '''
        return len(self.get_roi_ids())

    @staticmethod
    def write_segmentation(segmentation, savepath):
        '''
        Static method to write recording back to the native format.

        Parameters
        ----------
        segmentation_object: SegmentationExtracteor object
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        savepath: str
            path to save the native format.
        '''
        raise NotImplementedError