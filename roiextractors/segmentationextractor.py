from abc import ABC, abstractmethod
from spikeextractors.baseextractor import BaseExtractor
import numpy as np
from .extraction_tools import ArrayType
from .extraction_tools import _pixel_mask_extractor
from copy import deepcopy
import warnings
import yaml
from pathlib import Path


class SegmentationExtractor(ABC, BaseExtractor):
    """
    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    ouput from the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    def __init__(self):
        BaseExtractor.__init__(self)
        self._sampling_frequency = None
        self._channel_names = ['OpticalChannel']
        self._num_planes = 1
        self._roi_response_raw = None
        self._roi_response_dff = None
        self._roi_response_neuropil = None
        self._roi_response_deconvolved = None
        self._image_correlation = None
        self._image_mean = None
        self._raw_movie_file_location = None
        self._experiment_metadata = self._get_default_metadata()

    def _get_default_metadata(self):
        """
        Updates the base metadata with class specific version
        Returns
        -------
        default_metadata: dict
        """
        with open(Path(__file__).parent.parent.joinpath('metadatafiles', 'base_metadata.yaml'), 'r') as f:
            default_metadata = yaml.safe_load(f)
        return default_metadata

    def _set_default_segext_metadata(self):
        """
        Called by specific segext classes to set metadata specific to them
        """
        # Optical Channel name:
        for i in range(self.get_num_channels()):
            ch_name = self.get_channel_names()[i]
            if i==0:
                self._experiment_metadata['ophys']['ImagingPlane'][0]['optical_channels'][i]['name'] = ch_name
            else:
                self._experiment_metadata['ophys']['ImagingPlane'][0]['optical_channels'].append(dict(
                    name=ch_name,
                    emission_lambda=500.0,
                    description=f'{ch_name} description'
                ))

        # set roi_response_series rate:
        rate = np.float('NaN') if self.get_sampling_frequency() is None else self.get_sampling_frequency()
        for trace_name, trace_data in self.get_traces_dict().items():
            if trace_name=='raw':
                if trace_data is not None:
                    self._experiment_metadata['ophys']['Fluorescence']['roi_response_series'][0].update(rate=rate)
                continue
            if len(trace_data.shape)!=0:
                self._experiment_metadata['ophys']['Fluorescence']['roi_response_series'].append(dict(
                    name=trace_name.capitalize(),
                    description=f'description of {trace_name} traces',
                    rate=rate
                ))
        # TwoPhotonSeries update:
        self._experiment_metadata['ophys']['TwoPhotonSeries'][0].update(
            dimension=self.get_image_size())

    def set_experiment_metadata(self, metadata_file):
        """
        Update the default values of metadata with a new yaml file
        Parameters
        ----------
        metadata_file
        """
        if isinstance(metadata_file,str):
            with open(metadata_file, 'r') as f:
                self._experiment_metadata.update(yaml.safe_load(f))
        elif isinstance(metadata_file,dict):
            self._experiment_metadata.update(metadata_file)
        else:
            raise Exception('enter a valid metadata_file type: yaml filepath/dictionary')

    def get_experiment_metadata(self):
        """
        Retrns the edited version of experiment metadata
        Returns
        -------
        self._experiment_metadata: dict
        """
        return self._experiment_metadata

    @abstractmethod
    def _calculate_roi_locations(self):
        """
        extracts the ROI locations from the images: median of the roi
        Returns
        -------
        roi_locs: np.array
        """
        pass

    @abstractmethod
    def get_accepted_list(self) -> list:
        """
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        """
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        """
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of rejected ROIs
        """
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        """This function returns the number of frames in the recording.

        Returns
        -------
        num_of_frames: int
            Number of frames in the recording (duration of recording).
        """
        pass

    @abstractmethod
    def get_roi_locations(self, roi_ids=None) -> np.array:
        """
        Returns the locations of the Regions of Interest

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        ------
        roi_locs: numpy.ndarray
            2-D array: 2 X num_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        pass

    @abstractmethod
    def get_roi_ids(self) -> list:
        """Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list.
        """
        pass

    @abstractmethod
    def get_roi_image_masks(self, roi_ids=None) -> np.array:
        """Returns the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        pass

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """
        Returns the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        pixel_masks: [list, NoneType]
            list of length number of rois, each element is a 2-D array os shape (no-pixels, 2)
        """
        if roi_ids is None:
            return None
        return _pixel_mask_extractor(self.get_roi_image_masks(roi_ids=roi_ids), range(len(roi_ids)))

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        """
        Frame size of movie ( x and y size of image).

        Returns
        -------
        num_rois: array_like
            2-D array: image y x image x
        """
        pass

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name='raw'):
        """
        Return RoiResponseSeries
        Returns
        -------
        traces: array_like
            2-D array (ROI x timepoints)
        """
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        traces = self.get_traces_dict().get(name)
        if len(traces.shape) == 0:
            print(f'traces for {name} not found, enter one of {list(self.get_traces_dict().keys())}')
            return None
        else:
            return np.array([traces[int(i), start_frame:end_frame] for i in roi_idx_])

    def get_traces_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries
            Flourescence, Neuropil, Deconvolved, Background etc
        """
        return deepcopy(dict(raw=np.array(self._roi_response_raw),
                             dff=np.array(self._roi_response_dff),
                             neuropil=np.array(self._roi_response_neuropil),
                             deconvolved=np.array(self._roi_response_deconvolved)))

    def get_images_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
            Mean, Correlation image
        """
        return deepcopy(dict(mean=self._image_mean,
                             correlation=self._image_correlation))

    def get_images(self, name='correlation'):
        """
        Return specific images: mean or correlation
        Parameters
        ----------
        name:str
            name of the type of image to retrieve
        Returns
        -------
        images: np.ndarray
        """
        image = self.get_images_dict().get(f'_image_{name}')
        if image:
            return image
        else:
            warnings.warn(f'could not find {name} image, enter one of {list(self.get_images_dict().keys())}')

    def get_sampling_frequency(self):
        """This function returns the sampling frequency in units of Hz.

        Returns
        -------
        samp_freq: float
            Sampling frequency of the recordings in Hz.
        """
        return np.float(self._sampling_frequency) if self._sampling_frequency else None

    def get_num_rois(self):
        """Returns total number of Regions of Interest in the acquired images.

        Returns
        -------
        num_rois: int
            integer number of ROIs extracted.
        """
        return len(self.get_roi_ids())

    def get_channel_names(self):
        """
        Names of channels in the pipeline
        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        return self._channel_names

    def get_num_channels(self):
        """
        Number of channels in the pipeline
        Returns
        -------
        num_of_channels: int
        """
        return len(self._channel_names)

    def get_num_planes(self):
        """
        Returns the default number of planes of imaging for the segmentation extractor.
        Detaults to 1 for all but the MultiSegmentationExtractor
        Returns
        -------
        self._num_planes: int
        """
        return self._num_planes

    @staticmethod
    def write_segmentation(segmentation_extractor, save_path, plane_num=0, file_overwrite=True):
        """
        Static method to write recording back to the native format.

        Parameters
        ----------
        segmentation_extractor: SegmentationExtractor
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        save_path: str
            path to save the native format.
        plane_num: int
            plane number for the imaging plane used for the segmentation.
        file_overwrite: bool
            overwrite the file if it exists.
        """
        raise NotImplementedError
