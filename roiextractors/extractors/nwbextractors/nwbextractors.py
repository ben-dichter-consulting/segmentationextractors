import os
import uuid
import numpy as np
import yaml
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView

try:
    from pynwb import NWBHDF5IO, TimeSeries, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries, DfOverF
    from pynwb.file import Subject
    from pynwb.device import Device
    from hdmf.data_utils import DataChunkIterator

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False


def check_nwb_install():
    assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


def set_dynamic_table_property(dynamic_table, ids, row_ids, property_name, values, index=False,
                               default_value=np.nan, description='no description'):
    check_nwb_install()
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be a list of integers")
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values) and index is False:
        raise ValueError("'ids' and 'values' should be lists of same size")

    if index is False:
        if property_name in dynamic_table:
            for (row_id, value) in zip(row_ids, values):
                dynamic_table[property_name].data[ids.index(row_id)] = value
        else:
            col_data = [default_value] * len(ids)  # init with default val
            for (row_id, value) in zip(row_ids, values):
                col_data[ids.index(row_id)] = value
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=col_data,
                index=index
            )
    else:
        if property_name in dynamic_table:
            raise NotImplementedError
        else:
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=values,
                index=index
            )


def get_dynamic_table_property(dynamic_table, *, row_ids=None, property_name):
    all_row_ids = list(dynamic_table.id[:])
    if row_ids is None:
        row_ids = all_row_ids
    return [dynamic_table[property_name][all_row_ids.index(x)] for x in row_ids]


class NwbImagingExtractor(ImagingExtractor):
    """
    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    """

    extractor_name = 'NwbImaging'
    installed = HAVE_NWB # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb Extractor run:\n\n pip install pynwb\n\n"  # error message when not installed

    def __init__(self, file_path, optical_channel_name=None,
                 imaging_plane_name=None, image_series_name=None,
                 processing_module_name=None,
                 neuron_roi_response_series_name=None,
                 background_roi_response_series_name=None):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.nwb file.
        optical_channel_name: str(optional)
            optical channel to extract data from
        imaging_plane_name: str(optional)
            imaging plane to extract data from
        image_series_name: str(optional)
            imaging series to extract data from
        processing_module_name: str(optional)
            processing module to extract data from
        neuron_roi_response_series_name: str(optional)
            name of roi response series to extract data from
        background_roi_response_series_name: str(optional)
            name of background roi response series to extract data from
        """
        assert HAVE_NWB, self.installation_mesg
        ImagingExtractor.__init__(self)

    #TODO placeholders
    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        return self._video[frame_idx]

    def get_frames(self, frame_idxs):
        assert np.all(frame_idxs < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._video[frame_idx]
            planes[i] = plane
        return planes

    # TODO make decorator to check and correct inputs
    def get_video(self, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = self._video[start_frame: end_frame]

        return video

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self):
        return self._video.dtype

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self._channel_names

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        num_of_channels: int
            integer count of number of channels
        """
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, save_path):
        pass


class NwbSegmentationExtractor(SegmentationExtractor):

    extractor_name = 'NwbSegmentationExtractor'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        """
        Creating NwbSegmentationExtractor object from nwb file
        Parameters
        ----------
        file_path: str
            .nwb file location
        """
        check_nwb_install()
        SegmentationExtractor.__init__(self)
        if not os.path.exists(file_path):
            raise Exception('file does not exist')

        self.file_path = file_path
        self.image_masks = None
        self._roi_locs = None
        self._accepted_list = None
        self._io = NWBHDF5IO(file_path, mode='r+')
        nwbfile = self._io.read()
        self.nwbfile = nwbfile
        _nwbchildren_type = [type(i).__name__ for i in nwbfile.all_children()]
        _nwbchildren_name = [i.name for i in nwbfile.all_children()]
        _procssing_module = [_nwbchildren_name[f]
                             for f, u in enumerate(_nwbchildren_type) if u == 'ProcessingModule']
        mod = nwbfile.processing[_procssing_module[0]]
        if len(_procssing_module) > 1:
            print('multiple processing modules found, picking the first one')
        elif not mod:
            raise Exception('no processing module found')

        # Extract image_mask/background:
        _plane_segmentation_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'PlaneSegmentation']
        if not _plane_segmentation_exist:
            print('could not find a plane segmentation to contain image mask')
        else:
            ps = nwbfile.all_children()[_plane_segmentation_exist[0]]
        # self.image_masks = np.moveaxis(np.array(ps['image_mask'].data), [0, 1, 2], [2, 0, 1])
        if 'image_mask' in ps.colnames:
            self.image_masks = DatasetView(ps['image_mask'].data).lazy_transpose([1, 2, 0])
        if 'RoiCentroid' in ps.colnames:
            self._roi_locs = ps['RoiCentroid']
        if 'Accepted' in ps.colnames:
            self._accepted_list = ps['Accepted'].data[:]
        # Extract Image dimensions:

        # Extract roi_response:
        _roi_response_dict = dict()
        _roi_names = [_nwbchildren_name[val]
                      for val, i in enumerate(_nwbchildren_type) if i == 'RoiResponseSeries']
        if not _roi_names:
            raise Exception('no ROI response series found')
        else:
            for roi_name in _roi_names:
                _roi_response_dict[roi_name] = mod['Fluorescence'].get_roi_response_series(roi_name).data[:].T
        self._roi_response_raw = _roi_response_dict[_roi_names[0]]
        for trace_names in ['roiresponseseries','neuropil','deconvolved']:
            trace_name_find = [j for j,i in enumerate(_roi_names) if trace_names in i.lower()]
            if trace_name_find:
                trace_names = 'fluorescence' if trace_names == 'roiresponseseries' else trace_names
                setattr(self,f'_roi_response_{trace_names}',
                        mod['Fluorescence'].get_roi_response_series(_roi_names[trace_name_find[0]]).data[:].T)
                
        # Extract samp_freq:
        self._sampling_frequency = mod['Fluorescence'].get_roi_response_series(_roi_names[0]).rate
        # Extract get_num_rois()/ids:
        self._roi_idx = np.array(ps.id.data)

        # Imaging plane:
        _optical_channel_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'OpticalChannel']
        if _optical_channel_exist:
            self._channel_names = []
            for i in _optical_channel_exist:
                self._channel_names.append(nwbfile.all_children()[i].name)
        # Movie location:
        _image_series_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'TwoPhotonSeries']
        if not _image_series_exist:
            self._extimage_dims = None
        else:
            self._raw_movie_file_location = \
                nwbfile.all_children()[_image_series_exist[0]].external_file[:][0]
            self._extimage_dims = \
                nwbfile.all_children()[_image_series_exist[0]].dimension

        # property name/data extraction:
        self._property_name_exist = [
            i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
        self.property_vals = []
        for i in self._property_name_exist:
            self.property_vals.append(np.array(ps[i].data))

        #Extracting stores images as GrayscaleImages:
        _greyscaleimages = [i for i in nwbfile.all_children() if type(i).__name__ == 'GrayscaleImage']
        self._images_correlation = [i.data[()] for i in _greyscaleimages if 'corr' in i.name.lower()][0]
        self._images_mean = [i.data[()] for i in _greyscaleimages if 'mean' in i.name.lower()][0]

    def __del__(self):
        self._io.close()

    def get_accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.get_num_rois()))
        else:
            return np.where(self._accepted_list==1)[0].tolist()

    def get_rejected_list(self):
        return [a for a in self.get_roi_ids() if a not in set(self.get_accepted_list())]

    def _calculate_roi_locations(self):
        if self._roi_locs is None:
            return None
        else:
            return self._roi_locs.data[:].T

    def get_num_frames(self):
        return self._roi_response_raw.shape[1]

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self._calculate_roi_locations()
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self._calculate_roi_locations()[:, roi_idx_]

    def get_roi_ids(self):
        return self._roi_idx

    def get_roi_image_masks(self, roi_ids=None):
        if self.image_masks is None:
            return None
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in roi_idx_]).T

    def get_image_size(self):
        return self._extimage_dims

    def get_property_data(self, property_name):
        ret_val = []
        for j, i in enumerate(property_name):
            if i in self._property_name_exist:
                ret_val.append(self.property_vals[j])
            else:
                raise Exception('enter valid property name. Names found: {}'.format(
                    self._property_name_exist))
        return ret_val

    @staticmethod
    def write_segmentation(segext_obj, save_path, metadata, **kwargs):
        print(f'writing nwb for {segext_obj.extractor_name}\n')
        if isinstance(metadata, str):
            with open(metadata, 'r') as f:
                metadata = yaml.safe_load(f)

        metadata = metadata
        # NWBfile:
        nwbfile_args = dict(identifier=str(uuid.uuid4()), )
        nwbfile_args.update(**metadata['NWBFile'])
        nwbfile = NWBFile(**nwbfile_args)

        # Subject:
        nwbfile.subject = Subject(**metadata['Subject'])

        # Device:
        if isinstance(metadata['ophys']['Device'], list):
            for devices in metadata['ophys']['Device']:
                nwbfile.create_device(**devices)
        else:
            nwbfile.create_device(**metadata['ophys']['Device'])

        # Processing Module:
        ophys_mod = nwbfile.create_processing_module('ophys',
                                                     'contains optical physiology processed data')

        # ImageSegmentation:
        image_segmentation = ImageSegmentation(name=metadata['ophys']['ImageSegmentation']['name'])
        ophys_mod.add_data_interface(image_segmentation)

        #OPtical Channel:
        channel_names = [segext_obj.get_channel_names()]
        input_args=[[dict(name=i) for i in channel_names[k]] for k in range(segext_obj.get_num_planes())]
        for j,i in enumerate(metadata['ophys']['ImagingPlane']):
            for j2,i2 in enumerate(i['optical_channels']):
                input_args[j][j2].update(**i2)
        optical_channels=[[OpticalChannel(**input_args[k][j]) for j,i in enumerate(channel_names[k])]
                          for k in range(segext_obj.get_num_planes())]

        # ImagingPlane:
        input_kwargs = [dict(
            name=f'ImagingPlane{i}',
            description='no description',
            device=list(nwbfile.devices.values())[0],
            excitation_lambda=np.nan,
            imaging_rate=1.0,
            indicator='unknown',
            location='unknown'
        ) for i in range(segext_obj.get_num_planes())]
        for j, i in enumerate(metadata['ophys']['ImagingPlane']):
            _ = i.pop('optical_channels')
            i.update(optical_channel=optical_channels[j])
            input_kwargs[j].update(**i)#update with metadata
        imaging_planes = [nwbfile.create_imaging_plane(**i) for i in input_kwargs]

        # PlaneSegmentation:
        input_kwargs = [dict(
            name='PlaneSegmentation',
            description='output from segmenting my favorite imaging plane',
            imaging_plane=i
        ) for i in imaging_planes]
        [input_kwargs[j].update(**i)
         for j,i in enumerate(metadata['ophys']['ImageSegmentation']['plane_segmentations'])]  # update with metadata
        ps = [image_segmentation.create_plane_segmentation(**i) for i in input_kwargs]

        # ROI add:
        image_mask_list = [segext_obj.get_roi_image_masks()]
        roi_id_list = [segext_obj.get_roi_ids()]
        accepted_id_locs = [[1 if k in [segext_obj.get_accepted_list()][j] else 0 for k in i]
                            for j,i in enumerate(roi_id_list)]
        for j, ps_loop in enumerate(ps):
            [ps_loop.add_roi(id=id,image_mask=image_mask_list[j][:,:,arg_id])
             for arg_id, id in enumerate(roi_id_list[j])]
        # adding columns to ROI table:
            ps_loop.add_column(name='RoiCentroid',
                            description='x,y location of centroid of the roi in image_mask',
                               data=np.array([segext_obj.get_roi_locations().T][j]))
            ps_loop.add_column(name='Accepted',
                            description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation',
                               data=accepted_id_locs[j])

        # Fluorescence Traces:
        input_kwargs = dict(
            starting_time=0.0,
            rate=segext_obj.get_sampling_frequency(),
            unit='lumens'
        )
        container_type = [i for i in metadata['ophys'].keys() if i in ['DfOverF','Fluorescence']][0]
        f_container = eval(container_type+'()')
        ophys_mod.add_data_interface(f_container)
        roi_response_dict = segext_obj.get_traces_dict()
        c=0
        for plane_no in range(segext_obj.get_num_planes()):
            input_kwargs.update(rois=ps[plane_no].create_roi_table_region(
                description=f'region for Imaging plane{plane_no}',
                region=list(range(segext_obj.no_rois))))
            for i,j in roi_response_dict.items():
                input_kwargs.update(metadata['ophys'][container_type]['roi_response_series'][c])
                input_kwargs.update(data=j.T)
                c += 1
                f_container.create_roi_response_series(**input_kwargs)

        #create Two Photon Series: #TODO: need to validate of there are seperate movies for each plane
        input_kwargs = [dict(
            name=f'TwoPhotonSeries_{i.name}',
            description='no description',
            imaging_plane=i,
            external_file=[segext_obj.get_movie_location()],
            format='external',
            rate=segext_obj.get_sampling_frequency(),
            starting_time=0.0,
            starting_frame=[0],
            dimension=segext_obj.image_size
        ) for j,i in enumerate(imaging_planes)]
        [input_kwargs[j].update(**i) for j,i in enumerate(metadata['ophys']['TwoPhotonSeries'])]
        tps = [nwbfile.add_acquisition(TwoPhotonSeries(**i)) for i in input_kwargs]

        # adding images:
        images_dict = segext_obj.get_images_dict()
        images = Images('SegmentationImages')
        for img_name, img_no in images_dict.items():
            images.add_image(GrayscaleImage(name=img_name, data=img_no))
        ophys_mod.add(images)

        # saving NWB file:
        with NWBHDF5IO(save_path, 'w') as io:
            io.write(nwbfile)

        # test read
        with NWBHDF5IO(save_path, 'r') as io:
            io.read()
