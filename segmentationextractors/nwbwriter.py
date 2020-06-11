import numpy as np
import yaml
import uuid
import os
from typing import Dict
from pynwb.ophys import OpticalChannel, ImageSegmentation, ImagingPlane, TwoPhotonSeries, Fluorescence, DfOverF
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from hdmf.data_utils import DataChunkIterator


from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject

def iter_datasetvieww(datasetview_obj):
    '''
    Generator to return a row of the array each time it is called.
    This will be wrapped with a DataChunkIterator class.

    Parameters
    ----------
    datasetview_obj: DatasetView
        2-D array to iteratively write to nwb.
    '''

    for i in range(datasetview_obj.shape[0]):
        curr_data = datasetview_obj[i]
        yield curr_data
    return

class NWBConverter:
    """
    Common conversion code factored out so it can be used by multiple conversion projects
    """

    def __init__(self, metadata, nwbfile=None, source_paths=None):
        """

        Parameters
        ----------
        metadata: dict
        nwbfile: pynwb.NWBFile
        source_paths: dict
        """
        self.metadata = metadata
        self.source_paths = source_paths
        # create self.nwbfile object
        if nwbfile is None:
            self.create_nwbfile(metadata['NWBFile'])
        else:
            self.nwbfile = nwbfile

        # add subject information
        if 'Subject' in metadata:
            self.create_subject(metadata['Subject'])

        # add devices
        self.devices = dict()
        for domain in ('Icephys', 'Ecephys', 'Ophys'):
            if domain in metadata and 'Device' in metadata[domain]:
                self.devices.update(self.create_devices(metadata[domain]['Device']))

        if 'Ecephys' in metadata:
            if 'ElectrodeGroup' in metadata['Ecephys']:
                self.create_electrode_groups(metadata['Ecephys'])

        if 'Icephys' in metadata:
            if 'Electrode' in metadata['Icephys']:
                self.ic_elecs = self.create_icephys_elecs(metadata['Icephys']['Electrode'])

    def create_nwbfile(self, metadata_nwbfile):
        """
        This method is called at __init__.
        This method can be overridden by child classes if necessary.
        Creates self.nwbfile object.

        Parameters
        ----------
        metadata_nwbfile: dict
        """
        nwbfile_args = dict(identifier=str(uuid.uuid4()),)
        nwbfile_args.update(**metadata_nwbfile)
        self.nwbfile = NWBFile(**nwbfile_args)

    def create_subject(self, metadata_subject):
        """
        This method is called at __init__.
        This method can be overridden by child classes if necessary.
        Adds information about Subject to self.nwbfile.

        Parameters
        ----------
        metadata_subject: dict
        """
        self.nwbfile.subject = Subject(**metadata_subject)

    def create_devices(self, metadata_device) -> Dict:
        """
        This method is called at __init__.
        Use metadata to create Device object(s) in the NWBFile

        Parameters
        ----------
        metadata_device: list or dict

        Returns
        -------
        dict

        """
        if isinstance(metadata_device, list):
            devices = dict()
            [devices.update(self.create_devices(idevice_meta)) for idevice_meta in metadata_device]
            return devices
        else:
            if 'tag' in metadata_device:
                key = metadata_device['tag']
            else:
                key = metadata_device['name']
            return {key: self.nwbfile.create_device(**metadata_device)}

    def create_electrode_groups(self, metadata_ecephys):
        """
        This method is called at __init__.
        Use metadata to create ElectrodeGroup object(s) in the NWBFile

        Parameters
        ----------
        metadata_ecephys : dict
            Dict with key:value pairs for defining the Ecephys group from where this
            ElectrodeGroup belongs. This should contain keys for required groups
            such as 'Device', 'ElectrodeGroup', etc.
        """
        for metadata_elec_group in metadata_ecephys['ElectrodeGroup']:
            eg_name = metadata_elec_group['name']
            # Tests if ElectrodeGroup already exists
            aux = [i.name == eg_name for i in self.nwbfile.children]
            if any(aux):
                print(eg_name + ' already exists in current NWBFile.')
            else:
                device_name = metadata_elec_group['device']
                if device_name in self.nwbfile.devices:
                    device = self.nwbfile.devices[device_name]
                else:
                    print('Device ', device_name, ' for ElectrodeGroup ', eg_name, ' does not exist.')
                    print('Make sure ', device_name, ' is defined in metadata.')

                eg_description = metadata_elec_group['description']
                eg_location = metadata_elec_group['location']
                self.nwbfile.create_electrode_group(
                    name=eg_name,
                    location=eg_location,
                    device=device,
                    description=eg_description
                )

    def create_electrodes_ecephys(self):
        """
        This method should be overridden by child classes if necessary.
        Create electrodes in the NWBFile.
        """
        pass

    def create_icephys_elecs(self, elec_meta) -> Dict:
        """
        Use metadata to generate intracellular electrode object(s) in the NWBFile

        Parameters
        ----------
        elec_meta: list or dict

        Returns
        -------
        list

        """
        if isinstance(elec_meta, list):
            elecs = dict()
            [elecs.update(self.create_icephys_elecs(**ielec_meta)) for ielec_meta in elec_meta]
            return elecs

        else:
            if len(self.devices) == 1:
                device = list(self.devices.values())[0]
            elif elec_meta['device'] in self.devices:
                device = self.devices[elec_meta['device']]
            else:
                raise ValueError('device not found for icephys electrode {}'.format(elec_meta['name']))
            if 'tag' in elec_meta:
                key = elec_meta['tag']
            else:
                key = elec_meta['name']
            return {key: self.nwbfile.create_ic_electrode(device=device, **elec_meta)}

    def create_trials_from_df(self, df):
        """
        This method should not be overridden.
        Creates a trials table in self.nwbfile from a Pandas DataFrame.

        Parameters
        ----------
        df: Pandas DataFrame
        """
        # Tests if trials table already exists
        if self.nwbfile.trials is not None:
            print("Trials table already exist in current nwb file.\n"
                  "Use 'add_trials_columns_from_df' to include new columns.\n"
                  "Use 'add_trials_from_df' to include new trials.")
            pass
        # Tests if required column names are present in df
        if 'start_time' not in df.columns:
            print("Required column 'start_time' not present in DataFrame.")
            pass
        if 'stop_time' not in df.columns:
            print("Required column 'stop_time' not present in DataFrame.")
            pass
        # Creates new columns
        for colname in df.columns:
            if colname not in ['start_time', 'stop_time']:
                # Indexed columns should be of type 'object' in the dataframe
                if df[colname].dtype == 'object':
                    self.nwbfile.add_trial_column(name=colname, description='no description', index=True)
                else:
                    self.nwbfile.add_trial_column(name=colname, description='no description')
        # Populates trials table from df values
        for index, row in df.iterrows():
            self.nwbfile.add_trial(**dict(row))

    def add_trials_from_df(self, df):
        """
        This method should not be overridden.
        Adds trials from a Pandas DataFrame to existing trials table in self.nwbfile.

        Parameters
        ----------
        df: Pandas DataFrame
        """
        # Tests for mismatch between trials table columns and dataframe columns
        A = set(self.nwbfile.trials.colnames)
        B = set(df.columns)
        if len(A - B) > 0:
            print("Missing columns in DataFrame: ", A - B)
            pass
        if len(B - A) > 0:
            print("NWBFile trials table does not contain: ", B - A)
            pass
        # Adds trials from df values
        for index, row in df.iterrows():
            self.nwbfile.add_trial(**dict(row))

    def add_trials_columns_from_df(self, df):
        """
        This method should not be overridden.
        Adds trials columns from a Pandas DataFrame to existing trials table in self.nwbfile.

        Parameters
        ----------
        df: Pandas DataFrame
        """
        # Tests if dataframe columns already exist in nwbfile trials table
        A = set(self.nwbfile.trials.colnames)
        B = set(df.columns)
        intersection = A.intersection(B)
        if len(intersection) > 0:
            print("These columns already exist in nwbfile trials: ", intersection)
            pass
        # Adds trials columns with data from df values
        for (colname, coldata) in df.iteritems():
            # Indexed columns should be of type 'object' in the dataframe
            if df[colname].dtype == 'object':
                index = True
            else:
                index = False
            self.nwbfile.add_trial_column(
                name=colname,
                description='no description',
                data=coldata,
                index=index
            )

    def save(self, to_path, read_check=True):
        """
        This method should not be overridden.
        Saves object self.nwbfile.

        Parameters
        ----------
        to_path: str
        read_check: bool
            If True, try to read the file after writing
        """
        with NWBHDF5IO(to_path, 'w') as io:
            io.write(self.nwbfile)

        if read_check:
            with NWBHDF5IO(to_path, 'r') as io:
                io.read()

    def check_module(self, name, description=None):
        """
        Check if processing module exists. If not, create it. Then return module

        Parameters
        ----------
        name: str
        description: str | None (optional)

        Returns
        -------
        pynwb.module

        """

        if name in self.nwbfile.processing:
            return self.nwbfile.processing[name]
        else:
            if description is None:
                description = name
            return self.nwbfile.create_processing_module(name, description)



class OphysNWBConverter(NWBConverter):

    def __init__(self, metadata, nwbfile=None, source_paths=None):

        super(OphysNWBConverter, self).__init__(metadata, nwbfile=nwbfile, source_paths=source_paths)

        # device = Device('microscope')
        # self.nwbfile.add_device(device)
        # self.imaging_plane = self.add_imaging_plane()
        self.imaging_planes = None
        if self.imaging_plane_set:
            self.add_imaging_plane()
        # self.two_photon_series = self.create_two_photon_series()
        ophys_mods = [j for i, j in self.nwbfile.processing.items() if i in ['ophys', 'Ophys']]
        if len(ophys_mods) > 0:
            self.ophys_mod = ophys_mods[0]
        else:
            self.ophys_mod = self.nwbfile.create_processing_module('Ophys',
                                                                   'contains optical physiology processed data')

    def create_optical_channel(self, metadata=None):

        if metadata == []:
            metadata = None
        input_kwargs = dict(
            name='OpticalChannel',
            description='no description',
            emission_lambda=np.nan
        )

        if metadata:
            input_kwargs.update(metadata)

        return OpticalChannel(**input_kwargs)

    def add_imaging_plane(self, metadata=None):
        """
        Creates an imaging plane. Converts the device and optical channel attributes in the metadata file to an actual
        object.
        Parameters
        ----------
        metadata

        Returns
        -------

        """
        planes_list = []
        input_kwargs = dict(
            name='ImagingPlane',
            description='no description',
            device=self.devices[list(self.devices.keys())[0]],
            excitation_lambda=np.nan,
            imaging_rate=1.0,
            indicator='unknown',
            location='unknown'
        )
        c = 0
        if 'Ophys' in self.metadata and 'ImagingPlane' in self.metadata['Ophys']:
            if metadata is None:
                metadata = [dict()]*len(self.metadata['Ophys']['ImagingPlane'])
            elif isinstance(metadata,
                            dict):  # metadata should ideally be of the length of number of imaging planes in the metadata file input
                metadata = [metadata]*len(self.metadata['Ophys']['ImagingPlane'])
            for i in self.metadata['Ophys']['ImagingPlane']:
                # get device object
                if i.get('device'):
                    i['device'] = self.nwbfile.devices[i['device']]
                else:
                    i['device'] = self.devices[list(self.devices.keys())[0]]
                # get optical channel object
                if i.get('optical_channel'):
                    if len(i[
                               'optical_channel']) > 0:  # calling the bui creates an empty optical channel list when there was none.
                        i['optical_channel'] = [self.create_optical_channel(metadata=i) for i in i['optical_channel']]
                else:
                    i['optical_channel'] = self.create_optical_channel()

                input_kwargs.update(i)
                input_kwargs.update(metadata[c])
                planes_list.extend([self.nwbfile.create_imaging_plane(**input_kwargs)])
                c += 1
        else:
            if not isinstance(metadata, list):
                if metadata is not None:
                    metadata = [metadata]
                else:
                    metadata = [dict()]
            for i in metadata:
                input_kwargs.update(i)
                planes_list.extend([self.nwbfile.create_imaging_plane(**input_kwargs)])
        if self.imaging_planes is not None:
            self.imaging_planes.extend(planes_list)
        else:
            self.imaging_planes = planes_list
        return planes_list


class ProcessedOphysNWBConverter(OphysNWBConverter):

    def __init__(self, metadata, nwbfile=None, source_paths=None, imaging_plane_set=True):
        self.imaging_plane_set = imaging_plane_set
        super(ProcessedOphysNWBConverter, self).__init__(metadata, nwbfile=nwbfile, source_paths=source_paths)
        self.image_segmentation = self.create_image_segmentation()
        if self.image_segmentation.name not in [i.name for i in self.ophys_mod.children]:
            self.ophys_mod.add_data_interface(self.image_segmentation)
        else:
            self.image_segmentation = self.ophys_mod[self.image_segmentation.name]
        self.ps_list = []

    def create_image_segmentation(self):
        if 'ImageSegmentation' in self.metadata.get('Ophys', 'not_found'):
            return ImageSegmentation(name=self.metadata['Ophys']['ImageSegmentation']['name'])
        else:
            return ImageSegmentation()

    def create_plane_segmentation(self, metadata=None):
        """
        Create multiple plane segmentations.
        Parameters
        ----------
        metadata: list
            List of dicts with plane segmentation arguments
        Returns
        -------

        """
        input_kwargs = dict(
            name='PlaneSegmentation',
            description='output from segmenting my favorite imaging plane',
            imaging_plane=self.imaging_planes[0]  # pick a default one if none specified.
        )
        if metadata:
            if not isinstance(metadata, list):
                metadata = [metadata]
            for i in metadata:  # multiple plane segmentations
                if i.get('imaging_planes'):
                    if i['imaging_planes'] in [i.name for i in self.imaging_planes]:
                        current_img_plane = self.nwbfile.get_imaging_plane(name=i['imaging_plane'])
                    else:
                        current_img_plane = self.add_imaging_plane(dict(name=i['imaging_plane']))
                else:
                    current_img_plane = self.add_imaging_plane(dict(name=i['name']))
                input_kwargs.update(i)
                if input_kwargs['name'] not in self.image_segmentation.keys():
                    self.ps_list.append(self.image_segmentation.create_plane_segmentation(**input_kwargs))

        elif 'Ophys' in self.metadata and 'plane_segmentations' in self.metadata['Ophys']['ImageSegmentation']:
            for i in self.metadata['Ophys']['ImageSegmentation']['plane_segmentations']:
                metadata = i
                if metadata.get('imaging_planes'):
                    metadata['imaging_plane'] = self.nwbfile.get_imaging_plane(name=metadata['imaging_planes'])
                    metadata.pop('imaging_planes')  # TODO this will change when loopis implemented
                else:
                    metadata['imaging_plane'] = self.nwbfile.get_imaging_plane(
                        name=list(self.nwbfile.imaging_planes.keys())[0])

                input_kwargs.update(metadata)
                if input_kwargs['name'] not in self.image_segmentation.name:
                    self.ps_list.append(self.image_segmentation.create_plane_segmentation(**input_kwargs))


class SegmentationExtractor2NWBConverter(ProcessedOphysNWBConverter):

    def __init__(self, segext_obj, nwbfile, metadata):
        """
        Conversion of Sima segmentationExtractor object to an NWB file using GUI
        Parameters
        ----------
        segext_obj: SEgmentationExtractor object
            object to write nwb file from
        nwbfile: NWBfile
            pre-existing nwb file to append all the data to
        metadata: dict
            dict of metadata that will be used to create fields in the nwb file.
        """
        self.segext_obj = segext_obj
        source_path = self.segext_obj.filepath
        if isinstance(metadata,str):
            with open(metadata,'r') as f:
                metadata = yaml.safe_load(f)
        super().__init__(metadata,nwbfile,source_path, imaging_plane_set=False)

    def create_two_photon_series(self, metadata=None, imaging_plane=None):
        if imaging_plane is None:
            if self.nwbfile.imaging_planes:
                imaging_plane = self.nwbfile.imaging_planes[list(self.nwbfile.imaging_planes.keys())[0]]
            else:
                imaging_plane = self.add_imaging_plane()

        input_kwargs = dict(
            name='TwoPhotonSeries',
            description='no description',
            imaging_plane=imaging_plane,
            external_file=[self.segext_obj.get_movie_location()],
            format='external',
            rate=self.segext_obj.get_sampling_frequency(),
            starting_time=0.0,
            starting_frame=[0],
            dimension=self.segext_obj.image_dims
        )

        if metadata is None and 'Ophys' in self.metadata and 'TwoPhotonSeries' in self.metadata['Ophys']:
            metadata = self.metadata['Ophys']['TwoPhotonSeries']
            if len(metadata['imaging_planes']) > 0:
                metadata['imaging_planes'] = self.nwbfile.get_imaging_plane(name=metadata['imaging_planes'][0])
            else:
                metadata['imaging_planes'] = imaging_plane
        if metadata is not None:
            input_kwargs.update(metadata)
        if input_kwargs['name'] not in self.nwbfile.acquisition.keys():
            ret = self.nwbfile.add_acquisition(TwoPhotonSeries(**input_kwargs))
        else:
            ret = self.nwbfile.acquisition[input_kwargs['name']]
        return ret

    def create_imaging_plane(self, optical_channel_list=None):
        """
        :param optical_channel_list:
        :return:
        """
        if not optical_channel_list:
            optical_channel_list = []
        channel_names = self.segext_obj.get_channel_names()
        for i in channel_names:
            optical_channel_list.append(self.create_optical_channel(dict(name=i)))
        self.add_imaging_plane(metadata=dict(optical_channel=optical_channel_list))

    def add_rois(self):
        ps = self.ps_list[0]
        pixel_mask_exist = self.segext_obj.get_pixel_masks() is not None
        for i, roiid in enumerate(self.segext_obj.roi_idx):
            if pixel_mask_exist:
                ps.add_roi(id=roiid,
                           pixel_mask=self.segext_obj.get_pixel_masks(ROI_ids=[roiid])[:,0:-1])
            else:
                ps.add_roi(id=roiid,
                           image_mask=self.segext_obj.get_image_masks(ROI_ids=[roiid]))

    def add_roi_table_column(self):
        self.ps_list[0].add_column(name='RoiCentroid',
                                   description='x,y location of centroid of the roi in image_mask',
                                   data=np.array(self.segext_obj.get_roi_locations()).T)
        accepted = np.zeros(self.segext_obj.no_rois)
        for j,i in enumerate(self.segext_obj.roi_idx):
            if i in self.segext_obj.accepted_list:
                accepted[j] = 1
        self.ps_list[0].add_column(name='Accepted',
                                   description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation',
                                   data=accepted)

    def add_fluorescence_traces(self, metadata=None):
        """
        Create fluorescence traces for the nwbfile
        Parameters
        ----------
        metadata: list
            list of dictionaries with keys/words same as roi_response_series input arguments.
        Returns
        -------
        None
        """
        input_kwargs = dict(
            rois=self.create_roi_table_region(list(range(self.segext_obj.no_rois))),
            starting_time=0.0,
            rate=self.segext_obj.get_sampling_frequency(),
            unit='lumens'
        )
        if metadata:
            metadata_iter = metadata
            container_func = Fluorescence
        elif metadata is None and 'Ophys' in self.metadata and 'DfOverF' in self.metadata['Ophys'] \
                and 'roi_response_series' in self.metadata['Ophys']['DfOverF'] \
                and len(self.metadata['Ophys']['DfOverF']['roi_response_series']) > 0:
            metadata_iter = self.metadata['Ophys']['DfOverF']['roi_response_series']
            container_func = DfOverF
        elif metadata is None and 'Ophys' in self.metadata and 'Fluorescence' in self.metadata['Ophys'] \
                and 'roi_response_series' in self.metadata['Ophys']['Fluorescence'] \
                and len(self.metadata['Ophys']['Fluorescence']['roi_response_series']) > 0:
            metadata_iter = self.metadata['Ophys']['Fluorescence']['roi_response_series']
            container_func = Fluorescence
        else:
            metadata_iter = [input_kwargs]
            container_func = Fluorescence

        for i in metadata_iter:
            i.update(
                {'rois': self.create_roi_table_region(list(range(self.segext_obj.no_rois)))})
        # Create the main fluorescence container
        fl = container_func()
        self.ophys_mod.add_data_interface(fl)
        # Iteratively populate fluo container with various roi_resp_series
        for i in metadata_iter:
            i.update(**input_kwargs)
            i.update(
                name=i['name'],
                data=DataChunkIterator(data=iter_datasetvieww(
                    self.segext_obj.get_traces_info()[i['name']])
                )
            )
            fl.create_roi_response_series(**i)

    def create_roi_table_region(self, rois, region_name='NeuronROIs'):
        return self.ps_list[0].create_roi_table_region(region_name, region=rois)

    def add_images(self):
        images_dict = self.segext_obj.get_images()
        if 'Ophys' in self.metadata and list(images_dict.keys())[0] in self.metadata['Ophys']:
            images_names = [i['name'] for i in self.metadata['Ophys'][list(images_dict.keys())[0]]]
        else:
            images_names = list(list(images_dict.values())[0].keys())
        if images_dict is not None:
            image_names_obj = list(images_dict.keys())
            if 'Images' in self.metadata['Ophys'] and len(self.metadata['Ophys']['Images'])>0:
                image_names_obj = [i['name'] for i in self.metadata['Ophys']['Images']]
            for img_set_name, img_set in images_dict.items():
                images = Images(img_set_name)
                for img_name, img_no in img_set.items():
                    if img_name in images_names:
                        images.add_image(GrayscaleImage(name=img_name,data=img_no))
                self.ophys_mod.add(images)

    def run_conversion(self):
        """
        To populate the nwb file completely.
        """
        self.create_imaging_plane()
        self.create_plane_segmentation()
        self.add_rois()
        self.add_roi_table_column()
        self.add_fluorescence_traces()
        self.create_two_photon_series(imaging_plane=list(self.nwbfile.imaging_planes.values())[0])
        self.add_images()


