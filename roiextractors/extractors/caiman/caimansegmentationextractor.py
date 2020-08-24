import numpy as np
import h5py
from lazy_ops import DatasetView
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import _pixel_mask_extractor
import os
from scipy.sparse import csc_matrix

class CaimanSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    """
    extractor_name = 'CaimanSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath: str
            The location of the folder containing caiman *.hdmf output file.
        """
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        self._dataset_file = self._file_extractor_read()
        self._roi_response = self._trace_extractor_read('F_dff')
        self._roi_response_fluorescence = self._roi_response
        self._roi_response_neuropil = self._trace_extractor_read('C')
        self._roi_response_deconvolved = self._trace_extractor_read('S')
        self._images_correlation = self._summary_image_read()
        self._raw_movie_file_location = self._dataset_file['params']['data']['fnames'][0].decode('utf-8')
        self._sampling_frequency = self._dataset_file['params']['data']['fr'][()]
        self.image_masks = self._image_mask_sparse_read()[-1]

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        return f

    def _image_mask_sparse_read(self):
        roi_ids = self._dataset_file['estimates']['A']['indices']
        masks = self._dataset_file['estimates']['A']['data']
        ids = self._dataset_file['estimates']['A']['indptr']
        _image_mask = np.reshape(csc_matrix((masks, roi_ids, ids), shape=(np.prod(self.get_image_size()),self.no_rois)).toarray(),
            [self.get_image_size()[0],self.get_image_size()[1],-1],order='F')
        return masks, roi_ids, ids, _image_mask

    def _trace_extractor_read(self, field):
        if self._dataset_file['estimates'].get(field):
            return self._dataset_file['estimates'][field] # lazy read dataset)
        else:
            return None

    def _summary_image_read(self):
        if self._dataset_file['estimates'].get('Cn'):
            return np.array(self._dataset_file['estimates']['Cn']).T
        else:
            return None

    def get_accepted_list(self):
        accepted = self._dataset_file['estimates']['idx_components']
        if len(accepted.shape)==0:
            accepted = list(range(self.no_rois))
        return accepted

    def get_rejected_list(self):
        rejected = self._dataset_file['estimates']['idx_components_bad']
        if len(rejected.shape) == 0:
            rejected = [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]
        return rejected

    @property
    def roi_locations(self):
        _masks, _mask_roi_ids, _mask_ids, _ = self._image_mask_sparse_read()
        roi_location = np.ndarray([2, self.no_rois], dtype='int')
        for i in range(self.no_rois):
            max_mask_roi_id = _mask_roi_ids[_mask_ids[i]+np.argmax(
                _masks[_mask_ids[i]:_mask_ids[i+1]]
            )]
            roi_location[:, i] = [((max_mask_roi_id+1)%(self.image_size[0]+1))-1,#assuming order='F'
                                  ((max_mask_roi_id+1)//(self.image_size[0]+1))]
            if roi_location[0,i]<0:
                roi_location[0,i]=0
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath, **kwargs):
        plane_no = kwargs.get('plane_no', 0)
        filename = os.path.basename(savepath)
        savepath_folder = os.path.join(os.path.dirname(savepath), f'Plane_{plane_no}')
        savepath = os.path.join(savepath_folder, filename)
        if savepath.split('.')[-1]!='hdf5':
            raise ValueError('filetype to save must be *.hdf5')
        with h5py.File(savepath,'w') as f:
            #create base groups:
            estimates = f.create_group('estimates')
            params = f.create_group('params')
            #adding to estimates:
            if segmentation_object._roi_response_neuropil is not None:
                estimates.create_dataset('C',data=segmentation_object._roi_response_neuropil)
            estimates.create_dataset('F_dff', data=segmentation_object._roi_response_fluorescence)
            if segmentation_object._roi_response_deconvolved is not None:
                estimates.create_dataset('S', data=segmentation_object._roi_response_deconvolved)
            if segmentation_object._images_mean is not None:
                estimates.create_dataset('Cn', data=segmentation_object._images_mean)
            estimates.create_dataset('idx_components', data=np.array(segmentation_object.get_accepted_list()))
            estimates.create_dataset('idx_components_bad', data=np.array(segmentation_object.get_rejected_list()))

            #adding image_masks:
            image_mask_data = np.reshape(segmentation_object.get_roi_image_masks(),[-1,segmentation_object.get_num_rois()],order='F')
            image_mask_csc = csc_matrix(image_mask_data)
            estimates.create_dataset('A/data',data=image_mask_csc.data)
            estimates.create_dataset('A/indptr', data=image_mask_csc.indptr)
            estimates.create_dataset('A/indices', data=image_mask_csc.indices)
            estimates.create_dataset('A/shape', data=image_mask_csc.shape)

            #adding params:
            params.create_dataset('data/fr',data=segmentation_object._sampling_frequency)
            params.create_dataset('data/fnames', data=[bytes(segmentation_object._raw_movie_file_location,'utf-8')])
            params.create_dataset('data/dims', data=segmentation_object.get_image_size())
            f.create_dataset('dims',data=segmentation_object.get_image_size())

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
        return self.image_masks[:, :, roi_idx_]

    def get_image_size(self):
        return self._dataset_file['params']['data']['dims'][()]
