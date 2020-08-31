import numpy as np
from roiextractors import SegmentationExtractor
import os
from roiextractors.extraction_tools import _image_mask_extractor


class Suite2pSegmentationExtractor(SegmentationExtractor):

    extractor_name = 'Suite2pSegmentationExtractor'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, fileloc, combined=False, plane_no=0):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        fileloc: str
            ~/suite2p folder location on disk
        combined: bool
            if the plane is a combined plane as in the Suite2p pipeline
        plane_no: int
            the plane for which to extract segmentation for.
        """
        SegmentationExtractor.__init__(self)
        self.combined = combined
        self.plane_no = plane_no
        self.file_path = fileloc
        self.stat = self._load_npy('stat.npy')
        self._roi_response_raw = self._load_npy('F.npy', mmap_mode='r')
        self._roi_response_neuropil = self._load_npy('Fneu.npy', mmap_mode='r')
        self._roi_response_deconvolved = self._load_npy('spks.npy', mmap_mode='r')
        self.iscell = self._load_npy('iscell.npy', mmap_mode='r')
        self.ops = self._load_npy('ops.npy').item()
        self._sampling_frequency = self.ops['fs'] * [2 if self.combined else 1][0]
        self._raw_movie_file_location = self.ops['filelist'][0]
        self.image_masks = self.get_roi_image_masks()
        self._image_correlation = self._summary_image_read('Vcorr')
        self._image_mean = self._summary_image_read('meanImg')

    def _load_npy(self, filename, mmap_mode=None):
        fpath = os.path.join(self.file_path, f'Plane{self.plane_no}', filename)
        return np.load(fpath, mmap_mode=mmap_mode)

    def get_accepted_list(self):
        return np.where(self.iscell[:,0]==1)[0]

    def get_rejected_list(self):
        return np.where(self.iscell[:,0]==0)[0]

    def _summary_image_read(self, bstr='meanImg'):
        img = None
        if bstr in self.ops:
            if bstr == 'Vcorr' or bstr == 'max_proj':
                img = np.zeros((self.ops['Ly'], self.ops['Lx']), np.float32)
                img[self.ops['yrange'][0]:self.ops['yrange'][-1],
                self.ops['xrange'][0]:self.ops['xrange'][-1]] = self.ops[bstr]
            else:
                img = self.ops[bstr]
        return img

    def _calculate_roi_locations(self):
        return np.array([j['med'] for j in self.stat]).T

    @staticmethod
    def write_segmentation(segmentation_extractor, nwb_file_path):
        return NotImplementedError

    # defining the abstract class enforced methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_num_rois(self):
        return len(self.stat)

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self._calculate_roi_locations()
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self._calculate_roi_locations()[:, roi_idx_]

    def get_num_frames(self):
        return self.ops['nframes']

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return _image_mask_extractor(self.get_roi_pixel_masks(roi_ids=roi_idx_), range(len(roi_idx_)), self.get_image_size())

    def get_roi_pixel_masks(self, roi_ids=None):
        pixel_mask = []
        for i in range(self.get_num_rois()):
            pixel_mask.append(np.vstack([self.stat[i]['ypix'],
                                      self.stat[i]['xpix'],
                                      self.stat[i]['lam']]).T)
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return [pixel_mask[i] for i in roi_idx_]

    def get_image_size(self):
        return [self.ops['Lx'], self.ops['Ly']]
