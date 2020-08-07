import numpy as np
from segmentationextractors.segmentationextractor import SegmentationExtractor
import os


class Suite2pSegmentationExtractor(SegmentationExtractor):
    def __init__(self, fileloc, combined=False):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        op: dict
            options that need the suite 2p file takes as arguments.
        db: dict
            db overwrites any ops (allows for experiment specific settings)
        """
        self.combined = combined
        self.filepath = fileloc
        self.no_planes = None
        self.ops = [i.item() for i in self._load_npy('ops.npy', folder_no=1)]

        self.no_planes = 1 if self.combined else self.ops[0]['nplanes']
        self.ops = [i.item() for i in self._load_npy('ops.npy')]
        self.no_channels = [self.ops[i]['nchannels'] for i in range(self.no_planes)]
        self.stat = self._load_npy('stat.npy')
        self.F = self._load_npy('F.npy',mmap_mode='r')
        self.Fneu = self._load_npy('Fneu.npy',mmap_mode='r')
        self.spks = self._load_npy('spks.npy',mmap_mode='r')
        self.iscell = self._load_npy('iscell.npy',mmap_mode='r')
        self.roi_resp_dict = {'Fluorescence': self.F[0:self.no_planes],
                              'Neuropil':self.Fneu[0:self.no_planes],
                              'Deconvolved': self.spks[0:self.no_planes]}
        self.rois_per_plane = [i.shape[0] for i in self.iscell]
        self.raw_images = None
        self.roi_response = None

    def _load_npy(self, filename, mmap_mode=None, folder_no=None):
        loop_nos = folder_no if folder_no else self.no_planes
        ret_val = [[None]]*loop_nos
        for i in range(loop_nos):
            fold_name = 'combined' if self.combined else 'Plane{}'
            fpath = os.path.join(self.filepath, fold_name.format(i), filename)
            ret_val[i] = np.load(fpath,
                                 mmap_mode=mmap_mode,
                                 allow_pickle=not mmap_mode and True)
        return ret_val

    @property
    def image_dims(self):
        return [[self.ops[i]['Lx'], self.ops[i]['Ly']] for i in range(self.no_planes)]

    @property
    def no_rois(self):
        return [len(i) for i in self.stat]

    @property
    def roi_idx(self):
        return [[i for i in range(r)] for r in self.no_rois]

    @property
    def accepted_list(self):
        plane_wise = [np.where(i[:, 0] == 1)[0] for i in self.iscell]
        # return np.array([plane_wise[0].tolist(),(len(self.stat[0])+plane_wise[0]).tolist()])[0:self.no_planes].squeeze().tolist()
        return plane_wise

    @property
    def rejected_list(self):
        plane_wise = [np.where(i[:, 0] == 0)[0] for i in self.iscell]
        # return np.array([plane_wise[0].tolist(), (len(self.stat[0]) + plane_wise[0]).tolist()])[0:self.no_planes].squeeze().tolist()
        return plane_wise

    @property
    def roi_locs(self):
        plane_wise = [[j['med'] for j in i] for i in self.stat]
        # ret_val = []
        # [ret_val.extend(i) for i in plane_wise]
        # return np.array(ret_val).T.tolist()
        return plane_wise

    @property
    def num_of_frames(self):
        return [i['nframes'] for i in self.ops]

    @property
    def samp_freq(self):
        return self.ops[0]['fs']*self.no_planes

    @staticmethod
    def write_segmentation(segext_obj, savepath, metadata_dict=None, **kwargs):
        return NotImplementedError

    # defining the abstract class enforced methods:
    def get_roi_ids(self):
        return self.roi_idx

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self, ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            # ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            # ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            # ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
            # return self.roi_locs[:, ROI_idx_]
            return [[self.roi_locs[j][ids] for ids in i]for j,i in enumerate(ROI_ids)]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None, name=None):
        if name is None:
            name = 'Fluorescence'
            print(f'returning traces for {name}')
        if start_frame is None:
            start_frame = [0 for i in range(self.no_planes)]
        if end_frame is None:
            end_frame = [i+1 for i in self.get_num_frames()]
        if ROI_ids is None:
            ROI_ids = [list(range(i)) for i in self.get_num_rois()]
        # else:
        #     ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
        #     ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
        #     ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        # return np.concatenate(self.roi_resp_dict[name])[[ROI_idx_],start_frame:end_frame].squeeze()
        return [np.array([self.roi_resp_dict[name][i][id,start_frame[i]:end_frame[i]] for id in ROI_ids[i]])
                for i in range(self.no_planes)]

    def get_traces_info(self):
        roi_resp_dict = dict()
        name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved']
        for i in name_strs:
            roi_resp_dict[i] = self.get_traces(name=i)
        return roi_resp_dict

    def get_image_masks(self, ROI_ids=None):
        return None

    def get_pixel_masks(self, ROI_ids=None):
        pixel_mask = [[None for k in range(self.rois_per_plane[j])] for j in range(self.no_planes)]
        for i in range(self.no_planes):
            for j in range(self.rois_per_plane[i]):
                pixel_mask[i][j] = np.array([self.stat[i][j]['ypix'],
                                         self.stat[i][j]['xpix'],
                                         self.stat[i][j]['lam'],
                                         j*np.ones(self.stat[i][j]['lam'].size)]).T
        if ROI_ids is None:
            ROI_ids = [list(range(i)) for i in self.get_num_rois()]
        # else:
            # ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            # ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            # ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return [np.concatenate([pixel_mask[i][k] for k in ROI_ids[i]]) for i in range(self.no_planes)]

    def get_images(self):
        bg_strs = ['meanImg', 'Vcorr', 'max_proj']
        out_dict = dict()
        for noplanes in range(self.no_planes):
            name = f'Plane{noplanes}'
            out_dict[name]=dict()
            for bstr in bg_strs:
                if bstr in self.ops[noplanes]:
                    if bstr == 'Vcorr' or bstr == 'max_proj':
                        img = np.zeros((self.ops[noplanes]['Ly'], self.ops[noplanes]['Lx']), np.float32)
                        img[self.ops[noplanes]['yrange'][0]:self.ops[noplanes]['yrange'][-1],
                        self.ops[noplanes]['xrange'][0]:self.ops[noplanes]['xrange'][-1]] = self.ops[noplanes][bstr]
                    else:
                        img = self.ops[noplanes][bstr]
                    # out_dict['Background0'].update({bstr:img})
                    out_dict[name].update({bstr:img})
        return out_dict

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return os.path.abspath(os.path.join(self.filepath, os.path.pardir))

    def get_channel_names(self):
        return [[f'OpticalChannel{i}' for i in range(self.no_channels[k])] for k in range(self.no_planes)]

    def get_num_channels(self):
        return self.no_channels
