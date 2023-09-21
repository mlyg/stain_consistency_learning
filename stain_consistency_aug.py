import pickle
import numpy as np
from scipy import linalg


class StainConsistencyAug():
    """Stain consistency augmentation."""
    def __init__(self, param_file=None):
        """
        Args:
            param_file (pickle): file containing distribution parameters
        """
        self.param_file = param_file
    
    def preprocess(self, images):
        """Preprocess dataset to obtain stain and concentration distribution parameters"""
        # load stain and concentration distribution matrices
        if self.param_file is not None:
            with open(self.param_file, 'rb') as f:
                data = pickle.load(f)
                self.conc_mean = data['conc_mean']
                self.conc_std = data['conc_std']
                self.stain_mean = data['stain_mean']
                self.stain_std = data['stain_std']
            return
        
        conc_stats = np.zeros((len(images), 2, 3))
        stain_stats = np.zeros((len(images), 3, 3))
        
        # extract stain and concentration matrices per image
        for idx, img in enumerate(images):
            stain_matrix = self._get_stain_matrix(img)
            stain_stats[idx] = stain_matrix
            conc_stats[idx], _ = self._get_conc_matrix(img, stain_matrix)
        
        conc_stats = conc_stats.reshape(-1, 6)
        stain_stats = stain_stats.reshape(-1, 9)
 
        self.conc_mean = np.mean(conc_stats, axis=0)
        self.conc_std = np.cov(conc_stats, rowvar=0)
        
        self.stain_mean = np.mean(stain_stats, axis=0)
        self.stain_std = np.cov(stain_stats, rowvar=0)
    
    def convert_OD_to_RGB(self, img):
        """Convert image from optical density space to RGB space"""
        img = np.maximum(img, 1e-6)
        return (255 * np.exp(-1 * img)).astype(np.uint8)
    
    def convert_RGB_to_OD(self, img):
        """Convert image from RGB space to optical density space"""
        mask = (img == 0)
        img[mask] = 1
        return np.maximum(-1 * np.log(img / 255), 1e-6)
    
    def normalize_matrix_rows(self, A):
        """Normalise matrix"""
        return A / np.linalg.norm(A, axis=1)[:, None]
    
    def _get_stain_matrix(self, img, angular_percentile=99, beta=0.15):
        """Use Macenko et al. method to extract stain colour matrix
        Implementation based on: https://github.com/Peter554/StainTools
        """
        od_img = self.convert_RGB_to_OD(img).reshape((-1, 3))

        # remove any transparent pixels
        od_img = od_img[~np.any(od_img<beta, axis=1)]
        
        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(od_img, rowvar=False))

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        if V[0, 0] < 0: V[:, 0] *= -1
        if V[0, 1] < 0: V[:, 1] *= -1

        # Project on this basis.
        That = np.dot(od_img, V)

        # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])

        # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        if v1[0] > v2[0]:
            stain_matrix = np.array([v1, v2])
        else:
            stain_matrix = np.array([v2, v1])

        stain_matrix = self.normalize_matrix_rows(stain_matrix)
        # add third row to stain colour matrix
        stain_matrix = np.concatenate([stain_matrix, np.cross(stain_matrix[0, :], stain_matrix[1, :]).reshape(1, -1)], axis=0)   
        return stain_matrix
    
    def _get_conc_matrix(self, img, stain_matrix):
        """ Use stain colour matrix to obtain stain concentration matrix"""
        conc = self.convert_RGB_to_OD(img) @ linalg.inv(stain_matrix)
        np.maximum(conc, 0, out=conc)
        
        conc = conc.reshape((-1, 3))
        
        means = np.mean(conc,axis=0)
        stds = np.std(conc,axis=0)
        
        return np.stack([means, stds]), conc
    
    def _sample(self):
        """Sample stain colour and concentration values from normal distribution"""
        rng = np.random.default_rng()
        
        conc = rng.multivariate_normal(self.conc_mean, self.conc_std).reshape(2, 3)
        stain = rng.multivariate_normal(self.stain_mean, self.stain_std).reshape(3, 3)
        
        return np.abs(conc), stain
    
    def __call__(self, img, stain_matrix=None, conc=None, conc_stats=None, sample_stain=None, sample_conc=None, return_mat=False):
        """Apply stain augmentation
        
        Args:
            stain_matrix: stain colour matrix for image
            conc_matrix: concentration matrix for image
            conc_stats: concentration matrix statistics
            sample_stain: sampled stain colour matrix
            sample_conc: sampled concentration matrix statistics
            return_mat: whether to return intermediate values alongside augmented image
        
        """
        if sample_stain is None and sample_conc is  None:
            sample_conc, sample_stain = self._sample()
        elif sample_stain is None:
            _, sample_stain = self._sample()
        elif sample_conc is None:
            sample_conc, _ = self._sample()
        
        if stain_matrix is None:
            stain_matrix = self._get_stain_matrix(img)
        
        if conc is None or conc_stats is None:
            conc_stats, conc = self._get_conc_matrix(img, stain_matrix)
        
        orig_conc = conc.copy()
        
        # normalise concentration matrix with sampled statistics
        conc[:,0] = ((conc[:,0] - conc_stats[0][0]) * (sample_conc[1][0] / (conc_stats[1][0]+1e-6))) + sample_conc[0][0]
        conc[:,1] = ((conc[:,1] - conc_stats[0][1]) * (sample_conc[1][1] / (conc_stats[1][1]+1e-6))) + sample_conc[0][1]
        conc[:,2] = ((conc[:,2] - conc_stats[0][2]) * (sample_conc[1][2] / (conc_stats[1][2]+1e-6))) + sample_conc[0][2]
        
        # use sampled stain colour matrix
        aug_img = 255 * np.exp(-1 * np.dot(conc, sample_stain))
        
        img = aug_img.reshape(img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        if return_mat:
            return img, sample_stain, sample_conc, stain_matrix, orig_conc, conc_stats
        else:
            return img