import numpy as np
import math
import sys
import tensorflow as tf
import tensorflow_probability as tfp
import joblib


def get_dist_class_by_name(dist_type):
    if dist_type == "gauss_diag":
        return DiagonalGaussian
    else:
        sys.exit("Unknown distribution type")


def recover_distribution(dist_type, dist_params_dict):
    out_dist = get_dist_class_by_name(dist_type)
    return out_dist.recover(dist_params_dict)


def extract_parameters(dist):
    assert isinstance(dist, tfp.distributions.Distribution)
    out_dict = {}
    out_dict['type'] = type(dist)
    out_dict['params'] = {}
    for key, value in dist.parameters.items():
        if isinstance(value, tfp.distributions.Distribution):
            out_dict['params'][key] = extract_parameters(value)
        else:
            out_dict['params'][key] = value
    return out_dict

def reconstruct_distribution(dist_info):
    dist_type   = dist_info['type']
    dist_params = dist_info['params']
    reconstructed_params = {}
    for key, value in dist_params.items():
        if isinstance(value, dict) and 'type' in value and 'params' in value:
            print(value)
            reconstructed_params[key] = reconstruct_distribution(value)
        else:
            reconstructed_params[key] = value
    return dist_type(**reconstructed_params)

def save_tfp_distribution(distrib, file_name):
    out_dict = extract_parameters(distrib)
    with open(file_name, 'wb+') as file:
        joblib.dump(out_dict, file)


def load_tfp_distribution(file_name):
    dist_info_dict = joblib.load(file_name)
    distribution = reconstruct_distribution(dist_info_dict)
    return distribution
    
    




class DiagonalGaussian:
    def __init__(self, num_dims, std_scaler):
        self.num_dims   = num_dims
        self.std_scaler = std_scaler
        self.var_scaler = self.std_scaler * self.std_scaler

        self.gauss_mean = np.zeros(self.num_dims)
        self.var_vec    = np.ones(self.num_dims) * self.var_scaler
        self.prior_type = "gauss_diag"

    def sampler(self, batch_size):
        cov = np.eye(self.num_dims) * np.array(self.var_vec)
        z = np.random.multivariate_normal(self.gauss_mean, cov, batch_size).astype("float32")
        return z

    def get_params_dict(self):
        return {"num_dims": self.num_dims, "std_scaler": self.std_scaler}

    def get_id_str(self):
        return "_%s_%s" % (self.prior_type, self.std_scaler)

    @staticmethod
    def recover(params_dict):
        for p in ["num_dims", "std_scaler"]:
            assert p in params_dict, "Required parameters for recovery: [num_dims, std_scaler]"

        return DiagonalGaussian(**params_dict)
    
    def get_tfp_distribution(self, dtype=np.float32):
        std_vec = np.ones(self.num_dims, dtype=dtype) * self.std_scaler
        
        cur_gauss_mean = np.array(self.gauss_mean, dtype=dtype)
        cur_std_vec    = np.array(std_vec,         dtype=dtype)
        
        return tfp.distributions.MultivariateNormalDiag(loc=cur_gauss_mean, scale_diag=cur_std_vec)


def gaussian_mixture(x_stddev, y_stddev, n_classes, batch_size, labels=None):
    if labels is None:
        labels = np.random.randint(0, n_classes, size=[batch_size])

    shift = 3 * x_stddev

    x = np.random.normal(shift, x_stddev, batch_size).astype("float32")
    y = np.random.normal(0, y_stddev, batch_size).astype("float32")
    z = np.array([[xx, yy] for xx, yy in zip(x, y)])

    def rotate(z, label):
        angle = label * 2.0 * np.pi / n_classes
        rotation_matrix = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        z[np.where(labels == label)] = np.array(
            [rotation_matrix.dot(np.array(point)) for point in z[np.where(labels == label)]]
        )
        return z

    for label in set(labels):
        rotate(z, label)

    return z

