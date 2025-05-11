import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers, losses, Model
import tensorflow_probability as tfp
import joblib
from .utils import distributions
import sys

tfpd = tfp.distributions
tfb  = tfp.bijectors


def init_gauss_diag_prototypes(n_prototypes, dim_z, proto_prior):
    # Initialize the prototype means, based on the reference "prior" distribution
    init_proto_means = np.random.normal(loc=proto_prior.parameters["loc"],
                                        scale=proto_prior.parameters["scale_diag"],
                                        size=(n_prototypes, dim_z)) + 1e-8
    proto_distrib_means = tf.Variable(init_proto_means,
                                      trainable=True,
                                      name="proto_means",
                                      dtype=tf.float32)

    # Initialize the prototype standard deviation, based on the reference "prior" distribution
    init_proto_stds = np.random.normal(loc=proto_prior.parameters["scale_diag"] / 5.0,
                                       scale=proto_prior.parameters["scale_diag"] / 10.0,
                                       size=(n_prototypes, dim_z)) + 1e-8
    proto_distrib_stds = tf.Variable(init_proto_stds,
                                     trainable=True,
                                     name="proto_stds_init",
                                     dtype=tf.float32)

    # Ensure that the standard deviations are positive
    proto_distrib_stds_t = tfp.util.TransformedVariable(
        tfp.bijectors.Softplus()(proto_distrib_stds),
        bijector=tfp.bijectors.Softplus(), name="proto_stds"
    )

    prototype_distrib = tfpd.MultivariateNormalDiag(
        loc=proto_distrib_means,
        scale_diag=proto_distrib_stds_t
    )

    return prototype_distrib


def init_gauss_full_prototypes(n_prototypes, dim_z, proto_prior):
    init_proto_means = np.random.normal(loc=proto_prior.parameters["loc"],
                                        scale=proto_prior.parameters["scale_diag"],
                                        size=(n_prototypes, dim_z)) + 1e-8
    proto_distrib_means = tf.Variable(init_proto_means,
                                      trainable=True,
                                      name="proto_means",
                                      dtype=tf.float32)

    lower_mat_size = int(dim_z * (dim_z + 1) / 2)
    init_proto_stds = np.random.normal(loc=0.2,
                                       scale=0.1,
                                       size=(n_prototypes, lower_mat_size)) + 1e-8
    proto_distrib_stds = tf.Variable(init_proto_stds,
                                     trainable=True,
                                     name="proto_stds_init",
                                     dtype=tf.float32)

    scale_tril_init = tfp.bijectors.FillScaleTriL()(proto_distrib_stds)

    proto_distrib_stds_t = tfp.util.TransformedVariable(
        scale_tril_init, bijector=tfp.bijectors.FillScaleTriL(), name="proto_stds"
    )
    prototype_distrib = tfpd.MultivariateNormalTriL(
        loc=proto_distrib_means,
        scale_tril=proto_distrib_stds_t
    )

    return prototype_distrib


def init_gauss_mixture_prototypes(n_prototypes, dim_z, num_comps, proto_prior):
    # Rename the variables for clarity
    batch_dim = n_prototypes
    dist_dim = dim_z

    mix = np.random.normal(loc=1.0, scale=0.2, size=(batch_dim, num_comps)).astype(np.float32)
    mix_logits = tf.Variable(mix, name="proto_mix")

    mix_init_means = np.random.normal(loc=0.0, scale=1.0,
                                      size=(batch_dim, num_comps, dist_dim)).astype(np.float32)
    mix_means = tf.Variable(mix_init_means, trainable=True,
                            name="mix_means", dtype=tf.float32)

    lower_mat_size = int(dist_dim * (dist_dim + 1) / 2)
    mix_init_stds = np.random.normal(
        loc=0.3, scale=0.3,
        size=(batch_dim, num_comps, lower_mat_size)
    ).astype(np.float32)
    mix_dist_stds = tf.Variable(mix_init_stds,
                                trainable=True,
                                name="stds_init",
                                dtype=tf.float32)

    mix_scale_tril_init = tfb.FillScaleTriL()(mix_dist_stds)

    mix_dist = tfp.distributions.Categorical(logits=mix_logits)
    comp_dist = tfp.distributions.MultivariateNormalTriL(
        loc=mix_means,
        scale_tril=tfp.util.TransformedVariable(mix_scale_tril_init,
                                                bijector=tfb.FillScaleTriL())
    )
    prototype_distrib = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mix_dist,  # probs=mix_probs
        components_distribution=comp_dist
    )

    return prototype_distrib


class ProbPSENN():
    def __init__(self, encoder, decoder, n_prototypes, dim_z, n_classes, proto_prior, distrib_type,
                 prototype_distrib=None, W_classifier=None):
        # Key attributes
        self.encoder = encoder
        self.decoder = decoder
        self.n_prototypes = n_prototypes
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.proto_prior = proto_prior
        self.distrib_type = distrib_type

        # Initialize the distribution over the prototypes
        #################################################
        if not (prototype_distrib is None):
            self.prototype_distrib = prototype_distrib
        else:
            if distrib_type == "gauss_diag":
                self.prototype_distrib = init_gauss_diag_prototypes(n_prototypes, dim_z, proto_prior)
            elif distrib_type == "gauss_full":
                self.prototype_distrib = init_gauss_full_prototypes(n_prototypes, dim_z, proto_prior)
            elif distrib_type == "mix_gf":
                # if "mixture_num_comps" in kwargs: num_comps = kwargs["mixture_num_comps"]
                # else:                             num_comps = 2
                num_comps = 2
                self.prototype_distrib = init_gauss_mixture_prototypes(n_prototypes, dim_z, num_comps,
                                                                       proto_prior)
        # "Batch" the distribution
        self.prototype_distrib_batched = tfpd.Independent(distribution=self.prototype_distrib,
                                                          reinterpreted_batch_ndims=1,
                                                          name="proto_distrib_batched")

        # Figure out if KL(surrogate || prior) is defined
        self.kl_available = False
        try:
            _ = self.prototype_distrib_batched.distribution.kl_divergence(self.proto_prior)
            self.kl_available = True
        except:
            pass

        self.sel_out_distr = "cat" #Categorical
        # sel_out_distr = "oh"     #One-hot

        # Initialize the weight matrix W
        #################################################
        if not (W_classifier is None):
            self.W_classifier = W_classifier
        else:
            w_noise = np.array(np.random.uniform(low=-1e-8, high=1e-8, size=(n_prototypes, n_classes)),
                               dtype=np.float32)
            self.W_classifier = tf.Variable(-np.eye(n_prototypes, n_classes, dtype=np.float32) * 0.5 + w_noise,
                                            trainable=True,
                                            name='last_layer_w')

    def get_model_config(self):
        config_dict = {"n_prototypes": self.n_prototypes,
                       "dim_z": self.dim_z,
                       "n_classes": self.n_classes,
                       "distrib_type": self.distrib_type,
                       "W_classifier": self.W_classifier,
                       "class_name": self.__class__.__name__,
                       }
        return config_dict

    def save_model(self, path, params_str=""):
        #save the encoder / decoder modules
        self.encoder.save("%s/encoder%s" % (path, params_str))     #Legacy
        self.decoder.save("%s/decoder%s" % (path, params_str))     #Legacy

        #Save the tfp distributions
        distributions.save_tfp_distribution(self.proto_prior, "%s/proto_prior%s" % (path, params_str))
        distributions.save_tfp_distribution(self.prototype_distrib, "%s/prototype_distrib%s" % (path, params_str))
        #Save the remaining parameters in a configuration dictionary
        config_dict = self.get_model_config()
        with open("%s/model_config%s" % (path, params_str), 'wb+') as file:
            joblib.dump(config_dict, file)

    #@staticmethod
    @classmethod  #Makes it possible to generalize the loading function to inheritance
    def load_model(cls, path, params_str=""):
        #Load the encoder / decoder modules
        encoder = tf.keras.models.load_model("%s/encoder%s" % (path, params_str)) #Legacy
        decoder = tf.keras.models.load_model("%s/decoder%s" % (path, params_str)) #Legacy
        
        #Load the tfp distributions
        proto_prior = distributions.load_tfp_distribution("%s/proto_prior%s" % (path, params_str))
        prototype_distrib = distributions.load_tfp_distribution("%s/prototype_distrib%s" % (path, params_str))
        
        #Load the remaining configuration parameters (dictionary)
        config_dict = joblib.load("%s/model_config%s" % (path, params_str))
        #Retrieve the class 
        if "class_name" in config_dict:
            class_name = config_dict.pop("class_name")
            # NOTE: maybe remove this sanity check for flexibility in the future...
            assert cls.__name__ == class_name, f"Trying to load a {cls.__name__} but object was saved as {class_name}"
            print(f"Loading object from class: {class_name}")

        #Complete the dictionary so that the entire ProbPSENN can be recovered
        config_dict["encoder"] = encoder
        config_dict["decoder"] = decoder
        config_dict["proto_prior"] = proto_prior
        config_dict["prototype_distrib"] = prototype_distrib
        return cls(**config_dict)
    

    def estimate_max_memory_fitting(self, x_data, min_ratio=10.0):
        """Auxiliary function to estimate how much data can be used to fit the prototypes without
        exceeding the available memory.

        Args:
            x_data (tensor): Data tensor
            min_ratio (float): Minimum ratio of samples (if estimation is lower, raises an error)

        Returns:
            int: Estimated maximum number of samples that could be used.
        """
        _max_samples = len(x_data)
        _fitted = False
        _ratio  = 100.0
        while _ratio > min_ratio and not _fitted:
            try:
                _ = self.encode_data(x_data[:_max_samples])
                _fitted = True #Sampling can be done with _max_samples
            except Exception:
                #Reduce the number of samples by a factor of 0.9
                _max_samples = int(_max_samples * 0.9)
                _ratio = _max_samples / len(x_data) * 100
                print(f"Memory error! Reducing _max_samples to {_max_samples} (ratio: {_ratio}%)")

        if not _fitted: sys.exit("Could not fit prototypes with enough data!")

        ratio_used = _max_samples / len(x_data) * 100
        print(f"Fitting prototypes with {_max_samples} samples ({ratio_used}% of total data)")
        return _max_samples

    ### SET THE PROTOTYPES ###
    ##########################
    # Set the prototype mean and variance for FULL GAUSSIANS
    def fit_gauss_full_prototypes(self, x_data_, y_data_):

        y_data_int = tf.argmax(y_data_, axis=1)

        init_proto_means = np.zeros((self.n_prototypes, self.dim_z), dtype="float32")

        lower_mat_size = int(self.dim_z * (self.dim_z + 1) / 2)
        init_proto_stds = np.zeros((self.n_prototypes, lower_mat_size), dtype="float32")

        for i in range(self.n_classes):
            cur_data = self.encode_data(x_data_[y_data_int == i]).numpy() 
            
            # retrieve mean and covariance matrix
            init_proto_means[i, :] = np.mean(cur_data, axis=0)  # estimate mean
            cur_cov_estim = np.cov(cur_data, rowvar=0, dtype="float32")  # estimate cov. matrix
            cur_cov_estim = np.linalg.cholesky(cur_cov_estim)  # cholesky decomposition
            del cur_data
            # I want cur_cov_estim to be the value of the covariance matrix. 
            # Thus, i want the lower-triangular coefficients that produce such matrix
            # I think that this can be omitted, though!
            init_proto_stds[i, :] = tfp.bijectors.FillScaleTriL().inverse(cur_cov_estim).numpy()


        proto_distrib_means = tf.Variable(init_proto_means,
                                          trainable=True,
                                          dtype=tf.float32)

        proto_distrib_stds = tf.Variable(init_proto_stds,
                                         trainable=True,
                                         dtype=tf.float32)

        scale_tril_init = tfp.bijectors.FillScaleTriL()(proto_distrib_stds) #I'd say this can be omitted too

        #NEW REINITIALIZATION
        self.prototype_distrib.parameters["loc"].assign(proto_distrib_means)
        self.prototype_distrib.parameters["scale_tril"].assign(scale_tril_init)
        
        
    # Set the prototype mean and variance for DIAGONAL GAUSSIANS
    def fit_gauss_diag_prototypes(self, x_data_, y_data_):

        y_data_int = tf.argmax(y_data_, axis=1)

        init_proto_means = np.zeros((self.n_prototypes, self.dim_z), dtype="float32")
        init_proto_stds  = np.zeros((self.n_prototypes, self.dim_z), dtype="float32")

        for i in range(self.n_classes):
            cur_data = np.copy( self.encode_data(x_data_[y_data_int == i]).numpy() ) 
            # retrieve mean and covariance matrix
            init_proto_means[i, :] = np.mean(cur_data, axis=0)  # estimate mean

            # I can take the values directly because I will simply override (transformed) parameters
            init_proto_stds[i, :]  = np.var(cur_data, axis=0) # estimate stds

        #Transform numpy arrays to tf variables
        proto_distrib_means = tf.Variable(init_proto_means, trainable=True, dtype=tf.float32)
        proto_distrib_stds  = tf.Variable(init_proto_stds,  trainable=True, dtype=tf.float32)

        #NEW REINITIALIZATION
        self.prototype_distrib.parameters["loc"].assign(proto_distrib_means)
        self.prototype_distrib.parameters["scale_diag"].assign(init_proto_stds)
    
    

    #### BUILDING THE INFERENCE PIPELINE ###
    ########################################
    def get_prototype_distrib(self):
        return self.prototype_distrib_batched

    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),))
    def sample_prototypes(self, n_samples):
        proto_sample = self.prototype_distrib_batched.sample(n_samples)
        # debug
        #self.proto_sample = proto_sample
        return proto_sample

    def mode_prototypes(self):
        return self.prototype_distrib_batched.mode()

    def encode_data(self, x_data_):
        encoded_data = self.encoder(x_data_)
        return encoded_data

    #@tf.function
    def compute_distance_vector(self, encoded_data, n_samples):
        proto_sample = self.sample_prototypes(n_samples)

        XX = tf.reshape(tf.reduce_sum(tf.pow(encoded_data, 2), axis=1), shape=(-1, 1, 1))
        YY = tf.transpose(tf.expand_dims(tf.reduce_sum(tf.pow(proto_sample, 2), axis=2), 2))

        prototype_distances_ = XX + YY - 2.0 * tf.tensordot(encoded_data, tf.transpose(proto_sample), axes=((1), (0)))
        prototype_distances = tf.identity(prototype_distances_, name="prototype_distances")
        return prototype_distances

    #@tf.function
    def compute_distance_vector_from_samples(self, encoded_data, proto_sample):
        XX = tf.reshape(tf.reduce_sum(tf.pow(encoded_data, 2), axis=1), shape=(-1, 1, 1))
        YY = tf.transpose(tf.expand_dims(tf.reduce_sum(tf.pow(proto_sample, 2), axis=2), 2))

        prototype_distances_ = XX + YY - 2.0 * tf.tensordot(encoded_data, tf.transpose(proto_sample), axes=((1), (0)))
        prototype_distances = tf.identity(prototype_distances_, name="prototype_distances")
        return prototype_distances

    #@tf.function
    def classify_distances(self, prototype_distances):
        logits = tf.tensordot(prototype_distances, self.W_classifier, axes=((1), (0)))
        logits = tf.identity(logits, name="logits")
        class_probs = tf.keras.activations.softmax(logits, axis=2)
        return logits, class_probs

    def prepare_output_distrib(self, logits_, class_probs_, reduce_samples=False):
        if not reduce_samples:
            logits = tf.identity(logits_)
            pred_distrib = tfpd.Categorical(logits=logits)  # using probs could be more numerically unstable (in train)
        else:
            class_probs = tf.reduce_mean(class_probs_, axis=1)
            pred_distrib = tfpd.Categorical(probs=class_probs)  # for prediction should be ok

        return pred_distrib

    # Reduce samples should only be used at test time...
    @tf.function(reduce_retracing=True)
    def get_pred_distrib(self, x_data_, n_samples, reduce_samples=False):
        encoded_data = self.encoder(x_data_)
        prototype_distances = self.compute_distance_vector(encoded_data, n_samples)
        logits_, class_probs_ = self.classify_distances(prototype_distances)
        pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples)
        return pred_distrib
    

    #Predict using the modes of the distributions
    def get_pred_distrib_from_modes(self, x_data_):
        proto_modes = tf.expand_dims(self.mode_prototypes(),axis=0)

        #encoded_data = self.encode_data(x_data_)
        encoded_data = self.encoder(x_data_)
        prototype_distances = self.compute_distance_vector_from_samples(encoded_data, proto_modes)
        logits_, class_probs_ = self.classify_distances(prototype_distances)
        pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples=True)
        return pred_distrib.mode()#To get only the label


    # The predictive distribution should not be batched (i.e., it should be averaged if more than one sample
    # inference has been used to compute it...). In other words, its batch size should be given only by the
    # number of inputs classified.
    def compute_acc(self, pred_distrib, y_data_):
        assert len(pred_distrib.batch_shape) == 1, "The distribution should not be batched by inferences"
        cur_acc = np.sum(pred_distrib.mode() == np.argmax(y_data_, axis=1)) / y_data_.shape[0] * 100
        return cur_acc

    #@tf.function
    def compute_nll(self, pred_distrib, y_data_, n_samples):
        y_data_rebatch = tf.repeat(tf.expand_dims(tf.argmax(y_data_, axis=1), axis=1), n_samples, axis=1)
        log_prob_full = pred_distrib.log_prob(y_data_rebatch)
        log_prob = -tf.reduce_mean(log_prob_full)

        # debug
        self.log_prob = log_prob
        self.log_prob_full = log_prob_full

        return log_prob


    ####     TRAINING PROCEDURES     ###
    #####################################
    def enable_W_training(self):
        self.W_classifier.trainable = True

    def disable_W_training(self):
        self.W_classifier.trainable = False

    def enable_ae_training(self):
        self.encoder.trainable = True
        self.decoder.trainable = True

    def disable_ae_training(self):
        self.encoder.trainable = False
        self.decoder.trainable = False

    @staticmethod
    def reconstruction_loss(x_input, x_reconstruction):
        ae_loss = tf.reduce_mean(tf.math.squared_difference(x_input, x_reconstruction))
        return ae_loss

    #Loss weights just added for code compatibility with vae-like extensions
    def train_step_reconstruction(self, x_data, optimizer, loss_weights={"rec": 1.0}):
        assert self.encoder.trainable, "The encoder is in non-trainable mode"
        assert self.decoder.trainable, "The decoder is in non-trainable mode"

        ##########################################
        # AutoEncoder - train reconstruction error
        ##########################################
        with tf.GradientTape() as ae_tape:
            x_data_enc = self.encoder(x_data, training=True)
            x_reconstruction = self.decoder(x_data_enc, training=True)  # encode and decode
            ae_loss = loss_weights["rec"] * self.reconstruction_loss(x_data, x_reconstruction)  # reconstruction loss

        ae_trainable_variables = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        ae_gradients = ae_tape.gradient(ae_loss, ae_trainable_variables)
        optimizer.apply_gradients(zip(ae_gradients, ae_trainable_variables))

        #Return total loss and dict with each loss term
        #---Output format is completely redundant right now, but makes it possible to generalize (e.g., to VAEs)!
        loss_dict = {"REC": ae_loss}
        return ae_loss, loss_dict


    @tf.function
    def train_step_determ_as_prob(self, x_data_, y_data_, optimizer, n_samples,
                                  train_comps=["enc", "dec", "proto", "w"],
                                  loss_weights={"nll": 1.0, "rec": 10.0, "r1": 0.0, "r2": 0.05}):
        assert self.encoder.trainable, "The encoder is in non-trainable mode"
        assert self.decoder.trainable, "The decoder is in non-trainable mode"

        with tf.GradientTape() as tape:
            # Inference pipeline
            encoded_data = self.encoder(x_data_, training=True)
            prototype_distances = self.compute_distance_vector(encoded_data, n_samples)
            logits_, class_probs_ = self.classify_distances(prototype_distances)
            pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples=False)

            # NLL
            log_prob_ = self.compute_nll(pred_distrib, y_data_, n_samples)
            log_prob = loss_weights["nll"] * log_prob_

            # Reconstruction loss
            x_reconstruction = self.decoder(encoded_data, training=True)  # decode
            rec_loss_ = self.reconstruction_loss(x_data_, x_reconstruction)  # reconstruction loss
            rec_loss = loss_weights["rec"] * rec_loss_

            # R1 and R2
            data_logprob = - self.prototype_distrib_batched.distribution.log_prob(
                tf.expand_dims(encoded_data, 1))  # (n_data, n_protos)  

            opt_probs_R1 = tf.reduce_min(data_logprob, axis=0)  # (n_protos)
            error_R1_ = tf.reduce_mean(opt_probs_R1, name='error_R1')  # (1)
            error_R1 = loss_weights["r1"] * error_R1_

            opt_probs_R2 = tf.reduce_sum(tf.math.multiply(y_data_, data_logprob), axis=1)  

            error_R2_ = tf.reduce_mean(tf.clip_by_value(opt_probs_R2, -1, 10000),
                                       name='error_R2') 
            error_R2 = loss_weights["r2"] * error_R2_  

            # Combining the losses
            final_loss = (log_prob + rec_loss + error_R1 + error_R2)

        train_vars = ()
        if "proto" in train_comps:
            train_vars += self.prototype_distrib.trainable_variables
        if "enc" in train_comps:
            train_vars += tuple(self.encoder.trainable_variables)
        if "dec" in train_comps:
            train_vars += tuple(self.decoder.trainable_variables)
        if "w" in train_comps:
            train_vars += (self.W_classifier,)

        gradients = tape.gradient(final_loss, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

        loss_dict = {"NLL": log_prob, "REC": rec_loss, "R1": error_R1, "INT": error_R2}

        return final_loss, loss_dict



class ProbPSENN_VAE(ProbPSENN):
    def __init__(self, encoder, decoder, n_prototypes, dim_z, n_classes, proto_prior, distrib_type,
                 prototype_distrib=None, W_classifier=None):
        #Initialize as in Parent Class
        super().__init__(encoder, decoder, n_prototypes, dim_z, n_classes, proto_prior, distrib_type,
                 prototype_distrib, W_classifier)
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    #Override encoding data to support handling log-vars
    #@tf.function
    def encode_data(self, x_data_):
        encoded_data = self.encoder(x_data_)
        mean, logvar = tf.split(encoded_data, num_or_size_splits=2, axis=1)
        return mean
    
    #@tf.function
    def encode_data_distribution(self, x_data_):
        encoded_data = self.encoder(x_data_)
        mean, logvar = tf.split(encoded_data, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    # Reduce samples should only be used at test time...
    @tf.function(reduce_retracing=True)
    def get_pred_distrib(self, x_data_, n_samples, reduce_samples=False):
        encoded_data = self.encoder(x_data_)
        
        mean, logvar = tf.split(encoded_data, num_or_size_splits=2, axis=1)
        sampled_z = mean
        prototype_distances = self.compute_distance_vector(sampled_z, n_samples)

        logits_, class_probs_ = self.classify_distances(prototype_distances)
        pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples)
        return pred_distrib
    

    #Predict using the modes of the distributions
    def get_pred_distrib_from_modes(self, x_data_):
        proto_modes = tf.expand_dims(self.mode_prototypes(),axis=0)

        #encoded_data = self.encode_data(x_data_)
        encoded_data = self.encoder(x_data_)
        mean, logvar = tf.split(encoded_data, num_or_size_splits=2, axis=1)

        prototype_distances = self.compute_distance_vector_from_samples(mean, proto_modes)
        logits_, class_probs_ = self.classify_distances(prototype_distances)
        pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples=True)
        return pred_distrib
        

    def train_step_reconstruction(self, x_data, optimizer, loss_weights={"rec": 1.0, "kl": 0.05}):
        assert self.encoder.trainable, "The encoder is in non-trainable mode"
        assert self.decoder.trainable, "The decoder is in non-trainable mode"

        ##########################################
        # AutoEncoder - train reconstruction error
        ##########################################
        with tf.GradientTape() as ae_tape:
            x_data_enc = self.encoder(x_data, training=True)
            mean, logvar = tf.split(x_data_enc, num_or_size_splits=2, axis=1)
            
            sampled_z = self.reparameterize(mean, logvar)  #Reparameterization trick 
    
            x_reconstruction = self.decoder(sampled_z, training=True)  # encode and decode
            rec_loss_ = self.reconstruction_loss(x_data, x_reconstruction)  # reconstruction loss
            rec_loss  = loss_weights["rec"] * rec_loss_
            kl_loss_ = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1))
            kl_loss  = loss_weights["kl"] * kl_loss_
            ae_loss  = rec_loss + kl_loss

        ae_trainable_variables = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        ae_gradients = ae_tape.gradient(ae_loss, ae_trainable_variables)
        optimizer.apply_gradients(zip(ae_gradients, ae_trainable_variables))

        loss_dict = {"REC": rec_loss, "KL": kl_loss}
        return ae_loss, loss_dict
    


    @tf.function
    def train_step_determ_as_prob(self, x_data_, y_data_, optimizer, n_samples,
                                train_comps=["enc", "dec", "proto", "w"],
                                loss_weights={"nll": 1.0, "rec": 10.0, "r1": 0.0, "r2": 0.05, "kl": 0.05}):
        assert self.encoder.trainable, "The encoder is in non-trainable mode"
        assert self.decoder.trainable, "The decoder is in non-trainable mode"

        with tf.GradientTape() as tape:
            # Inference pipeline
            encoder_distrib = self.encoder(x_data_, training=True)
            mean, logvar = tf.split(encoder_distrib, num_or_size_splits=2, axis=1)

            encoded_data = self.reparameterize(mean, logvar)  #Reparameterization trick

            prototype_distances = self.compute_distance_vector(encoded_data, n_samples)
            logits_, class_probs_ = self.classify_distances(prototype_distances)
            pred_distrib = self.prepare_output_distrib(logits_, class_probs_, reduce_samples=False)

            # NLL
            log_prob_ = self.compute_nll(pred_distrib, y_data_, n_samples)
            log_prob = loss_weights["nll"] * log_prob_

            # Reconstruction loss
            x_reconstruction = self.decoder(encoded_data, training=True)  # decode
            rec_loss_ = self.reconstruction_loss(x_data_, x_reconstruction)  # reconstruction loss
            rec_loss = loss_weights["rec"] * rec_loss_

            # R1 and R2
            data_logprob = - self.prototype_distrib_batched.distribution.log_prob(
                tf.expand_dims(encoded_data, 1))  # (n_data, n_protos)  

            opt_probs_R1 = tf.reduce_min(data_logprob, axis=0)  # (n_protos)
            error_R1_ = tf.reduce_mean(opt_probs_R1, name='error_R1')  # (1)
            error_R1 = loss_weights["r1"] * error_R1_

            opt_probs_R2 = tf.reduce_sum(tf.math.multiply(y_data_, data_logprob), axis=1)  

            error_R2_ = tf.reduce_mean(tf.clip_by_value(opt_probs_R2, -1, 10000),
                                    name='error_R2')
            error_R2 = loss_weights["r2"] * error_R2_  

            #KULLBACK LEIBLER LOSS for VAE
            kl_loss = loss_weights["kl"] * tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1))

            # Combining the losses
            final_loss = (log_prob + rec_loss + error_R1 + error_R2 + kl_loss)


        train_vars = ()
        if "proto" in train_comps:
            train_vars += self.prototype_distrib.trainable_variables
        if "enc" in train_comps:
            train_vars += tuple(self.encoder.trainable_variables)
        if "dec" in train_comps:
            train_vars += tuple(self.decoder.trainable_variables)
        if "w" in train_comps:
            train_vars += (self.W_classifier,)
        gradients = tape.gradient(final_loss, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

        loss_dict = {"NLL": log_prob, "REC": rec_loss, "R1": error_R1, "INT": error_R2, "KL": kl_loss}

        return final_loss, loss_dict
