from __future__ import print_function

import os
import dateutil.tz
import datetime
import argparse
import importlib
import numpy as np
import tensorflow as tf
import sonnet as snt
import util

tf.set_random_seed(0)


class VQVAE(object):
  """docstring for VQ-VAE"""
  def __init__(self, encoder, decoder, xs, data, model, num_embeddings, embedding_dim, 
               commitment_cost, vq_use_ema, learning_rate, batch_size):
    self.encoder = encoder
    self.decoder = decoder
    self.xs = xs
    self.data = data
    self.model = model
    self.batch_size = batch_size
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.commitment_cost = commitment_cost

    image_size =self.xs.shape[0]
    image_channel =self.xs.shape[2]

    self.pre_vq_conv1 = snt.Conv2D(output_channels=self.embedding_dim,
        kernel_shape=(1, 1),
        stride=(1, 1),
        name="to_vq") 
        
    if vq_use_ema:
      self.vq_vae = snt.nets.VectorQuantizerEMA(
          embedding_dim=self.embedding_dim,
          num_embeddings=self.num_embeddings,
          commitment_cost=self.commitment_cost,
          decay=decay)
    else:
      self.vq_vae = snt.nets.VectorQuantizer(
          embedding_dim=self.embedding_dim,
          num_embeddings=self.num_embeddings,
          commitment_cost=self.commitment_cost)


    # Process inputs with conv stack, finishing with 1x1 to get to correct size.
    self.x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, image_channel))
    self.z = self.pre_vq_conv1(self.encoder(self.x))

    # vq_output_train["quantize"] are the quantized outputs of the encoder.
    # That is also what is used during training with the straight-through estimator. 
    # To get the one-hot coded assignments use vq_output_train["encodings"] instead.
    # These encodings will not pass gradients into to encoder, 
    # but can be used to train a PixelCNN on top afterwards.

    # For training
    vq_output_train = self.vq_vae(self.z, is_training=True)
    x_recon = self.decoder(vq_output_train["quantize"])
    self.recon_error = tf.reduce_mean((x_recon - self.x)**2) / self.xs.data_variance  # Normalized MSE
    self.loss = self.recon_error + vq_output_train["loss"]
    
    # For evaluation, make sure is_training=False!
    self.vq_output_eval = self.vq_vae(self.z, is_training=False)
    self.x_recon_eval = self.decoder(self.vq_output_eval["quantize"])
    
    # The following is a useful value to track during training.
    # It indicates how many codes are 'active' on average.
    self.perplexity = vq_output_train["perplexity"] 
    
    # Create optimizer and TF session.
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)

    self.saver = tf.train.Saver()
     
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.train.SingularMonitoredSession(config=config)


    self.im_save_dir = 'logs/{}/{}_k{}_d{}_cost{}'.format(self.data, self.model, self.num_embeddings,
                                                             self.embedding_dim, self.commitment_cost)
    if not os.path.exists(self.im_save_dir):
      os.makedirs(self.im_save_dir)    

  def train(self, num_training_updates):

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(0, num_training_updates):
      feed_dict = {self.x: self.xs.train(batch_size)}
      results = self.sess.run([self.train_op, self.recon_error, self.perplexity],
                         feed_dict=feed_dict)
      train_res_recon_error.append(results[1])
      train_res_perplexity.append(results[2])

      if (i+1) % 100 == 0:
        print('{} iterations, recon_error: {}, perplexity: {}'
          .format((i+1), np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))

      if i == num_training_updates-1:
        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
          f.write('{} : {} iterations, K = {}, D = {}, commitment_cost = {}, recon_error = {}, perplexity = {}\n'
                  .format(timestamp, num_training_updates, self.num_embeddings, self.embedding_dim, 
                    self.commitment_cost, np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))
          f.flush()

    self.save(timestamp)
    util.plot_loss(train_res_recon_error, train_res_perplexity, self.im_save_dir)


  def get_session(self, sess):
      session = sess
      while type(session).__name__ != 'Session':
          #pylint: disable=W0212
          session = session._sess
      return session

  def save(self, timestamp):

    checkpoint_dir = 'checkpoint_dir/{}/{}_{}_k{}_d{}_cost{}'.format(self.data, 
                                                                    timestamp, 
                                                                    self.model, 
                                                                    self.num_embeddings,
                                                                    self.embedding_dim, 
                                                                    self.commitment_cost)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.get_session(self.sess), os.path.join(checkpoint_dir, 'model.ckpt'))


  def load(self, pre_trained = False, timestamp = ''):

    if pre_trained == True:
      print('Loading Pre-trained Model...')
      checkpoint_dir = 'pre_trained_models/{}/{}_{}_k{}_d{}_cost{}'.format(self.data, 
                                                                          timestamp, 
                                                                          self.model, 
                                                                          self.num_embeddings,
                                                                          self.embedding_dim, 
                                                                          self.commitment_cost)
    else:
      if timestamp == '':
        print('Best Timestamp not provided. Abort !')
        checkpoint_dir = ''
      else:
        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_k{}_d{}_cost{}'.format(self.data, 
                                                                        timestamp, 
                                                                        self.model, 
                                                                        self.num_embeddings,
                                                                        self.embedding_dim, 
                                                                        self.commitment_cost)

    self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
    print('Restored model weights.')    


  def recon_enc(self, timestamp, val = True):

    if val:
      data_recon, label_recon = self.xs.validation()
    else:
      data_recon, label_recon = self.xs.test()
      #data_recon, label_recon = self.xs.load_all()

    num_pts_to_plot = data_recon.shape[0]
    recon_batch_size = self.batch_size
    reconstructions = np.zeros_like(data_recon) 
    latent = np.zeros(shape=(num_pts_to_plot, 8, 8, self.embedding_dim))

    print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))
    for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
        if (b+1)*recon_batch_size > num_pts_to_plot:
           pt_indx = np.arange(b*recon_batch_size, num_pts_to_plot)
        else:
           pt_indx = np.arange(b*recon_batch_size, (b+1)*recon_batch_size)
        xtrue = data_recon[pt_indx, :]

        x_recon, vq_output = self.sess.run([self.x_recon_eval, self.vq_output_eval], feed_dict={self.x : xtrue})

        reconstructions[pt_indx, :] =  x_recon

        latent[pt_indx, :] = vq_output["quantize"] 
    
    print(reconstructions.shape)
    print(latent.shape)

    np.save('logs/%s_latent.npy'%(self.data,), latent)

    util.plot_reconstructions(data_recon, reconstructions, self.im_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--model', type=str, default='vq_vae')
    parser.add_argument('--K', type=int, default=512)
    parser.add_argument('--dz', type=int, default=64)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--cost', type=float, default=0.25)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')

    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)

    tf.reset_default_graph()

    # Set hyper-parameters.
    batch_size = args.bs
    # This value is not that important, usually 64 works.
    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = args.K    
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = args.dz
    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = args.cost
    timestamp = args.timestamp

    # 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
    num_training_updates = args.steps

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    # These hyper-parameters define the size of the model (number of parameters and layers).
    # The hyper-parameters in the paper were (For ImageNet):
    # batch_size = 128
    # image_size = 128
    # num_hiddens = 128
    # num_residual_hiddens = 32
    # num_residual_layers = 2

    # Use EMA updates for the codebook (instead of the Adam optimizer).
    # This typically converges faster, and makes the model less dependent on choice
    # of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
    # developed afterwards). See Appendix of the paper for more details.
    vq_use_ema = False

    # This is only used for EMA updates.
    decay = 0.99

    learning_rate = 3e-4

    xs = data.DataSampler()

    # Build modules
    encoder = model.Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder = model.Decoder(num_hiddens, num_residual_layers, num_residual_hiddens) 

    vq_vae = VQVAE(encoder, decoder, xs, args.data, args.model, num_embeddings, embedding_dim, 
                  commitment_cost, vq_use_ema, learning_rate, batch_size)

    if args.train == 'True':
      vq_vae.train(num_training_updates)
    else:
      print('Attempting to Restore Model ...')
      if timestamp == '':
        vq_vae.load(pre_trained=True)
        timestamp = 'pre-trained'
      else:
        vq_vae.load(pre_trained=False, timestamp = timestamp) 

      vq_vae.recon_enc(timestamp, val=False)     
