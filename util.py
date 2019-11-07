import numpy as np
import tensorflow as tf
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def unpickle(filename):
  with open(filename, 'rb') as fo:
    return cPickle.load(fo, encoding='latin1')

def reshape_flattened_image_batch(flat_image_batch):
  return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])  # convert from NCHW to NHWC

def combine_batches(batch_list):
  images = np.vstack([reshape_flattened_image_batch(batch['data'])
                      for batch in batch_list])
  labels = np.vstack([np.array(batch['labels']) for batch in batch_list]).reshape(-1, 1)
  return {'images': images, 'labels': labels}


def cast_and_normalise_images(data_dict):
  """Convert images to floating point with the range [0.5, 0.5]"""
  images = data_dict['images']
  data_dict['images'] = (tf.cast(images, tf.float32) / 255.0) - 0.5
  return data_dict

def plot_loss(train_res_recon_error, train_res_perplexity, save_dir):
  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(1,2,1)
  ax.plot(train_res_recon_error)
  ax.set_yscale('log')
  ax.set_title('NMSE.')
  
  ax = f.add_subplot(1,2,2)
  ax.plot(train_res_perplexity)
  ax.set_title('Average codebook usage (perplexity).')
  
  f.savefig(os.path.join(save_dir, 'loss.png'))
  plt.close(f)  # close the figure window


def convert_batch_to_image_grid(img_batch, img_shape):
  reshaped = (img_batch.reshape(4, 8, img_shape[0], img_shape[1], img_shape[2])
              .transpose(0, 2, 1, 3, 4)
              .reshape(4 * img_shape[0], 8 * img_shape[1], img_shape[2]))
  return np.squeeze(reshaped)

def plot_reconstructions(data_recon, reconstructions, img_shape, save_dir):
  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(2,2,1)
  if img_shape[2] == 1:
    ax.imshow(convert_batch_to_image_grid(data_recon[:32], img_shape),
              cmap='gray') 
  else:
    ax.imshow(convert_batch_to_image_grid(data_recon[:32], img_shape),
              interpolation='nearest')
  ax.set_title('normal data originals')
  plt.axis('off')
  
  ax = f.add_subplot(2,2,2)
  if img_shape[2] == 1:
    ax.imshow(convert_batch_to_image_grid(reconstructions[:32], img_shape),
              cmap='gray')    
  else:  
    ax.imshow(convert_batch_to_image_grid(reconstructions[:32], img_shape),
              interpolation='nearest')
  ax.set_title('normal data reconstructions')
  plt.axis('off')
  
  ax = f.add_subplot(2,2,3)
  if img_shape[2] == 1:
    ax.imshow(convert_batch_to_image_grid(data_recon[-32:], img_shape),
              cmap='gray')    
  else:  
    ax.imshow(convert_batch_to_image_grid(data_recon[-32:], img_shape),
              interpolation='nearest')
  ax.set_title('adversarial data originals')
  plt.axis('off')
  
  ax = f.add_subplot(2,2,4)
  if img_shape[2] == 1:
    ax.imshow(convert_batch_to_image_grid(reconstructions[-32:], img_shape),
              cmap='gray')    
  else:  
    ax.imshow(convert_batch_to_image_grid(reconstructions[-32:], img_shape),
              interpolation='nearest')
  ax.set_title('adversarial data reconstructions')
  plt.axis('off')
  
  f.savefig(os.path.join(save_dir, 'reconstruction.png'))
  plt.close(f)    # close the figure window