import glob
import os
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import wavio

# Ignore frequency components below this value (in Hz)
MIN_RELEVANT_FREQUENCY = 0
# Ignore frequency components above this value (in Hz)
MAX_RELEVANT_FREQUENCY = 12500

def get_spectrograms(dir_path, prefix, num_freq_bins, num_time_samples, num_time_bins):
  '''Computes and processes spectrograms of all matching .wav files in the target directory

  Args:
    dir_path (string) : The path of the directory of .wav files
    prefix (string) : A filename prefix of all the desired .wav files in the directory
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrograms
    num_time_samples (int) : The desired length (in time samples) to either clip or right-pad each .wav file recording
    num_time_bins (int) : The number of desired time bins in the processed binned spectrograms

  Returns:
    The indices (from alphabetical order by file name) of validly computed spectrograms
    The indices (from alphabetical order by file name) of files from which spectrograms were not successfully computed
    The validly computed spectrograms (in alphabetical order by file name)
  '''
  error_indices = []
  valid_indices = []
  spectrograms = []

  audio_files = glob.glob(os.path.join(dir_path, '%s*.wav'%prefix))
  for i, audio_file_name in enumerate(sorted(audio_files)):
    try:
      wav_obj = wavio.read(audio_file_name)
    except:
      error_indices.append(i)
      continue
    sample_rate = wav_obj.rate
    # This is a mono stream of audio, so we get rid of a dimension by taking the first component.
    # If this were stereo, the [0,:] below would just take the first channel.
    audio_data = np.transpose(wav_obj.data)[0,:]
    spectrogram = process_audio(audioData, sample_rate, num_freq_bins, num_time_samples, num_time_bins)
    valid_indices.append(i)
    spectrograms.append(spectrogram)

  return (valid_indices, error_indices, spectrograms)

def process_audio(audio_data, sample_rate, num_freq_bins, num_time_samples, num_time_bins, check_valid=True):
  '''Computes and processes a binned spectrogram from a raw audio (unclipped and unpadded) signal array.

  Args:
    audio_data (numpy.array): Array for a raw audio signal (one channel only)
    sample_rate (int) : The number of samples per second of the audio signal.
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram
    num_time_samples (int) : The desired length (in time samples) to either clip or right-pad the audio signal array
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram
    check_valid (boolean) : Whether to interrupt the function on a processing error and debug with plots

  Returns:
    A numpy.array representing the processed and binned spectrogram
  '''
  padded_data = np.zeros(num_time_samples, dtype=audio_data.dtype)
  fin_length = min(audio_data.shape[0], num_time_samples)
  padded_data[:fin_length] = audio_data[:fin_length]

  fully_binned_spectrogram, binned_freq_spectrogram = compute_spectrogram(padded_data, sample_rate, num_freq_bins, num_time_bins)

  # This is for debugging any invalid spectrograms that slip through the cracks.
  if check_valid and np.mean(fully_binned_spectrogram) < 1:
    print(np.mean(fully_binned_spectrogram))
    plt.imshow(binned_freq_spectrogram)
    plt.colorbar()
    plt.show()
    plt.imshow(fully_binned_spectrogram)
    plt.colorbar()
    plt.show()

  return fully_binned_spectrogram

def compute_spectrogram(audio_data, sample_rate, num_freq_bins, num_time_bins):
  '''Computes and processes a spectrogram directly from an audio signal.

  Args:
    audio_data (numpy.array): Array for a raw audio signal (one channel only)
    sample_rate (int) : The number of samples per second of the audio signal.
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram

  Returns:
    A numpy.array representing the fully processed and binned spectrogram
    A numpy.array for the processed spectrogram only binned on the frequency dimension (for debugging purposes)
  '''
  # Sxx has first dim Freq, second dim time
  f, t, Sxx = signal.spectrogram(audio_data, sample_rate, scaling='spectrum', return_onesided=True)
  Sxx = np.array(Sxx)

  # Find the indices of the bounds of the relevant frequencies
  min_relevant_freq_idx = np.searchsorted(f, MIN_RELEVANT_FREQUENCY)
  max_relevant_freq_idx = np.searchsorted(f, MAX_RELEVANT_FREQUENCY)

  trimmed_spectrogram = Sxx[min_relevant_freq_idx:max_relevant_freq_idx,:]
  trimmed_freqs = f[min_relevant_freq_idx:max_relevant_freq_idx]

  binned_freq_spectrogram = bin_spectrogram_freq(trimmed_spectrogram, num_freq_bins)
  fully_binned_spectrogram = bin_spectrogram_time(binned_freq_spectrogram, num_time_bins)

  return fully_binned_spectrogram, binned_freq_spectrogram

def bin_spectrogram_freq(spectrogram, num_freq_bins):
  '''Bins a spectrogram on its frequency dimension.

  Args:
    spectrogram (numpy.array) : The unbinned spectrogram
    num_freq_bins (int) : The number of desired frequency bins in the processed binned spectrogram

  Returns:
    The spectrogram binned on its frequency dimension.
  '''
  return __bin_matrix_dimension(spectrogram, 0, num_freq_bins)

def bin_spectrogram_time(spectrogram, num_time_bins):
  '''Bins a spectrogram on its time dimension.

  Args:
    spectrogram (numpy.array) : The unbinned spectrogram
    num_time_bins (int) : The number of desired time bins in the processed binned spectrogram

  Returns:
    The spectrogram binned on its time dimension.
  '''
  return __bin_matrix_dimension(spectrogram, 1, num_time_bins)

def __bin_matrix_dimension(m, dimension, num_bins):
  '''Bins a matrix on a specified dimension.

  Args:
    m (numpy.array) : The original matrix
    dimension (int) : The dimension to bin
    num_bins (int) : The desired number of bins for the specified dimension

  Returns:
    A numpy.array of the matrix binned on the specified dimension.
  '''
  bin_size = int(np.floor(m.shape[dimension]/(num_bins+0.0)))

  binned_matrix = np.zeros((m.shape[1-dimension], num_bins))
  for b in range(num_bins):
    min_bin_idx = b * bin_size
    max_bin_idx = min((b+1) * bin_size, m.shape[dimension])
    binned_matrix[:,b] = np.sum(m[:,min_bin_idx:max_bin_idx], axis=1)

  return binned_matrix