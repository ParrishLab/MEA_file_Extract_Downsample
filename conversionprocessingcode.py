# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:23:30 2025

@author: sthor
"""

# import necessary librarys

import torch 
import scipy.io
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
#from slice_creator import Slice
import random

# Updated slice class! Comments on methods that are altered

import scipy.io
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

class Slice:

    '''
    Attributes:
    er_data: dictionary where channel number is key and values are arrays of data (millions of elements long)
    regions: list of brain regions
    time_steps: int number of discrete time steps recorded in er_data
    sampling_rate: number of time steps per second (default is 1000)
    starting_time: first time step that's stored (default is 0)
    genotype: which genotype the specimen has
    specimen_id: specimen ids
    slice_num: which slice number it is
    model: which convulsant media is used
    seizure_data: dictionary where the channel is the key and the value is the list
        [timesteps, ll_short, ll_trend, threshold, detections] array of seizure data for that channel. 
        timesteps is an indexing array for ll_short, ll_trend, threshold, and detections. 
        This dictionary is not automatically created. the gen_seizure_data or gen_all_seizure_data methods must be called
    seizure_metdata: dictionary where the channel is the key and the value is the list
        [short_window_size, step_size, trend_window_size, offset]
    _regions_dict: dictionary with keys as brain regions and values as lists of which channels (in utah array format) are in that region
    '''

    # info can be a string with the filename, or an array with filename, genotype, specimen_id, and slice_num
    # time_section is a tuple with the starting time step proportion of the way through and what proportion beyond it to keep
    def __init__(self, info):
        
        if isinstance(info, str):
            filename = info
            self.genotype = None
            self.specimen_id = None
            self.slice_num = None
        else:
            if not (isinstance(info, (list, tuple)) and len(info) == 4):
                raise ValueError("info must be filename or (filename, genotype, specimen_id, slice_num)")
            filename, genotype, specimen_id, slice_num = info
            self.genotype = genotype
            self.specimen_id = specimen_id
            self.slice_num = slice_num
        
        self.sampling_rate = 1000.0
        self.starting_time = 0.0
        self.model = None # This is overridden in __format data if the file has a model.
        raw_data = self.__load_mat_file(filename)
        self.er_data, self._regions_dict, self.regions, self.time_steps = self.__format_data(raw_data)
        self.seizure_data = {channel: None for channel in self.er_data}
        self.seizure_metadata = {channel: None for channel in self.er_data}

    #Loads raw flie and turns it into dictionary  
    def __load_mat_file(self, filename):
        try:
            # Try with scipy (works for most non-v7.3 .mat files)
            mat = scipy.io.loadmat(filename)
            
            # Remove MATLAB-specific metadata keys
            mat = {k: v for k, v in mat.items() if not k.startswith("__")}
            return mat
        
        except NotImplementedError:
            # If it's v7.3 (HDF5), use h5py
            mat = {}
            with h5py.File(filename, "r") as f:
                def recurse(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        mat[name] = obj[()]  # Load dataset
                    elif isinstance(obj, h5py.Group):
                        for subname, subobj in obj.items():
                            recurse(f"{name}/{subname}", subobj)
                
                for key, item in f.items():
                    recurse(key, item)
            return mat

    # Formats the raw data into a er_data dictionary
    def __format_data(self, raw_data):

        channel_lists = []
        er_data_1d = []
        regions_dict = {}
        regions = []

        for key in raw_data.keys():
            if 'recA' in key:
                er_data_1d.append(raw_data[key][0,0]['data'] )
                channels = raw_data[key][0,0]['ElecID']
                channel_lists.append(channels)
                region = key.split('recA')[0]
                channels.T[0].sort()
                regions_dict[region] = channels.T[0]
                regions.append(region)
            if 'Model' in key:
                self.model = raw_data[key]

        time_steps = len(er_data_1d[0][0,:])
        er_data = {}

        for i in range(len(channel_lists)):
            for j in range(len(channel_lists[i])):
                channel = channel_lists[i][j][0]
                new_er_data_list = er_data_1d[i][j,:]
                er_data[channel] = new_er_data_list

        del channel_lists, er_data_1d 

        return er_data, regions_dict, regions, time_steps

    # Given a channel in either matrix or Utah array form, return utah array number
    def __to_utah(self, channel):

        if isinstance(channel, tuple) or isinstance(channel, list):
            return channel[0] * 10 + (channel[1] + 1)
        else:
            return int(channel)

    # Given a channel in either matrix or Utah array form, return matrix row, column
    def __to_matrix(self, channel):

        if isinstance(channel, tuple) or isinstance(channel, list):
            row = channel[0]
            col = channel[1]
        else:
            row = (channel - 1) // 10
            col = (channel - 1) % 10 

        return (row,col)

    # Convert a time in seconds to an integer index.
    def __time_to_index(self, time):
        idx = int(round((time - self.starting_time) * self.sampling_rate))
        if self.time_steps > 0:
            idx = max(0, min(idx, self.time_steps - 1))
        else:
            idx = 0
        return idx

    # Convert an integer index to the corresponding time in seconds.
    def __index_to_time(self, idx):
        return self.starting_time + (idx / self.sampling_rate)

    # Sub-method of plot_er_segment, called to plot the trace if needed
    def __plot_trace(self, ax, channel, start_idx, end_idx):
        vec = self.er_data[channel]
        segment = vec[start_idx:end_idx]
        x = [self.__index_to_time(i) for i in range(start_idx, end_idx)]
        ax.plot(x, segment, linewidth=0.8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage")

    # Sub-method of plot_er_segment, called to plot the seizure data if needed
    def __plot_seizure(self, ax, channel, start_idx, end_idx):
        timesteps, ll_short, ll_trend, threshold, *_ = self.seizure_data[channel]

        mask = (timesteps >= start_idx) & (timesteps <= end_idx)

        timesteps = timesteps[mask]
        ll_short = ll_short[mask]
        ll_trend = ll_trend[mask]
        threshold = threshold[mask]

        times = self.__index_to_time(timesteps)
        ax.plot(times, ll_short, label='Short-term Line Length', color='black')
        ax.plot(times, ll_trend, label='Trend (Long-term LL)', color='orange')
        ax.plot(times, threshold, '--', label='Threshold', color='red')
        ax.set_ylabel('Line Length')
        ax.legend()

    # Plot electrophysiology recording (er) for a certain channel
    def plot_er_segment(self, channel, time_section = (None,None), trace = True, show_seizures = True, line_lengths = True, save = False):
        """
        Plot a segment of a very large 1D vector as a line graph.
        
        Parameters
        ----------
        channel : int channel number OR (row, column) spot on grid
        time_section: tuple of start and end time (in seconds) to plot
        trace: bool plot the trace of the er data
        seizure_data: bool plot the seizure data
        save: bool should the plot be saved
        """

        if (self.seizure_data[channel] is None) and (show_seizures or line_lengths):
            print('Seizure data has not yet been computed')
            show_seizures = False
            line_lengths = False

        start, end = time_section
        
        if start is None:
            start = self.__index_to_time(0)
        if end is None:
            end = self.__index_to_time(self.time_steps - 1)
        start_idx = self.__time_to_index(start)
        end_idx = self.__time_to_index(end)

        if end_idx <= start_idx:
            raise ValueError("end time must be after start time")

        channel = self.__to_utah(channel)

        if channel not in self.er_data:
            raise KeyError(f"Channel {channel} not found")

        fig, ax1 = plt.subplots(figsize=(12, 4))

        if trace and line_lengths:
            ax2 = ax1.twinx()
            self.__plot_trace(ax1, channel, start_idx, end_idx)
            self.__plot_seizure(ax2, channel, start_idx, end_idx)
            ax1.set_title(f"Electrophysiology Recording with Seizure Detection: {start} to {end}")
        elif trace:
            self.__plot_trace(ax1, channel, start_idx, end_idx)
            ax1.set_title(f"Electrophysiology Recording: {start} to {end}")
        elif line_lengths:
            self.__plot_seizure(ax1, channel, start_idx, end_idx)
            ax1.set_title(f"Seizure Detection: {start} to {end}")

        if show_seizures:
            timesteps, *_, detections = self.seizure_data[channel]

            mask = (timesteps >= start_idx) & (timesteps <= end_idx)
            timesteps = timesteps[mask]
            detections = detections[mask]
            
            times = self.__index_to_time(timesteps)
            ymin, ymax = ax1.get_ylim()
            ax1.fill_between(times, ymin, ymax, where=detections, color='red', alpha=0.2, label='Seizure Detected')

        fig.tight_layout()
        plt.show()

    # Plot which channels are activated in a given slice
    def plot_channels(self, time = 0, cmap="viridis"):
        """
        Plot a 2D heatmap of er_data for each channel at a given time.
        Zeros are shown as black, nonzeros follow the colormap.
        Origin is at top-left:
        - x = spots in from the left
        - y = spots down from the top
        """
        
        grid = np.zeros((10,10))
        
        time_idx = self.__time_to_index(time)

        for channel in range(1,101):
            if channel in self.er_data.keys():
                (row,col) = self.__to_matrix(channel)
                grid[row,col] = self.er_data[channel][time_idx]
            
        # Mask zeros
        masked_grid = np.ma.masked_where(grid == 0, grid)

        # Use chosen colormap, but set masked (zero) values to black
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="black")

        plt.figure(figsize=(6, 6))
        im = plt.imshow(masked_grid, cmap=cmap_obj, origin="upper")

        # Add colorbar
        plt.colorbar(im, label="Voltage")

        # Label axes
        plt.xlabel("Column")
        plt.ylabel("Row")

        plt.title(f"Heatmap at time {time}")
        plt.tight_layout()
        plt.show()

    # Given a channel, return its region (or None if it has no region)
    def find_region(self, channel):
        channel = self.__to_utah(channel)
        for key,values in self._regions_dict.items():
            if channel in values:
                return key
        return None

    # Given a region, return a list of its channels
    def find_channels(self, region):
        if region in self._regions_dict:
            return self._regions_dict[region]
        else:
            return []

    # Shrink the time steps stored to free up memory
    def shrink_time(self, new_start, new_end):

        if new_end <= new_start:
            raise ValueError("end time must be after start time")

        new_start_idx = self.__time_to_index(new_start)
        new_end_idx = self.__time_to_index(new_end)

        for channel in self.er_data.keys():
            self.er_data[channel] = self.er_data[channel][new_start_idx:new_end_idx]

        self.time_steps = new_end_idx - new_start_idx
        self.starting_time += new_start

    # Downsample to a given number of timesteps
    def downsample(self, new_sampling_rate):
        old_time = np.linspace(0, 1, self.time_steps)
        new_time = np.linspace(0, 1, new_sampling_rate)

        new_er_data = {}
        for key, vec in self.er_data.items():
            f = interp1d(old_time, vec, kind='linear')
            new_er_data[key] = f(new_time)

        # Update attributes
        self.er_data = new_er_data
        self.sampling_rate *= (new_sampling_rate / self.time_steps)
        self.time_steps = new_sampling_rate

    # Compute the line length across a signal with a given window size and step between windows
    def __line_length(self, signal, window_size, step_size):
        """
        Compute short-term line length for a signal using a sliding window.
        
        Parameters
        ----------
        signal : np.ndarray
            1D array of signal values.
        window_size : int
            Window length in samples (short-term window, e.g., 250 for 1s at 250Hz).
        step_size : int
            Step size between windows (e.g., 50 for 0.2s overlap).
        
        Returns
        -------
        ll_values : np.ndarray
            Array of line length values for each window.
        times : np.ndarray
            Center time of each window.
        """
        n = len(signal)
        ll_values = []
        times = []

        for start in range(0, n - window_size, step_size):
            end = start + window_size
            window = signal[start:end]
            # Line length formula (Eq. 4): sum of absolute differences
            ll = np.sum(np.abs(np.diff(window)))
            ll_values.append(ll / window_size)  # Normalized (K = 1/N)
            times.append(start + window_size // 2)
            
        return np.array(ll_values), np.array(times)

    # Compute the long term trend given the line lenght signal from a short term window
    def __compute_trend(self, ll_values, trend_window_size):
        """
        Compute the long-term trend (baseline) of the line length signal.
        
        Parameters
        ----------
        ll_values : np.ndarray
            Short-term line length values.
        trend_window_size : int
            Number of samples (of LL) over which to smooth.
        
        Returns
        -------
        trend : np.ndarray
            Smoothed (long-term) line length trend.
        """
        trend = np.convolve(ll_values, np.ones(trend_window_size)/trend_window_size, mode='same')
        return trend

    # For each poitn in the short term trend, determine if a seizure is occurring
    def __detect_seizures(self, ll_short, ll_trend, offset):
        """
        Detect seizure onsets based on line length thresholding.

        Parameters
        ----------
        ll_short : np.ndarray
            Short-term line length values.
        ll_trend : np.ndarray
            Long-term trend values.
        offset : float
            Threshold offset (either relative or absolute).
        relative : bool
            If True, offset is a percentage (e.g., 0.2 = 20% above trend).
        
        Returns
        -------
        detections : np.ndarray
            Boolean array: True where seizure is detected.
        """
        threshold = ll_trend * (1 + offset)
        
        detections = ll_short > threshold
        return detections, threshold

    # Detect seizures for a given channel and adds findings to self.seizure_data
    def gen_seizure_data(self, channel, short_window_size, step_size, trend_window_size, offset):
        
        signal = self.er_data[channel]
        ll_short, timesteps = self.__line_length(signal, short_window_size, step_size)
        trend_window_size = int(trend_window_size / step_size)
        ll_trend = self.__compute_trend(ll_short, trend_window_size)
        
        detections, threshold = self.__detect_seizures(ll_short, ll_trend, offset)
    
        self.seizure_data[channel] = [timesteps, ll_short, ll_trend, threshold, detections]
        self.seizure_metadata[channel] = [short_window_size, step_size, trend_window_size, offset]

    # Detect seizures in every channel
    def gen_all_seizure_data(self, short_window_size, step_size, trend_window_size, offset):
        for channel in tqdm(self.er_data.keys(), desc='Computing seizure data for each channel'):
            self.gen_seizure_data(channel, short_window_size, step_size, trend_window_size, offset)

def save_pickle(file_path, myslice):
    with open(file_path, 'wb') as file:
            pickle.dump(myslice, file)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        myslice = pickle.load(file)
    return myslice 

# Recreating the pickle files from all the files I have so far

filenames = [["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-714232/Newcastle University File Dnm1 Slice 1 (2).mat", 'Dnm1', 'unkown', 1], 
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-714232/Newcastle University File Dnm1 Slice 2.mat", 'Dnm1', 'unkown', 2],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-714232/Newcastle University File Slice 2.mat", 'unknown', 'unkown', 1],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-714232/Newcastle University File Dnm1 Slice 4.mat", 'Dnm1', 'unkown', 4],

            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut_id2391_slice1.mat", 'Glut1', '2391', 1],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut1_2388_slice1.mat", 'Glut1', '2388', 1],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut1_2388_slice2.mat", 'Glut1', '2388', 2],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut1_2388_slice4.mat", 'Glut1', '2388', 4],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut1_2390_slice1.mat", 'Glut1', '2390', 1],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-ofVWfHcbk7yoSZg9/Glut1_2390_slice2.mat", 'Glut1', '2390', 2],
            
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-YTwvmPrNSoPY7tFP/Glut1_2391_slice3.mat", 'Glut1', '2391', 3],
            ["/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Newcastle University File Drop-Off-YTwvmPrNSoPY7tFP/Glut1_2392_slice2.mat", 'Glut1', '2392', 2]]

for info in tqdm(filenames, desc="Processing files"):
    new_slice = Slice(info)
    save_pickle(f'{new_slice.genotype}_{new_slice.specimen_id}_{new_slice.slice_num}.pkl', new_slice)
    del new_slice
    