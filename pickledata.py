# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:37:29 2025

@author: sthor
"""
from tqdm import tqdm
import os
import pickle
from slice_creator import Slice
import matplotlib.pyplot as plt
import numpy as np
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        myslice = pickle.load(file)
    return myslice 


# Generate n random plots from anywhere in the data
directory = "C:/Users/sthor/Box/Parrish_lab_folder/Utah-array-NCL/Calebs_Analysis/Pickle Files"

filepaths = []
for filename in tqdm(os.listdir(directory), desc="Processing files"):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):  # ensures it’s not a folder
        filepaths.append(filepath)

slices = []
#for filepath in tqdm(filepaths, desc="Creating slices"):
 #   slices.append(load_pickle(filepath))
slices.append(load_pickle(filepaths[1]))

import random

n = 2

for i in range(n):
    slice_num = random.randint(0,len(slices)-1)
    myslice = slices[slice_num]
    possible_channels = list(myslice.er_data.keys())
    channel = int(random.choice(possible_channels))
    print(f"Slice number {slice_num}")
    print(f"Channel number {channel}")
    myslice.plot_er_segment(channel)
    
    
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

import random

n = 2
for i in range(n):
    slice_num = random.randint(0, len(slices) - 1)
    myslice = slices[slice_num]
    possible_channels = list(myslice.er_data.keys())
    channel = int(random.choice(possible_channels))
    
    print(f"\n=== Plot {i+1} ===")
    print(f"File path: {filepaths[slice_num]}")
    print(f"Slice number: {slice_num}")
    print(f"Channel number: {channel}")
    
    row_col = myslice._Slice__to_utah(channel)
    print(f"Row, Column on Utah array: {row_col}")
    
    myslice.plot_er_segment(channel)
    myslice.plot_channels(0) # Plots the Utah array voltages at time 0

channel = 56

# Plot both figures
myslice.plot_er_segment(channel)
myslice.plot_channels(0)

# Get y-values for the plotted trace
utah_channel = myslice._Slice__to_utah(channel)
y_values = myslice.er_data[utah_channel]

# Get grid position for that channel in heatmap
row_col = myslice._Slice__to_matrix(channel)

print(f"Channel {channel} grid position -> Row {row_col[0]}, Column {row_col[1]}")
print(f"Y-values vector length: {len(y_values)}")
print("First 10 y-values:", y_values[:10])


channels_with_data = list(myslice.er_data.keys())
print("Channels with data:", channels_with_data)
print("Total:", len(channels_with_data))
 
channel_info = []  # will hold tuples of (channel, row, col, y_values)

for ch in channels_with_data:
    row, col = myslice._Slice__to_matrix(ch)
    y_values = myslice.er_data[ch]
    channel_info.append((ch, row, col, y_values))

print(f"Stored data for {len(channel_info)} valid channels.")

# find length of each trace
first_ch = next(iter(myslice.er_data))
n_time = len(myslice.er_data[first_ch])

# make an array for 100 channels × n_timepoints
data_matrix = np.full((100, n_time), np.nan)

# fill rows according to Utah array grid order
for ch in range(1, 101):
    if ch in myslice.er_data:
        r, c = myslice._Slice__to_matrix(ch)
        row_index = r * 10 + c          # 0–99 in grid order
        data_matrix[row_index, :] = myslice.er_data[ch]

print("data_matrix shape:", data_matrix.shape)
def get_utah_data_matrix(myslice):
    """
    Return a 2D array where each row corresponds to a Utah array channel's
    y-values (time-series data), ordered by their (row, column) position
    in the Utah array layout.
    """
    grid_size = 10  # 10x10 Utah array
    num_channels = grid_size * grid_size
    
    # Find one valid channel to get time length
    first_key = next(iter(myslice.er_data.keys()))
    time_length = len(myslice.er_data[first_key])
    
    # Initialize 2D matrix (100 rows, N columns)
    data_matrix = np.zeros((num_channels, time_length))
    
    # Iterate through all possible grid positions
    for row in range(grid_size):
        for col in range(grid_size):
            # Convert (row, col) to channel number
            try:
                channel_num = myslice._Slice__to_matrix_inverse(row, col)
            except AttributeError:
                # If you don't have an inverse, just find manually
                # by looping through existing channels
                for ch in myslice.er_data.keys():
                    r, c = myslice._Slice__to_matrix(ch)
                    if (r, c) == (row, col):
                        channel_num = ch
                        break
            
            # Skip if channel doesn't exist
            if channel_num not in myslice.er_data:
                continue
            
            # Get data for this channel
            y_values = myslice.er_data[channel_num]
            index = row * grid_size + col
            data_matrix[index, :] = y_values
    
    return data_matrix

# find length of each trace
first_ch = next(iter(myslice.er_data))
n_time = len(myslice.er_data[first_ch])

# make an array for 100 channels × n_timepoints
data_matrix = np.full((100, n_time), np.nan)

# fill rows according to Utah array grid order
for ch in range(1, 101):
    if ch in myslice.er_data:
        r, c = myslice._Slice__to_matrix(ch)
        row_index = r * 10 + c          # 0–99 in grid order
        data_matrix[row_index, :] = myslice.er_data[ch]

print("data_matrix shape:", data_matrix.shape)

import pandas as pd
df = pd.DataFrame(data_matrix,
                  index=[f"R{r}C{c}" for r in range(10) for c in range(10)])
channels_with_data = list(myslice.er_data.keys())
print("Channels with data:", channels_with_data)
print("Total:", len(channels_with_data))

channel_info = []  # will hold tuples of (channel, row, col, y_values)

for ch in channels_with_data:
    row, col = myslice._Slice__to_matrix(ch)
    y_values = myslice.er_data[ch]
    channel_info.append((ch, row, col, y_values))

print(f"Stored data for {len(channel_info)} valid channels.")

for ch, r, c, y in channel_info[:5]:
    print(f"Channel {ch} -> Row {r}, Col {c}, Trace length: {len(y)}")


rows = []
for ch, r, c, y in channel_info:
    rows.append({
        "channel": ch,
        "row": r,
        "col": c,
        "trace_length": len(y)
    })

df = pd.DataFrame(rows)
print(df.head())

row_data = [(ch, *myslice._Slice__to_matrix(ch)) for ch in channels_with_data]
print(row_data[:10])

#rol-col

data = {
    "channel": [33, 34, 41, 42, 43, 44, 45, 50, 51, 52, 53, 55, 56, 60, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 75, 76, 81, 82, 83, 85, 86, 93, 94, 95],
    "row": [3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9],
    "col": [2, 3, 0, 1, 2, 3, 4, 9, 0, 1, 2, 4, 5, 9, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 4, 5, 2, 3, 4],
    "trace_length": [3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014, 3690014]
}

df = pd.DataFrame(data)

# Make (row, col) tuples as plain integers
row_col = [tuple(map(int, x)) for x in zip(df["row"], df["col"])]

print("Row-Col tuples:")
print(row_col)

#traces for all 34 valid channels

trace_length = [
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014),
    np.random.rand(3690014)
]

# Flatten all traces end-to-end into one continuous vector
continuous_trace = np.concatenate(trace_length, axis=0)

print("Continuous trace shape:", continuous_trace.shape)
print(continuous_trace)

#verify rol/col and continuous trace vectors for all 34 valid channels

# Example variables:
channels = channels_with_data          # list of 34 valid channel IDs
row_col = np.array(row_col)  # list of (row, col) pairs
continuous_trace = continuous_trace         # your concatenated data

# 1️⃣ Check number of channels
num_channels = len(channels)
print(f"Number of channels: {num_channels}")

# 2️⃣ Check tuple shape
print(f"Row/col tuple shape: {row_col.shape}")
assert row_col.shape == (num_channels, 2), "Row/col tuple shape mismatch!"

# 3️⃣ Check that each trace is N elements long
# (assuming myslice.er_data[ch] gives one channel’s signal)
lengths = [len(myslice.er_data[ch]) for ch in channels]
unique_lengths = np.unique(lengths)
print(f"Unique trace lengths: {unique_lengths}")

# 4️⃣ Check combined continuous trace length
expected_total = num_channels * unique_lengths[0]
print(f"Continuous trace shape: {continuous_trace.shape}")
print(f"Expected total length: {expected_total}")

# 5️⃣ Verify consistency
if continuous_trace.size == expected_total:
    print("✅ Continuous trace length is consistent with 34 channels × N samples each.")
else:
    print("❌ Length mismatch! Check how you concatenated your traces.")


#for this, I had to do an int16 conversion from float64 

print(continuous_trace.dtype) #checks what trace this is currently in

#checks and gives everything before the conversion
first_channel = next(iter(myslice.er_data.keys()))
y = myslice.er_data[first_channel]
print("Trace length:", len(y))
print("First 10 values of raw trace:", y[:10])
print("Data type:", type(y[0]))

#now the actual converting
# 1️⃣ Gather all trace arrays from valid channels
channels_with_data = list(myslice.er_data.keys())
trace_list = [myslice.er_data[ch] for ch in channels_with_data]

# 2️⃣ Concatenate them into one continuous vector
continuous_trace_float = np.concatenate(trace_list, axis=0)

print("Before conversion:")
print("  dtype:", continuous_trace_float.dtype)
print("  min:", np.min(continuous_trace_float))
print("  max:", np.max(continuous_trace_float))
print("  first 10:", continuous_trace_float[:10])

# 3️⃣ Convert to int16 safely (scale first to preserve information)
scaled_trace = np.interp(
    continuous_trace_float,
    (np.min(continuous_trace_float), np.max(continuous_trace_float)),
    (np.iinfo(np.int16).min, np.iinfo(np.int16).max)
).astype(np.int16)

print("\nAfter conversion:")
print("  dtype:", scaled_trace.dtype)
print("  min:", np.min(scaled_trace))
print("  max:", np.max(scaled_trace))
print("  first 10:", scaled_trace[:10])

#sanity checks to verify literally everything to make sure it is int16 correctly lol
print(scaled_trace.dtype)
print(np.issubdtype(scaled_trace.dtype, np.int16))
print(scaled_trace.itemsize)  # bytes per element
print(np.min(scaled_trace), np.max(scaled_trace))