# Updated slice class! Comments on methods that are altered

import scipy.io
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.ndimage import binary_closing, binary_opening

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
        [times, detections] array of seizure data for that channel. 
        times[i] is the time when detections[i] occur. 
        This dictionary is not automatically created. the gen_seizure_data or gen_all_seizure_data methods must be called
    seizure_metdata: dictionary where the channel is the key and the value is the another dictionary containing metadata which depends on the seziure data generation method
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
                self.model = raw_data[key][0]

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
    def time_to_index(self, time):
        idx = (time - self.starting_time) * self.sampling_rate
        idx = np.asarray(idx).astype(int) if np.ndim(idx) > 0 else int(idx)
        if self.time_steps > 0:
            idx = np.clip(idx, 0, self.time_steps - 1)
        else:
            idx = 0
        return idx

    # Convert an integer index to the corresponding time in seconds.
    def index_to_time(self, idx):
        return self.starting_time + (idx / self.sampling_rate)

    # Converts from a duration to a number of timesteps using sampling rate 
    def duration_to_timesteps(self, duration):
        return duration * self.sampling_rate   

    # Converts from a number of timestep to a duration using sampling rate
    def timesteps_to_duration(self, timesteps):
        return timesteps / self.sampling_rate
    
    # Sub-method of plot_er_segment, called to plot the trace if needed
    def __plot_trace(self, ax, channel, start_idx, end_idx):
        vec = self.er_data[channel]
        segment = vec[start_idx:end_idx]
        x = [self.index_to_time(i) for i in range(start_idx, end_idx)]
        ax.plot(x, segment, linewidth=0.8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage")

    # Sub-method of plot_er_segment, called to plot the ll seizure data if needed
    def __plot_ll(self, ax, channel, start_idx, end_idx):
        times = self.seizure_data[channel][0]
        ll_short = self.seizure_metadata[channel]['ll_short']
        ll_trend = self.seizure_metadata[channel]['ll_trend']
        threshold = self.seizure_metadata[channel]['threshold']

        timesteps = self.time_to_index(times)
        mask = (timesteps >= start_idx) & (timesteps <= end_idx)

        timesteps = timesteps[mask]
        ll_short = ll_short[mask]
        ll_trend = ll_trend[mask]
        threshold = threshold[mask]
        times = self.index_to_time(timesteps)

        ax.plot(times, ll_short, label='Short-term Line Length', color='black')
        ax.plot(times, ll_trend, label='Trend (Long-term LL)', color='orange')
        ax.plot(times, threshold, '--', label='Threshold', color='red')
        ax.set_ylabel('Line Length')
        ax.legend()

    # Sub-method of plot_er_segment, called to plot the var seizure data if needed
    def __plot_var(self, ax, channel, start_idx, end_idx):
        times = self.seizure_data[channel][0]
        baseline = self.seizure_metadata[channel]['baseline']
        variances = self.seizure_metadata[channel]['variances']
        variance_exp = self.seizure_metadata[channel]['variance_exp']

        timesteps = self.time_to_index(times)
        mask = (timesteps >= start_idx) & (timesteps <= end_idx)

        timesteps = timesteps[mask]
        variances = variances[mask]
        times = self.index_to_time(timesteps)

        ax.plot(times, variances, color='black')
        ax.axhline(y=baseline, color = 'red')
        if variance_exp == 1:
            ax.set_ylabel(f'Variance')
        else:
            ax.set_ylabel(f'Variance^{variance_exp}')

    # Plot electrophysiology recording (er) for a certain channel
    def plot_er_segment(self, channel, time_section = (None,None), trace = True, show_seizures = True, intermediate_step = True, save = False, filename = None):
        """
        Plot a segment of a very large 1D vector as a line graph.
        
        Parameters
        ----------
        channel : int channel number OR (row, column) spot on grid
        time_section: tuple of start and end time (in seconds) to plot
        trace: bool plot the trace of the er data
        intermediate_step: bool plot the intermediate step (variance for variance method or line length for line length method)
        save: bool should the plot be saved
        """

        if (self.seizure_data[channel] is None) and (show_seizures or intermediate_step):
            print('Seizure data has not yet been computed')
            show_seizures = False
            intermediate_step = False

        start, end = time_section
        
        if start is None:
            start = self.index_to_time(0)
        if end is None:
            end = self.index_to_time(self.time_steps)
        start_idx = self.time_to_index(start)
        end_idx = self.time_to_index(end)

        if end_idx <= start_idx:
            print(end)
            print(end_idx)
            print(start)
            print(start_idx)
            raise ValueError("end time must be after start time")

        channel = self.__to_utah(channel)

        if channel not in self.er_data:
            raise KeyError(f"Channel {channel} not found")

        fig, ax1 = plt.subplots(figsize=(12, 4))

        if trace and intermediate_step:
            ax2 = ax1.twinx()
            self.__plot_trace(ax1, channel, start_idx, end_idx)
            if self.seizure_metadata[channel]['method'] == 'll':
                self.__plot_ll(ax2, channel, start_idx, end_idx)
            elif self.seizure_metadata[channel]['method'] == 'var':
                self.__plot_var(ax2, channel, start_idx, end_idx)
            ax1.set_title(f"Electrophysiology Recording with Seizure Detection: {start} to {end}")
        elif trace:
            self.__plot_trace(ax1, channel, start_idx, end_idx)
            ax1.set_title(f"Electrophysiology Recording: {start} to {end}")
        elif intermediate_step:
            if self.seizure_metadata[channel]['method'] == 'll':
                self.__plot_ll(ax1, channel, start_idx, end_idx)
            elif self.seizure_metadata[channel]['method'] == 'var':
                self.__plot_var(ax1, channel, start_idx, end_idx)
            ax1.set_title(f"Seizure Detection: {start} to {end}")

        if show_seizures:
            times, detections = self.seizure_data[channel]

            timesteps = self.time_to_index(times)
            mask = (timesteps >= start_idx) & (timesteps <= end_idx)
            timesteps = timesteps[mask]
            detections = detections[mask]
            times = self.index_to_time(timesteps)

            ymin, ymax = ax1.get_ylim()
            ax1.fill_between(times, ymin, ymax, where=detections, color='red', alpha=0.2, label='Seizure Detected')

        fig.tight_layout()
        if save:
            if filename is None:
                filename = f"{self.genotype}_{self.specimen_id}_{channel}"
            plt.savefig(filename)
        else:
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
        
        time_idx = self.time_to_index(time)

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

    # Given a channel, return its region (None if it has no region)
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

        new_start_idx = self.time_to_index(new_start)
        new_end_idx = self.time_to_index(new_end)

        for channel in self.er_data.keys():
            self.er_data[channel] = self.er_data[channel][new_start_idx:new_end_idx]

        self.time_steps = new_end_idx - new_start_idx
        self.starting_time += new_start

    # Downsamples every element of er_data
    def downsample(self, new_sample_rate: float):
        """
        Downsample all signals in er_data to a new sampling rate using linear interpolation.

        ```
        Parameters
        ----------
        new_sample_rate : float
            Desired new sampling rate (samples per second). Must be less than the current sampling_rate.
        """
        if new_sample_rate <= 0:
            raise ValueError("new_sample_rate must be positive.")
        if new_sample_rate > self.sampling_rate:
            raise ValueError("new_sample_rate must be less than or equal to the current sampling rate.")

        # Compute the time axis for original and new sampling rates
        old_rate = self.sampling_rate
        n_samples = self.time_steps
        duration = n_samples / old_rate

        old_time = np.linspace(0, duration, n_samples, endpoint=False)
        new_time_steps = int(np.floor(duration * new_sample_rate))
        new_time = np.linspace(0, duration, new_time_steps, endpoint=False)

        # Interpolate each channel
        new_er_data = {}
        for ch, signal in self.er_data.items():
            signal = np.asarray(signal)
            new_signal = np.interp(new_time, old_time, signal)
            new_er_data[ch] = new_signal

        # Update attributes
        self.er_data = new_er_data
        self.sampling_rate = new_sample_rate
        self.time_steps = new_time_steps

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
            times.append(self.index_to_time(start + window_size // 2))

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
        threshold = ll_trend * (offset)
        
        detections = ll_short > threshold
        return detections, threshold

    # Smooth out small gaps in the detections
    def __smooth_detections(self, detections, min_block_size):
        # Fill small False gaps inside True regions (closing)
        closed = binary_closing(detections, structure=np.ones(min_block_size))
        # Remove small True spikes inside False regions (opening)
        opened = binary_opening(closed, structure=np.ones(min_block_size))
        return opened

    # Detect seizures based on the variance of successsive windows
    def __var_detect_seizures(self, data, window_size, min_block_size, step_size, offset, quantile, begin_baseline, end_baseline, variance_exp):
        indices = []
        variances = []
        step = 0
        while step + window_size <= len(data):
            data_segment = data[step:step + window_size]
            var = np.linalg.norm(data_segment)**variance_exp
            variances.append(var)
            indices.append(int(step + window_size/2))
            step += step_size

        if len(variances) == 0:
            print("Window Size is too big")
            return

        baseline = np.quantile(variances[max(0,int(begin_baseline*len(variances))):min(int(end_baseline*len(variances)), len(variances))],quantile)
        detections= np.zeros_like(np.array(variances)).astype(bool)

        for i in range(len(variances)):
            if variances[i] >= baseline * offset:
                detections[i] = True

        detections = self.__smooth_detections(detections, min_block_size)

        times = self.index_to_time(np.array(indices))

        return baseline, np.array(variances), np.array(detections), np.array(times)

    # Detect seizures for a given channel and adds findings to self.seizure_data
    # method should be each 'll' for line length or 'var' for variance
    # parameters is a dictionary whos elements depend on with method:
    # For method = 'll', parameters contains short_window_size, step_size, trend_window_size, offset, min_block_size
    def gen_seizure_data(self, method, channel, parameters):

        data = self.er_data[channel]
        if method == 'll':            
            short_window_size = int(parameters['short_window_size'])
            step_size = int(parameters['step_size'])
            trend_window_size = int(parameters['trend_window_size'])
            offset = parameters['offset']
            min_block_size = int(parameters['min_block_size'])

            ll_short, times = self.__line_length(data, short_window_size, step_size)
            trend_window_size = int(trend_window_size / step_size)
            ll_trend = self.__compute_trend(ll_short, trend_window_size)
            
            detections, threshold = self.__detect_seizures(ll_short, ll_trend, offset)
            detections = self.__smooth_detections(detections, min_block_size)
    
            self.seizure_data[channel] = [times, detections]
            self.seizure_metadata[channel] = {'method': method, 'll_short': ll_short, 'll_trend': ll_trend, 'threshold': threshold, 
                                              'short_window_size': short_window_size, 'step_size': step_size, 'trend_window_size': trend_window_size, 
                                              'offset': offset, 'min_block_size': min_block_size}

        elif method == 'var':
            window_size = int(parameters['window_size'])
            min_block_size = int(parameters['min_block_size'])
            step_size = int(parameters['step_size'])
            offset = parameters['offset']
            quantile = parameters['quantile']
            begin_baseline = parameters['begin_baseline']
            end_baseline = parameters['end_baseline']
            variance_exp = parameters['variance_exp']
            
            baseline, variances, detections, times = self.__var_detect_seizures(data, window_size, min_block_size, step_size, 
                                                                            offset, quantile, begin_baseline, end_baseline, variance_exp)
            self.seizure_data[channel] = [times, detections]
            self.seizure_metadata[channel] = {'method': method, 'baseline': baseline, 'variances': variances, 'variance_exp': variance_exp,
                                              'window_size': window_size, 'min_block_size': min_block_size, 'step_size': step_size, 
                                              'offset': offset, 'quantile': quantile, 'begin_baseline': begin_baseline, 'end_baseline': end_baseline}
        else:
            print("Invalid method to generate seizure data")

    # Detect seizures in every channel
    def gen_all_seizure_data(self, method, parameters):
        if not(method == 'll' or method == 'var'):
            print("Invalid method to generate seizure data")
            return
        for channel in tqdm(self.er_data.keys(), desc='Computing seizure data for each channel'):
            self.gen_seizure_data(method, channel, parameters)

    # Return the piece of the trace on which to do more computations given seizure detections
    def processed_data(self, fn, channel = None):
        """
        Return the piece of the trace on which to do more computations given seizure detections,
        and the start_duration matrix, each channel is a row. Column one is the start time of the returned trace
        and columns 2 is the duration of the returned trace

        ```
        Parameters
        ----------
        fn: function
            Takes in parameters (trace, start_idxs, end_idxs), where trace is a float array and start_idxs is an array of the indices of traces
            where seizures start and end_idxs is an array of the indices of trace where seizures end.
            Returns the new trace, the window start in terms of time steps, and the duration in terms of time_steps
        channel: int
            Determines which channel to return the array for. If None, it dose all of them
        """

        if channel == None:
            channels = self.er_data.keys()
        else:
            channels = [channel]
        
        final_matrix = []
        start_duration_matrix = []
        for channel in channels:
            trace = self.er_data[channel]
            times, detections = self.seizure_data[channel]
            detection_starts = np.where((detections == True) & (np.concatenate(([True], detections[:-1] == False))))[0]
            detection_ends   = np.where((detections == True) & (np.concatenate((detections[1:] == False, [True]))))[0]
            if len(detection_starts) == 0:
                print(f"Channel {channel} has no detected seizures.")
                continue
            start_times = np.array([times[idx] for idx in detection_starts])
            end_times = np.array([times[idx] for idx in detection_ends])
            start_idxs = self.time_to_index(start_times)
            end_idxs = self.time_to_index(end_times)
            new_trace, window_start_idx, duration_idx = fn(trace, start_idxs, end_idxs)
            if new_trace is None:
                print(f"Channel {channel} has no seizures that can be detected by the processing function.")
                continue
            final_matrix.append(new_trace)
            start_duration_matrix.append([self.index_to_time(window_start_idx), self.timesteps_to_duration(duration_idx)])
        
        return np.array(final_matrix), np.array(start_duration_matrix)

