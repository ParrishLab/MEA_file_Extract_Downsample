"""
Pipeline (no template .brw needed):
1. Load pickle slice
2. Extract traces + order with Utah coordinates
3. Flatten & scale to int16
4. Create a new BRW HDF5 file from scratch
   - /3BData/Raw
   - /3BRecInfo/3BMeaStreams/Raw/Chs
   - /3BRecInfo/3BMeaChip/*
   - /3BRecInfo/3BMeaSystem/*
   - /3BRecInfo/3BRecVars/*
   - /3BUserInfo/*
"""

import sys
import os
import pickle
import numpy as np
import h5py
from pathlib import Path
from slice_creator import Slice
import tkinter as tk
from tkinter import filedialog

# Ensure directory containing slice_creator.py is on Python path
slice_dir = str(Path(__file__).resolve().parent)
if slice_dir not in sys.path:
    sys.path.append(slice_dir)

# ============================================================
# GUI: PICK PATHS
# ============================================================

root = tk.Tk()
root.withdraw()

# 1) Pickle directory (optional but nice as a starting folder)
pickle_dir = filedialog.askdirectory(
    title="Select folder for Pickle Files"
)
if not pickle_dir:
    raise RuntimeError("No pickle directory selected; aborting.")

# 2) Select the pickle slice file
pickle_file = filedialog.askopenfilename(
    title="Select pickle slice to convert",
    initialdir=pickle_dir,
    filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
)
if not pickle_file:
    raise RuntimeError("No pickle file selected; aborting.")

# 3) Output BRW file: default name = same base as pickle, no extension
default_brw_name = Path(pickle_file).stem  # <- just 'name', no .pkl/.brw

working_copy = filedialog.asksaveasfilename(
    title="Select location/name for output .brw",
    defaultextension=".brw",          # Tk will append .brw
    initialfile=default_brw_name,      # e.g. "MySlice_01"
    filetypes=[("BRW files", "*.brw"), ("All files", "*.*")]
)

if not working_copy:
    raise RuntimeError("No output .brw file selected; aborting.")

print("pickle_dir:   ", pickle_dir)
print("pickle_file:  ", pickle_file)
print("working_copy: ", working_copy)

# ============================================================
# STEP 1 — LOAD ONE PICKLE
# ============================================================

def load_pickle(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

# Use the user-selected pickle file, not filepaths[0]
myslice = load_pickle(pickle_file)

# ============================================================
# STEP 2 — EXTRACT ALL VALID CHANNELS + TRACES
# ============================================================

channels_with_data = list(myslice.er_data.keys())
print(f"Valid channels found: {len(channels_with_data)}")

row_col = []
trace_list = []

for ch in channels_with_data:
    r, c = myslice._Slice__to_matrix(ch)  # 0-based Utah coords
    row_col.append((r, c))
    trace_list.append(myslice.er_data[ch])

row_col = np.array(row_col, dtype=np.int32)  # shape (Nch, 2)
print("row_col shape:", row_col.shape)

num_channels = len(channels_with_data)
samples_per_ch = trace_list[0].shape[0]

# Sanity check: all channels same length
for i, tr in enumerate(trace_list):
    if tr.shape[0] != samples_per_ch:
        raise ValueError(f"Channel {i} has different length: {tr.shape[0]} vs {samples_per_ch}")

# ============================================================
# STEP 3 — FLATTEN TRACES & CONVERT TO INT16
# ============================================================

continuous_trace_float = np.concatenate(trace_list, axis=0)

# scale to int16
scaled_trace = np.interp(
    continuous_trace_float,
    (continuous_trace_float.min(), continuous_trace_float.max()),
    (np.iinfo(np.int16).min, np.iinfo(np.int16).max)
).astype(np.int16)

print("Scaled trace dtype:", scaled_trace.dtype)
print("Scaled trace length:", scaled_trace.shape)

# ============================================================
# STEP 4 — CREATE BRW FILE FROM SCRATCH
# ============================================================

def create_brw_file(
    filename,
    scaled_trace,
    row_col,
    max_volt=4125.0,
    min_volt=-4125.0,
    bit_depth=12
):
    """
    Create a new BRW-like HDF5 file with the key groups/datasets:
      /3BData/Raw
      /3BRecInfo/3BMeaStreams/Raw/Chs
      /3BRecInfo/3BMeaChip/*
      /3BRecInfo/3BMeaSystem/*
      /3BRecInfo/3BRecVars/*
      /3BUserInfo/*
    `row_col` is 0-based; will be written as 1-based.
    """

    num_channels = row_col.shape[0]
    total_samples = scaled_trace.shape[0]
    samples_per_ch = total_samples // num_channels

    # Hard-code sampling rate here to be explicit
    sampling_rate_hz = 1000.0   # set to real sample rate: 1000 Hz

    print(f"\nCreating BRW file at: {filename}")
    print(f"  num_channels   = {num_channels}")
    print(f"  samples_per_ch = {samples_per_ch}")
    print(f"  total_samples  = {total_samples}")
    print(f"  num_channels * samples_per_ch = {num_channels * samples_per_ch}")
    print(f"  SamplingRate to write = {sampling_rate_hz} Hz")

    vlen_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(filename, "w") as f:
        # -----------------------------
        # /3BData
        # -----------------------------
        g_3BData = f.create_group("3BData")
        g_3BData.create_dataset(
            "Raw",
            data=scaled_trace,
            dtype='int16'
        )
        print("✅ Created /3BData/Raw")

        # -----------------------------
        # /3BRecInfo
        # -----------------------------
        g_3BRecInfo = f.create_group("3BRecInfo")

        # ----- 3BMeaChip -----
        g_chip = g_3BRecInfo.create_group("3BMeaChip")

        layout = np.ones((64, 64), dtype=np.uint8)
        g_chip.create_dataset("Layout", data=layout, dtype='uint8')

        g_chip.create_dataset("MeaType", data=np.array([65536], dtype=np.int32))
        g_chip.create_dataset("NCols",   data=np.array([64],    dtype=np.uint32))
        g_chip.create_dataset("NRows",   data=np.array([64],    dtype=np.uint32))

        syschs_dtype = np.dtype([('Row', '<i2'), ('Col', '<i2')])
        g_chip.create_dataset(
            "SysChs",
            data=np.array([(1, 1)], dtype=syschs_dtype),
            dtype=syschs_dtype
        )

        color_dtype = np.dtype([
            ('KnownColor', '<i4'),
            ('Alpha',      'u1'),
            ('Red',        'u1'),
            ('Green',      'u1'),
            ('Blue',       'u1'),
        ])
        roi_dtype = np.dtype([
            ('Name',      vlen_str),
            ('Color',     color_dtype),
            ('Chs',       vlen_str),
            ('IsVisible', 'u1'),
            ('Units',     vlen_str),
        ])
        g_chip.create_dataset(
            "ROIs",
            shape=(0,),
            maxshape=(None,),
            dtype=roi_dtype
        )

        print("✅ Created /3BRecInfo/3BMeaChip/*")

        # ----- 3BMeaStreams/Raw/Chs -----
        g_streams = g_3BRecInfo.create_group("3BMeaStreams")
        g_raw_stream = g_streams.create_group("Raw")

        row_col_1based = (row_col + 1).astype('<i2')
        chs_dtype = np.dtype([('Row', '<i2'), ('Col', '<i2')])
        chs_flat = row_col_1based.view(chs_dtype).reshape(-1)

        g_raw_stream.create_dataset(
            "Chs",
            data=chs_flat,
            dtype=chs_dtype
        )
        print("✅ Created /3BRecInfo/3BMeaStreams/Raw/Chs")

        # ----- 3BMeaSystem -----
        g_sys = g_3BRecInfo.create_group("3BMeaSystem")
        fw_hw_dtype = np.dtype([
            ('Major',    '<i4'),
            ('Minor',    '<i4'),
            ('Build',    '<i4'),
            ('Revision', '<i4'),
        ])

        g_sys.create_dataset(
            "FwVersion",
            data=np.array([(1, 4, 1, 4)], dtype=fw_hw_dtype),
            dtype=fw_hw_dtype
        )
        g_sys.create_dataset(
            "HwVersion",
            data=np.array([(3, 0, -1, -1)], dtype=fw_hw_dtype),
            dtype=fw_hw_dtype
        )
        g_sys.create_dataset(
            "System",
            data=np.array([1], dtype=np.int32)
        )
        print("✅ Created /3BRecInfo/3BMeaSystem/*")

        # ----- 3BRecVars -----
        g_vars = g_3BRecInfo.create_group("3BRecVars")

        g_vars.create_dataset("BitDepth",        data=np.array([bit_depth], dtype=np.uint8))
        g_vars.create_dataset("ExperimentType",  data=np.array([0],         dtype=np.int32))
        g_vars.create_dataset("MaxVolt",         data=np.array([max_volt],  dtype=np.float64))
        g_vars.create_dataset("MinVolt",         data=np.array([min_volt],  dtype=np.float64))

        # 🔑 NRecFrames: number of samples per channel
        g_vars.create_dataset(
            "NRecFrames",
            data=np.array([samples_per_ch], dtype=np.int64)
        )

        # 🔑 SamplingRate: force to 2000.0 Hz
        g_vars.create_dataset(
            "SamplingRate",
            data=np.array([sampling_rate_hz], dtype=np.float64)
        )

        g_vars.create_dataset(
            "SignalInversion",
            data=np.array([1.0], dtype=np.float64)
        )

        print("✅ Created /3BRecInfo/3BRecVars/*")

        # -----------------------------
        # /3BUserInfo
        # -----------------------------
        g_user = f.create_group("3BUserInfo")

        markers_dtype = np.dtype([
            ('Type',    '<i2'),
            ('MarkIn',  '<i8'),
            ('MarkOut', '<i8'),
            ('Desc',    vlen_str),
            ('Color',   color_dtype),
        ])
        g_user.create_dataset(
            "ExpMarkers",
            shape=(0,),
            maxshape=(None,),
            dtype=markers_dtype
        )

        notes_dtype = np.dtype([
            ('Title', vlen_str),
            ('Value', vlen_str),
        ])
        g_user.create_dataset(
            "ExpNotes",
            shape=(0,),
            maxshape=(None,),
            dtype=notes_dtype
        )

        print("✅ Created /3BUserInfo/*")

    print("\n🎉 All done — new .brw created without template.")
    print(f"   • {num_channels} channels")
    print(f"   • {samples_per_ch} samples per channel")
    print(f"   • total samples written to /3BData/Raw: {total_samples}")
    print(f"   • SamplingRate written: {sampling_rate_hz} Hz")

# Call: no sampling_rate_hz argument anymore, it’s fixed inside as 2000.0
create_brw_file(
    working_copy,
    scaled_trace=scaled_trace,
    row_col=row_col,
    max_volt=4125.0,
    min_volt=-4125.0,
    bit_depth=12
)

# ============================================================
# OPTIONAL: HDF5 SUMMARY (same as before)
# ============================================================

def print_h5_summary(filename, max_elems=100):
    """
    Recursively print the structure of an HDF5 file and show the first
    `max_elems` elements of each dataset (flattened if necessary).
    """
    def _preview_dataset(ds, max_elems=100):
        if ds.size == 0:
            return "[]"

        if ds.size <= max_elems:
            arr = ds[()]
            return repr(arr)

        shape = ds.shape
        if len(shape) == 1:
            arr = ds[:max_elems]
        else:
            n0 = min(max_elems, shape[0])
            arr = ds[:n0, ...]

        arr = np.array(arr)
        flat = arr.ravel()
        if flat.size > max_elems:
            flat = flat[:max_elems]
        return repr(flat)

    def _walk(name, obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{pad}Group: /{name}")
            for key, item in obj.items():
                child_name = f"{name}/{key}" if name else key
                _walk(child_name, item, indent + 1)
        elif isinstance(obj, h5py.Dataset):
            ds = obj
            print(f"{pad}Dataset: /{name}")
            print(f"{pad}  shape={ds.shape}, dtype={ds.dtype}")
            try:
                preview = _preview_dataset(ds, max_elems=max_elems)
                print(f"{pad}  data (first {max_elems} elements):")
                print(f"{pad}  {preview}")
            except Exception as e:
                print(f"{pad}  <error reading data: {e}>")

    with h5py.File(filename, "r") as f:
        print(f"\n=== HDF5 summary for: {filename} ===")
        for key, obj in f.items():
            _walk(key, obj, indent=0)

print_h5_summary(working_copy, max_elems=100)
