import os, os.path
import subprocess
import json
import logging

from phi.flow import *
from phi.field import Field 
from phi.math import Shape

import numpy as np
import h5py

from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


def dims_for(
        n_steps: int=1000,
        grid_size: tuple=(100, 100),
        t_sample_interval: int=1,
        n_batch: int=1,
        **kwargs):
    """
    return a dict of fields and their shapes
    """
    n_frames = ((n_steps-1)//t_sample_interval) + 1
    return {
        "velocity": (n_batch, n_frames, *grid_size, len(grid_size)),
        "density": (n_batch, n_frames, *grid_size, 1),
        "pressure": (n_batch, n_frames, *grid_size, 1),
        "t": (n_batch, n_frames),
        "initial_density": (n_batch, *grid_size, 1),
    }


def dict_for(config):
    spec = dims_for(**config)
    data_store = dict(
        latest_index = -1,
        config = config
    )
    for field_name, full_shape in spec.items():
        data_store[field_name] = np.ndarray(full_shape, dtype='float32')
    return data_store


def h5_for(config):
    log.info(f"config: {config}")
    spec = dims_for(**config)
    log.info(f"spec: {spec}")
    fname = f"{config['sim_name']}.h5"
    data_store = h5py.File(fname, 'a')
    data_store.attrs['config'] = OmegaConf.to_yaml(config)
    data_store.attrs['latestIndex'] = -1
    for field_name, full_shape in spec.items():
        # dataset shape is (batch, t_length, x1, ..., xd, v)
        # Open a dataset, creating it if it doesn’t exist.
        data_store.require_dataset( 
            field_name,
            full_shape,
            'float32',
            compression="lzf")
    return data_store


def to_centre_grid(field: Field) -> CenteredGrid:
    '''
    resample the input `Field` and return a corresponding `CenterGrid`
    used because the `StaggeredGrid`, which is usually the Object for velocity, does pack into nice tensors for typical neural nets
    '''
    if isinstance(field, CenteredGrid):
        return field
    return CenteredGrid(field, resolution=field.shape.spatial, bounds=field.bounds)


def _get_dim_order(shape: Shape):
    '''
    Return a tuple of string, represents the order of dimensions
    e.g. ('batch','x','y','vector')
    If the current Shape does not have channel dims, fill in "vector" as 1.
    '''
    batchNames = shape.batch.names if (shape.batch_rank > 0) else ('batch',)
    channelNames = shape.channel.names if (shape.channel_rank > 0) else ('vector',)
    return batchNames + ('y', 'x') + channelNames


def to_ndarray(field: Field) -> np.ndarray:
    '''
    Turn the current Field into ndarray, with shape (batch, x1, ..., xd, v)
    '''
    centered = to_centre_grid(field)
    order = _get_dim_order(centered.shape)
    ndarray = centered.values.numpy(order=order)
    return ndarray
