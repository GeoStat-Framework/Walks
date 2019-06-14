# -*- coding: utf-8 -*-
"""
Walk module providing plotting functionality.

.. currentmodule:: walks.plot

The following methods are provided

.. autosummary::
   video
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as pt
import matplotlib.animation as animation
import seaborn as sns


def video(x, y, pos, field=None, fps=10, title='Random Walks', filename='random_walks.mp4'):
    """Create a video of the  2d walkers.
    
    Parameters
    ----------
    x : :any:`numpy.ndarray`
        the x grid
    y : :any:`numpy.ndarray`
        the y grid
    pos : :any:`numpy.ndarray`
        the walker positions in time
    field : :any:`numpy.ndarray`, optional
        the velocity field
    fps : :class:`int`, optional
        frames per second of the video
    title : :class:`str`, optional
        title of the video
    filename : :class:`str`, optional
        filename of the video
    """
    print('Creating video...')
    c = sns.color_palette()
    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict(title=title, artist='Ministry of Random Walks')
    writer = ffmpeg_writer(fps=fps, metadata=metadata)

    fig = pt.figure()
    ax = fig.add_subplot(111)

    if field is not None:
        norm = np.sqrt(field[0,:].T**2 + field[1,:].T**2)
        ax.streamplot(x, y, field[0,:].T, field[1,:].T, color=norm, linewidth=norm/2)
    ax.set_xlabel(r'$x$ / m')
    ax.set_ylabel(r'$y$ / m')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    with writer.saving(fig, filename, dpi=150):
        frames = pos.shape[0]
        for t in range(frames):
            print('\tframe {:3d} / {}\r'.format(t+1, frames), end='', flush=True)
            lines = ax.plot(pos[0:t,0,:], pos[0:t,1,:], color=c[3], alpha=1., linewidth=0.3)
            writer.grab_frame()
            [l.remove() for l in lines]
            del lines

        print('\tframe {0:3d} / {0}'.format(frames), end='', flush=True)
