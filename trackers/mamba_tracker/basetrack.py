import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Virtual = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    virtual_frames = 0

    # multi-camera
    location = (np.inf, np.inf)

    @staticmethod
    def reset_id():
        """Reset the track ID counter to 0."""
        BaseTrack._count = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_virtual(self):
        self.state = TrackState.Virtual
        self.virtual_frames =1
    
    def update_frame_id(self):
        self.virtual_frames +=1

    def mark_removed(self):
        self.state = TrackState.Removed
        