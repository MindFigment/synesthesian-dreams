import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.prev_val = 0
        self.anger = 0
        self.stop_early = False

    
    def feed(self, value):
        if self.prev_val <= value + self.delta:
            self.anger += 1
            if self.verbose:
                print(f"Early stopping anger: {self.anger} out of {self.patience}")
            if self.anger > self.patience:
                self.stop_early = True
                print(f"Anger reaches it's limit, training to coming to the end :(")
                exit(-1)
        else:
            self.anger = 0
        
        self.prev_val = value
            



class EarlyStopping2(object):
    def __init__(self, patience=10, low_threshold=0.009, up_threshold=0.99, verbose=False):
        self.patience = patience
        self.low_t = low_threshold
        self.up_t = up_threshold
        self.verbose = verbose
        self.anger = 0
        self.stop_early = False

    
    def feed(self, value):
        if value <= self.low_t or value >= self.up_t:
            self.anger += 1
            if self.verbose:
                print(f"Early stopping anger: {self.anger} out of {self.patience}")
            if self.anger > self.patience:
                self.stop_early = True
                print(f"Anger reaches it's limit, training to coming to the end :(")
                exit(-1)
        else:
            self.anger = 0

