"""
Class History:

"""

from collections import deque, Counter
import numpy as np

class History:
    def __init__(self):
        self.max_size = 500000
        # history keeps all the previous values of data entries, for an certain data value of the socket
        self.window = deque(maxlen=self.max_size)
        self.count = Counter()
        # keep track of adress that were flaged as malicius
        self.bad_addres = deque(maxlen=self.max_size)
        self.count_bad = Counter()
        self.hidden_entropy = None
        self.entropy_review = True # define if entropy needs to be recalculated
        self.last_seen = {}
        

    def updat_h(self, entry):
        if len(self.window) == self.max_size:
            old = self.window[0]
            self.count[old] -= 1
            if self.count[old] == 0 :
                del self.count[old]
        self.window.append(entry)
        self.count[entry] += 1
        self.entropy_review = True
        self.last_seen[entry] = len(self.window)-1
        return
            

    def avg_rate(self, data_input):
        if len(self.window) == 0: return 0.0
        total = len(self.window)
        count = self.count.get(data_input, 0)
        
        if count == 0: return 0.0
        
        rate = count / total
        log_rate = np.log(rate+1e-6)   # add small value to avoid log(0)
        
        normalized = (log_rate + 6) / 6
        return np.clip(normalized, 0.0, 1.0)
    
    # returns the data type with the highest rate
    def get_max_rate(self):
        max_rate = 0
        max_value = None
        for d in self.window:
            v = self.avg_rate(d)
            if v > max_rate:
                max_rate = v
                max_value = d            
        return max_value, max_rate
    
    def get_window(self):
        return self.window
    
    def values(self):
        """Return the count values (makes History compatible with dict-like operations)"""
        return self.count.values()

    def items(self):
        """Return (value, count) pairs"""
        return self.count.items()

    def keys(self):
        """Return all unique values seen"""
        return self.count.keys()

    def get(self, key, default=None):
        """Get count for a specific value"""
        return self.count.get(key, default)
    
    def get_entropy(self):
        if len(self.window) == 0: return 0.0
        if not self.entropy_review and self.hidden_entropy is not None:
            return self.hidden_entropy
        total = len(self.window)
        entropy = 0.0
        for count in self.count.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p + 1e-10)   # add small value to avoid log(0)
        self.hidden_entropy = entropy
        self.entropy_review = False
        return entropy if not np.isnan(entropy) else 0.0
    
    def get_recency(self, value):
        "return how recent was the value seen (1 = rencently; 0 = never or on a long time)"
        if len(self.window) == 0: return 0.0
        last_idx = self.last_seen.get(value)
        if last_idx is None: return 0.0
        return (last_idx+1) / len(self.window)