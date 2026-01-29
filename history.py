from collections import deque, Counter
import numpy as np

class History:
    def __init__(self):
        self.max_size = 500000
        self.window = deque(maxlen=self.max_size)    # history keeps all the previous values of data entries, for an certain data value of the socket
        self.count = Counter()
        
    def updat_h(self, entry):
        if len(self.window) == self.max_size:
            old = self.window.popleft()
            self.count[old] -= 1
            if self.count[old] == 0 :
                del self.count[old]
        self.window.append(entry)
        self.count[entry] += 1
        return

    def avg_rate(self, data_input):
        if len(self.window) == 0: return 0.0
        total = len(self.window)
        count = self.count.get(data_input, 0)
        
        if count == 0: return 0.0
        
        rate = count / total
        log_rate = np.log(rate+10e-6)   # add small value to avoid log(0)
        
        normalized = (log_rate + 6) / 6
        return np.clip(normalized, 0.0, 1.0)
    
    def get_entropy(self):
        if len(self.window) == 0: return 0.0
        total = len(self.window)
        entropy = 0.0
        for count in self.count.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p + 1e-10)   # add small value to avoid log(0)
        return float(entropy) if not np.isnan(entropy) else 0.0
    
    def get_recency(self, value):
        "return how recent was the value seen (1 = rencently; 0 = never or on a long time)"
        if len(self.window) == 0: return 0.0
        try:
            for i in range(len(self.window)-1, 0, -1):
                if self.window[i] == value:
                    recently = (i+1) / len(self.window)
                    return recently
            return 0.0
        except:
            return 0.0