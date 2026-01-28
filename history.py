from collections import deque, Counter

class History:
    def __init__(self):
        self.max_size = 50000
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
        total = len(self.window)
        rate = self.count[data_input] / total
        return rate