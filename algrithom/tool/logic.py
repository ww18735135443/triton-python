from collections import deque
class WarnLogic:
    def __init__(self,warnThreshold,warnInterval,warnPercent):
        self.warnThreshold=warnThreshold
        self.warnInterval=warnInterval
        self.warnPercent=warnPercent
        self.state=deque()
        self.lastInterval=-warnInterval
    def update(self,curFrameResult,timestamp):
        warnFlag=0
        if curFrameResult==1:
            self.state.appendleft(timestamp)
        else:
            self.state.appendleft(-timestamp)
        while timestamp-abs(self.state[-1])>self.warnThreshold:
            self.state.pop()
        count=sum(i>0 for i in self.state)
        # print("count:\n",count)
        # print("state.size:%d\n", len(self.state))
        if count*1.0/ len(self.state) >= self.warnPercent and self.state[-1]>0 and curFrameResult==1 and len(self.state)>=5:
            # print("count * 1.0 / state.size() >= percent, frameId:%d, lastWarnFrameId: %d, interval:%d\n",
            #             frameId, self.lastInterval, self.warnInterval)
            if timestamp -self.lastInterval>= self.warnInterval and self.warnInterval>0:
                self.lastInterval=timestamp
                warnFlag=1
        # self.state.pop()
        return warnFlag
    def clearState(self):
        self.state.clear()

