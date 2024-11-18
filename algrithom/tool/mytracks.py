

class Tracker:
    def __init__(self,detection):
        self.current_xyxy=detection['xyxy']
        self.track_id=detection['track_id']
        self.classes=detection['cls']
        self.conf=detection['conf']
        self.alarm_time=None
        self.alarm_state=None
        self.last_xyxy=detection['xyxy']
        self.age=0
    def update(self,detection):
        self.last_xyxy =self.current_xyxy
        self.current_xyxy=detection['xyxy']
        self.age=0

class Tracks:
    def __init__(self):
        self.tracks=[]
        self.track_id=[]
    def update(self,detections):
        self.track_id = [track.track_id for track in self.tracks]
        for detection in detections:
            if detection['track_id'] not in self.track_id:
                self.tracks.append(Tracker(detection))
            else:
                for track in self.tracks:
                    if track.track_id==detection['track_id']:
                        track.update(detection)
        for track in self.tracks:
            track.age+=1
        tracks=[track for track in self.tracks if track.age<=60]
        self.tracks=tracks