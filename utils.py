"""Utility classes

Some utility functions or classes that are irrelavent to this project
Contains::
Class Timer: a class estimating remain training time
Usage::
>>> timer = Timer(total_steps=1000)
>>> print(timer)
0m:5s
>>> timer.step(500)
>>> print(timer.remain())
0m:6s (- 0m:6s)
>>> print(timer.remain(percent=0.1))
0m:48s (- 7m:20s)
>>> Timer.Start()
>>> Timer.Now(total_steps=100)
'0m:18s'
>>> Timer.Remain(40)
'0m:21s (- 0m:32s)'
>>> Timer.Stop()
"""
import time
import math

def sec2Hours(s):
    """Converts second into string of form 'h:m:s'"""
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    if h > 0:
        return '%dh:%dm:%ds' % (h, m, s)
    else:
        return '%dm:%ds' % (m, s)


class Timer:
    """A timer class"""
    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.start= time.time()
        self.steps = 0
    
    def __str__(self):
        now = time.time()
        s = now - self.start
        return sec2Hours(s)

    def step(self, delta=1):
        self.steps += delta
    
    def remain(self, step=-1, percent=-1):
        if step < 0:
            step = self.steps
        if percent < 0:
            percent = step / self.total_steps
        
        time_pass = time.time() - self.start
        if percent == 0:
            return "%s (-)" % (sec2Hours(time_pass))
        else:
            estimate = time_pass / percent
            time_rem = estimate - time_pass
            return "%s (- %s)" % (sec2Hours(time_pass), sec2Hours(time_rem))
    
    running_timer = None

    @classmethod
    def Start(cls, total_steps=100):
        cls.running_timer = Timer(total_steps)
    
    @classmethod
    def Step(cls, delta=1):
        if not cls.running_timer:
            cls.Start()
        cls.running_timer.step(delta)
    
    @classmethod
    def Now(cls):
        if not cls.running_timer:
            cls.Start()
        return str(cls.running_timer)
    
    @classmethod
    def Remain(cls, step=-1, percent=-1):
        if not cls.running_timer:
            cls.Start()
        return cls.running_timer.remain(step, percent)
    
    @classmethod
    def Stop(cls):
        cls.running_timer = None