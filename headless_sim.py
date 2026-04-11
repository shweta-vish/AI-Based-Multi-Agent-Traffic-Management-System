import random

defaultRed=150; defaultYellow=5; defaultGreen=20
defaultMinimum=10; defaultMaximum=60
noOfSignals=4; simTime=300
carTime=2; bikeTime=1; rickshawTime=2.25; busTime=2.5; truckTime=2.5
noOfLanes=2
speeds={'car':2.25,'bus':1.8,'truck':1.8,'rickshaw':2,'bike':2.5}
stopLines={'right':590,'down':330,'left':800,'up':535}
defaultStop={'right':580,'down':320,'left':810,'up':545}
vehicleSizes={'car':{'w':45,'h':25},'bus':{'w':75,'h':30},'truck':{'w':70,'h':30},'rickshaw':{'w':40,'h':22},'bike':{'w':30,'h':18}}
vehicleTypes={0:'car',1:'bus',2:'truck',3:'rickshaw',4:'bike'}
directionNumbers={0:'right',1:'down',2:'left',3:'up'}
gap=15
SUBSTEPS=60

signals=[]; vehicles={}; x_spawn={}; y_spawn={}
currentGreen=0; nextGreen=1; currentYellow=0; timeElapsed=0

class TrafficSignal:
    def __init__(self,r,y,g,mn,mx):
        self.red=r;self.yellow=y;self.green=g;self.minimum=mn;self.maximum=mx
        self.signalText="30";self.totalGreenTime=0

def reset_globals():
    global signals,currentGreen,nextGreen,currentYellow,timeElapsed,vehicles,x_spawn,y_spawn
    currentGreen=0;nextGreen=1;currentYellow=0;timeElapsed=0
    vehicles={d:{0:[],1:[],2:[],'crossed':0} for d in ['right','down','left','up']}
    x_spawn={'right':[0,0,0],'down':[755,727,697],'left':[1400,1400,1400],'up':[602,627,657]}
    y_spawn={'right':[348,370,398],'down':[0,0,0],'left':[498,466,436],'up':[800,800,800]}
    signals.clear()
    signals.append(TrafficSignal(0,defaultYellow,defaultGreen,defaultMinimum,defaultMaximum))
    signals.append(TrafficSignal(signals[0].yellow+signals[0].green,defaultYellow,defaultGreen,defaultMinimum,defaultMaximum))
    signals.append(TrafficSignal(defaultRed,defaultYellow,defaultGreen,defaultMinimum,defaultMaximum))
    signals.append(TrafficSignal(defaultRed,defaultYellow,defaultGreen,defaultMinimum,defaultMaximum))

def get_queue_lengths():
    return [sum(1 for lane in range(3) for v in vehicles[directionNumbers[i]][lane] if not v['crossed']) for i in range(noOfSignals)]

def _spawn_vehicle():
    vtype_id=random.randint(0,4); vtype=vehicleTypes[vtype_id]
    lane=0 if vtype_id==4 else random.randint(0,1)+1
    r=random.random()
    dir_num=0 if r<0.7 else(1 if r<0.9 else(2 if r<0.97 else 3))
    direction=directionNumbers[dir_num]
    sz=vehicleSizes[vtype]; w=sz['w']; h=sz['h']
    v={'lane':lane,'type':vtype,'speed':speeds[vtype],'direction':direction,'dir_num':dir_num,
       'x':x_spawn[direction][lane],'y':y_spawn[direction][lane],
       'crossed':0,'wait_ticks':0,'width':w,'height':h,'stop':defaultStop[direction]}
    vehicles[direction][lane].append(v)
    if direction=='right': x_spawn[direction][lane]-=(w+gap)
    elif direction=='left': x_spawn[direction][lane]+=(w+gap)
    elif direction=='down': y_spawn[direction][lane]-=(h+gap)
    elif direction=='up': y_spawn[direction][lane]+=(h+gap)

def _gap_ok(v, direction, lane):
    idx=vehicles[direction][lane].index(v)
    if idx==0: return True
    prev=vehicles[direction][lane][idx-1]
    if direction=='right':  return (prev['x']-(v['x']+v['width']))>gap
    elif direction=='left': return (v['x']-(prev['x']+prev['width']))>gap
    elif direction=='down': return (prev['y']-(v['y']+v['height']))>gap
    elif direction=='up':   return (v['y']-(prev['y']+prev['height']))>gap
    return True

def _move_vehicle(v, direction, lane):
    is_green=(v['dir_num']==currentGreen and currentYellow==0)
    spd=v['speed']
    if direction=='right':
        if not v['crossed'] and v['x']+v['width']>stopLines[direction]:
            v['crossed']=1; vehicles[direction]['crossed']+=1
        if (v['crossed'] or is_green or v['x']+v['width']<v['stop']) and _gap_ok(v,direction,lane):
            v['x']+=spd
        else: v['wait_ticks']+=1
    elif direction=='left':
        if not v['crossed'] and v['x']<stopLines[direction]:
            v['crossed']=1; vehicles[direction]['crossed']+=1
        if (v['crossed'] or is_green or v['x']>v['stop']) and _gap_ok(v,direction,lane):
            v['x']-=spd
        else: v['wait_ticks']+=1
    elif direction=='down':
        if not v['crossed'] and v['y']+v['height']>stopLines[direction]:
            v['crossed']=1; vehicles[direction]['crossed']+=1
        if (v['crossed'] or is_green or v['y']+v['height']<v['stop']) and _gap_ok(v,direction,lane):
            v['y']+=spd
        else: v['wait_ticks']+=1
    elif direction=='up':
        if not v['crossed'] and v['y']<stopLines[direction]:
            v['crossed']=1; vehicles[direction]['crossed']+=1
        if (v['crossed'] or is_green or v['y']>v['stop']) and _gap_ok(v,direction,lane):
            v['y']-=spd
        else: v['wait_ticks']+=1

def _purge_exited():
    for direction in directionNumbers.values():
        for lane in range(3):
            b=vehicles[direction][lane]
            if direction=='right': vehicles[direction][lane]=[v for v in b if v['x']<1450]
            elif direction=='left': vehicles[direction][lane]=[v for v in b if v['x']>-100]
            elif direction=='down': vehicles[direction][lane]=[v for v in b if v['y']<850]
            elif direction=='up': vehicles[direction][lane]=[v for v in b if v['y']>-50]

    


def step_tick(action_green_index):

    global currentGreen,currentYellow,timeElapsed
    currentGreen=action_green_index; currentYellow=0
    # spawn ~1.33 vehicles/sec total
    n_spawn = max(1, int(random.gauss(2.0, 0.6)))
    for _ in range(n_spawn): _spawn_vehicle()
    # move every vehicle at full pixel speed for 60 sub-steps
    for _ in range(SUBSTEPS):
        for direction in directionNumbers.values():
            for lane in range(3):
                for v in list(vehicles[direction][lane]):
                    _move_vehicle(v,direction,lane)
    _purge_exited()
    timeElapsed+=1
    queue=get_queue_lengths()
    total_wait=sum(queue)
    done=(timeElapsed>=simTime)
    return queue,total_wait,done
