import numpy as np

DEBUG_PID = False

class PID:
    """
    PID controller class.

    Inputs:
        setpoint: desired value
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        offset: offset value (default = 0.0)
    """
    def __init__(self, agent, Kp, Ki, Kd, dim, offset= 0.0, llim = -25, ulim = 25, debug=False):
        self.agent = agent
        self.dim = dim
        self.Kp_initial = Kp
        self.Ki_initial = Ki
        self.Kd_initial = Kd
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(self.dim)
        self.time_prev = 0.0
        self.e_prev = np.zeros(self.dim)
        self.offset = offset
        self.init = False
        self.debug = debug
        self.llim_initial = llim
        self.ulim_initial = ulim
        self.llim = llim
        self.ulim = ulim

    def update(self, setpoint, measurement, time):
        # PID calculations
        e = np.zeros(len(setpoint))
        e = setpoint[:] - measurement[:]
        # print("%s e: " % self.agent, e)
        P = self.Kp*e
        if self.init == False:
            self.time_prev = time
            self.init = True
            D = 0.0
        else: D = self.Kd*(e - self.e_prev)/(time - self.time_prev)
        delta_I = self.Ki*e*(time - self.time_prev)
        if DEBUG_PID or self.debug:
            print("############# Update %s PID #################" % self.agent)
            print("time - self.time_prev: ", time - self.time_prev)
            print("e: ", e)
        self.integral += delta_I

        # Velocity Damper
        D = self.Damper(D)

        # Calculate Manipulated Variable - MV 
        if DEBUG_PID or self.debug:
            print("P: ", P)
            print("I: ", self.integral)
            print("D: ", D)
        MV = self.offset + P + self.integral + D

        # update stored data for next iteration
        self.e_prev = e
        self.time_prev = time
        return MV

    def reset(self, k=1.0):
        self.init = False
        self.integral = np.zeros(self.dim)
        self.time_prev = 0.0
        self.e_prev = np.zeros(self.dim)
        # Reload the Params
        self.Kp = k*self.Kp_initial
        self.llim = k*self.llim_initial
        self.ulim = k*self.ulim_initial

    def Damper(self, val_array):
        return np.clip(val_array, self.llim, self.ulim)

class IncremPID:
    """
    Incremental PID controller class.

    Inputs:
        setpoint: desired value
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        offset: offset value (default = 0.0)
    """
    def __init__(self, Kp, Ki, Kd, dim, offset= 0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.e_prev = np.zeros(dim)
        self.e_prev2 = np.zeros(dim)
        self.offset = offset

    def update(self, setpoint, measurement):
        # PID calculations
        e = np.zeros(len(setpoint))
        e = setpoint[:] - measurement[:]
        
        P = self.Kp*(e-self.e_prev)
        I = self.Ki*e
        D = self.Kd*(e - 2*self.e_prev + self.e_prev2)

        # calculate manipulated variable - MV 
        MV = self.offset + P + I + D
        if DEBUG_PID:
            print("P: ", P)
            print("I: ", I)
            print("D: ", D)
        # update stored data for next iteration
        self.e_prev2 = self.e_prev
        self.e_prev = e

        return MV