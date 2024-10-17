"""
Implement the Inverse Kinematics (IK) for the XArm6.

Reference Link: https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
                https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
                https://github.com/petercorke/robotics-toolbox-python
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import mujoco
import numpy as np
from abc import ABC, abstractmethod
import time
import qpsolvers as qp
from functools import wraps
from utils.util import calculate_arm_Te, angle_axis_python

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

DEBUG_IK = False

class IK(ABC):
    """
    An abstract super class which provides basic functionality to perform numerical inverse
    kinematics (IK). Superclasses can inherit this class and implement the solve method.
    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 100,
        # slimit: int = 100,
        tol: float = 2e-3,
        we: np.ndarray = np.ones(6),
        # problems: int = 1000,
        reject_jl: bool = True,
        ps: float=0.1,
        λΣ: float=0.0,
        λm: float=0.0, 
        copy: bool = False,
        debug: bool = False
    ):
        """
        name: The name of the IK algorithm
        ilimit: How many iterations are allowed within a search before a new search is started
        # slimit: How many searches are allowed before being deemed unsuccessful
        tol: Maximum allowed residual error E
        we: A 6 vector which assigns weights to Cartesian degrees-of-freedom
        # problems: Total number of IK problems within the experiment
        reject_jl: Reject solutions with joint limit violations
        ps: The minimum angle/distance (in radians or metres) in which the joint is allowed to approach to its limit
        λΣ: The gain for joint limit avoidance. Setting to 0.0 will remove this completely from the solution
        λm: The gain for maximisation. Setting to 0.0 will remove this completely from the solution (always 0.0 for now)
        """

        # Solver parameters
        self.name = name
        # self.slimit = slimit
        self.ilimit = ilimit
        self.tol = tol
        self.We = np.diag(we)
        self.reject_jl = reject_jl
        self.λΣ = λΣ
        self.λm = λm
        self.ps = ps
        self.copy = copy
        self.debug = debug

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q0: np.ndarray):
        """
        This method will attempt to solve the IK problem and obtain joint coordinates
        which result the the end-effector pose Tep.

        The method returns a tuple:
        q: The joint coordinates of the solution (ndarray). Note that these will not
            be valid if failed to find a solution
        success: True if a solution was found (boolean)
        iterations: The number of iterations it took to find the solution (int)
        # searches: The number of searches it took to find the solution (int)
        residual: The residual error of the solution (float)
        jl_valid: True if joint coordinates q are within the joint limits
        total_t: The total time spent within the step method
        """

        # Iteration count
        i = 0
        total_i = 0
        total_t = 0.0
        q = np.zeros(model.nv)
        q[:] = q0[:]
        error = -1
        q_solved = np.zeros(model.nv)
        # print("initial q: ", q)
        # print("initial q0: ", q0)
        while i <= self.ilimit:
            i += 1

            # Attempt a step
            try:
                t, E, q = self.step(model, data, Tep, q, i)
                error = E
                q_solved[:] = q[:]

                # Acclumulate total time
                total_t += t
            except np.linalg.LinAlgError:
                # Abandon search and try again
                print("break LinAlgError")
                break

            # Check if we have arrived
            if E < self.tol:
                # Wrap q to be within +- 180 deg
                # If your robot has larger than 180 deg range on a joint
                # this line should be modified in incorporate the extra range
                # q = (q + np.pi) % (2 * np.pi) - np.pi

                # Check if we have violated joint limits
                jl_valid = self.check_jl(model, q)

                if not jl_valid and self.reject_jl:
                    # Abandon search and try again
                    # print("break limits!!!!!!!!!!!!!!!!!!!!!")
                    if DEBUG_IK or self.debug: print("break limits!!!!!!!!!!!!!!!!!!!!!")
                    continue
                else:
                    if DEBUG_IK or self.debug: 
                        print("q_solved: {}, error: {}".format(q_solved, error))
                        print("iteration: {}, total_t: {}".format(i, total_t))
                        print("solved ik!! \n")
                    return q, True, total_i + i, E, jl_valid, total_t

        # Note: If we make it here, then we have failed because of the iteration limit or the joint limits
        # print("q_solved: {}, error: {}".format(q_solved, error))
        # print("iteration: {}, total_t: {}".format(i, total_t))
        # print("failed ik!! \n")
        if DEBUG_IK or self.debug: 
            print("q_solved: {}, error: {}".format(q_solved, error))
            print("iteration: {}, total_t: {}".format(i, total_t))
            print("failed ik!! \n")
        # Return the initial joint position (not the last solution we get)
        # data.qpos[:] = q0[:]
        return q, False, np.nan, E, np.nan, np.nan

    def error(self, Te: np.ndarray, Tep: np.ndarray):
        """
        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

        Returns a tuple:
        e: angle-axis error (ndarray in R^6)
        E: The quadratic error weighted by We
        """
        # e = rtb.angle_axis(Te, Tep)
        e = angle_axis_python(Te, Tep)
        E = 0.5 * e @ self.We @ e

        return e, E

    def check_jl(self, model: mujoco.MjModel, q: np.ndarray):
        """
        Checks if the joints are within their respective limits

        Returns a True if joints within feasible limits otherwise False
        """

        # Loop through the joints in the ETS
        for i in range(model.nv):

            # Get the corresponding joint limits
            ql0 = model.joint(i).range[0]
            ql1 = model.joint(i).range[1]
            # print("ql0: ", ql0)
            # print("ql1: ", ql1)

            # Check if q exceeds the limits
            if q[i] < ql0 or q[i] > ql1:
                # print("i: ", i)
                # print("q[i]: ", q[i])
                # print("ql0: ", ql0)
                # print("ql1: ", ql1)
                return False

        # If we make it here, all the joints are fine
        return True

    @abstractmethod
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray):
        """
        Superclasses will implement this method to perform a step of the implemented
        IK algorithm
        """
        pass

def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        t_start = time.time()
        E, q = func(*args, **kw)
        t_end = time.time()
        t = t_end - t_start
        return t, E, q
    return wrap

class QP(IK):
    def __init__(self, name="QP", λj=1.0, λs=1.0, **kwargs):
        super().__init__(name, **kwargs)

        self.name = f"QP (λj={λj}, λs={λs})"
        self.λj = λj
        self.λs = λs

        if self.λΣ > 0.0:
            self.name += ' Σ'

        if self.λm > 0.0:
            self.name += ' Jm'
        # print("self.ilimit: ", self.ilimit)

    @timing
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray, i: int):
        # Calculate forward kinematics (Te)
        # mujoco.mj_resetData(model, data)
        data.qpos[:] = q[:]
        # Do not use mj_kinematics, it does more than foward the position kinematics!
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(model, data)
        Te = calculate_arm_Te(data.body("link6").xpos, data.body("link6").xquat)
        # print("Tep: ", Tep)
        # print("Te: ", Te)
        # exit(1)
        # Calculate the error
        e, E = self.error(Te, Tep)
        if E < self.tol and i <= 1:
            # print("NO NEED to calculate IK!!!!!!!!!!!!!")
            # data.qpos[:] = q[:]
            return E, q
        
        # Calculate the Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("link6").id)
        J = np.concatenate((jacp, jacr), axis=0)
        # print("J: \n", J)
        # Quadratic component of objective function
        Q = np.eye(model.nv + 6)

        # Joint velocity component of Q
        Q[: model.nv, : model.nv] *= self.λj

        # Slack component of Q
        Q[model.nv :, model.nv :] = self.λs * (1 / np.sum(np.abs(e))) * np.eye(6)

        # The equality contraints
        Aeq = np.concatenate((J, np.eye(6)), axis=1)
        beq = 2*e.reshape((6,))

        # The inequality constraints for joint limit avoidance
        if self.λΣ > 0.0:
            Ain = np.zeros((model.nv + 6, model.nv + 6))
            bin = np.zeros(model.nv + 6)

            # Form the joint limit velocity damper
            Ain_l = np.zeros((model.nv, model.nv))
            Bin_l = np.zeros(model.nv)

            for i in range(model.nv):
                ql0 = model.joint(i).range[0]
                ql1 = model.joint(i).range[1]
                # Calculate the influence angle/distance (in radians or metres) in null space motion becomes active
                pi = (model.joint(i).range[1] - model.joint(i).range[0])/2

                if ql1 - q[i] <= pi:
                    Bin_l[i] = ((ql1 - q[i]) - self.ps) / (pi - self.ps)
                    Ain_l[i, i] = 1

                if q[i] - ql0 <= pi:
                    Bin_l[i] = -(((ql0 - q[i]) + self.ps) / (pi - self.ps))
                    Ain_l[i, i] = -1

            Ain[: model.nv, : model.nv] = Ain_l
            bin[: model.nv] =  (1.0 / self.λΣ) * Bin_l
        else:
            Ain = None
            bin = None
        
        # TODO: Manipulability maximisation
        # if self.λm > 0.0:
        #     Jm = ets.jacobm(q).reshape((model.nv,))
        #     c = np.concatenate(((1.0 / self.λm) * -Jm, np.zeros(6)))
        # else:
        #     c = np.zeros(model.nv + 6)
        c = np.zeros(model.nv + 6)
            
        # print("Q: ", Q)
        # print("c: ", c)
        # print("Ain: ", Ain)
        # print("bin: ", bin)
        # print("Aeq: ", Aeq)
        # print("beq: ", beq)
        xd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=None, ub=None, solver='quadprog')
        # print("xd: ", xd)
        # print("xd: ", xd[: 6])
        q += xd[: model.nv]
        # print("q: ", q)
        # data.qpos[:] = q[:]
        return E, q

def null_Σ(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, ps: float):
    """
    Formulates a relationship between joint limits and the joint velocity.
    When this is projected into the null-space of the differential kinematics
    to attempt to avoid exceeding joint limits

    q: The joint coordinates of the robot
    ps: The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi: The influence angle/distance (in radians or metres) in which the velocity
        damper becomes active

    returns: Σ 
    """

    # Add cost to going in the direction of joint limits, if they are within
    # the influence distance
    Σ = np.zeros((model.nv, 1))

    for i in range(model.nv):
        qi = q[i]
        ql0 = model.joint(i).range[0]
        ql1 = model.joint(i).range[1]
        pi = (model.joint(i).range[1] - model.joint(i).range[0])/2

        if qi - ql0 <= pi:
            Σ[i, 0] = (
                -np.power(((qi - ql0) - pi), 2) / np.power((ps - pi), 2)
            )
        if ql1 - qi <= pi:
            Σ[i, 0] = (
                np.power(((ql1 - qi) - pi), 2) / np.power((ps - pi), 2)
            )

    return -Σ

def calc_qnull(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q: np.ndarray,
        J: np.ndarray,
        λΣ: float,
        # λm: float,
        ps: float,
    ):
    """
    Calculates the desired null-space motion according to the gains λΣ and λm.
    This is a helper method that is used within the `step` method of an IK solver

    Returns qnull: the desired null-space motion
    """

    qnull_grad = np.zeros(model.nv)
    qnull = np.zeros(model.nv)

    # Add the joint limit avoidance if the gain is above 0
    if λΣ > 0:
        Σ = null_Σ(model, data, q, ps)
        qnull_grad += (1.0 / λΣ * Σ).flatten()

    # TODO: Add the manipulability maximisation if the gain is above 0
    # if λm > 0:
    #     Jm = ets.jacobm(q)
    #     qnull_grad += (1.0 / λm * Jm).flatten()

    # Calculate the null-space motion
    if λΣ > 0.0:
        null_space = (np.eye(model.nv) - np.linalg.pinv(J) @ J)
        qnull = null_space @ qnull_grad

    return qnull.flatten()

class LM_Chan(IK):
    def __init__(self, λ=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.name = f"LM (Chan λ={λ})"
        self.λ = λ

        if self.λΣ > 0.0:
            self.name += ' Σ'

        if self.λm > 0.0:
            self.name += ' Jm'

    @timing
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray):
        # Calculate forward kinematics (Te)
        mujoco.mj_resetData(model, data)
        data.qpos = q
        # Do not use mj_kinematics, it does more than foward the position kinematics!
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(model, data)
        Te = np.eye(4)
        Te[:3,3] = data.body("link6").xpos
        res = np.zeros(9)
        mujoco.mju_quat2Mat(res, data.body("link6").xquat)
        Te[:3,:3] = res.reshape((3,3))
        # print(Te)

        # Calculate the error
        e, E = self.error(Te, Tep)
        # Calculate the Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("link6").id)
        J = np.concatenate((jacp, jacr), axis=0)
        g = J.T @ self.We @ e

        Wn = self.λ * E * np.eye(model.nv)

        # Null-space motion
        qnull = calc_qnull(model, data, q, J, self.λΣ, self.ps)
        print("qnull: ", qnull)
        q += np.linalg.inv(J.T @ self.We @ J + Wn) @ g + qnull

        return E, q

class IKArm:
    def __init__(self, solver_type='QP', ps=0.001, λΣ=10, λj=0.1, λs=1.0, λ=0.1, tol=2e-3, ilimit=100):
        if solver_type=='QP':
            self.solver = QP(λj=λj, λs=λs, ps=ps, λΣ=λΣ, tol=tol, ilimit=ilimit)
        elif solver_type=='LM_Chan':
            self.solver = LM_Chan(λ=λ, ps=ps, λΣ=λΣ, tol=tol, ilimit=ilimit)
        else:
            raise ValueError("Invalid solver type")
    
    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q0: np.ndarray):
        self.q0 = np.zeros(model.nv)
        self.q0[:] = q0[:]
        # print("before self.q0: ", self.q0)
        result_IK = self.solver.solve(model, data, Tep, q0)
        # print("after self.q0: ", self.q0)
        if not result_IK[1]:
            # print("Failed result_IK: ", result_IK)
            return self.q0, result_IK[1], result_IK[2], result_IK[3], result_IK[4], result_IK[5]
        return result_IK
