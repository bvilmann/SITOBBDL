import numpy as np
from scipy.integrate import ode

class Solver():
    def __init__(self,
                 A,
                 B,
                 solver = 'dopri5', # dopri5 -> Runge-Kutta
                 xlabels=None,
                 ):
        
        self.N = A.shape[0]
        self.A = A
        self.B = B

        # Prepare solver
        self.r = ode(self.dxdt).set_integrator(solver)
        self.r = ode(lambda t, x: self.dxdt(t, x)).set_integrator(solver) # , method='bdf'

        return

    def dxdt(self,t,x,u=None):

        # Reshaping state vector
        # x = self.x.reshape(self.N,1)

        # u = 1*np.sin(2*np.pi*50*t)

        # Input vector and matrix

        # Reshaping state vector
        x = x.reshape((self.N, 1))

        # Calculate system of ODEs
        dx = self.A @ x + self.B * 1

        return dx

    def run_dynamic_simulation(self,
                               x0: np.array,
                               t0: float,
                               t1: float,
                               dt = 0.001):
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt

        # Prepare time series and time series of state vectors
        self.t = t = np.arange(t0, t1, dt) # Create evenly spaced time array t
        self.x = x = np.empty((len(self.x0) ,len(t))) # the inital values as the first element

        # apply the inital conditions
        self.r.set_initial_value(self.x0 , t0) # .set_f_params(param)

        # Set first state as the initial state conditions
        self.x[:,0] = x[:,0] = self.x0
        i = 1 # set the loop counter to 1 and start the integration loop:
        while self.r.successful() and i < len(t) :
            # Set event if t > 0 

            # Integrate                
            self.x[:,i] = x[:, i] = self.r.integrate(t[i])
            
            # Increment counter
            i +=1

        return t, x

    def dq02abc(self,v= None):
        # construct Clark-park transformation matrix container for t indices
        self.dq0 = dq0 = np.zeros((3,3,len(self.t)))
        for i, t in enumerate(self.t):
            self.dq0[:,:,i] = dq0[:,:,i] = \
                2/3*np.array([[np.cos(lp.omega_0*t),    np.cos(lp.omega_0*t-2*np.pi/3),     np.cos(lp.omega_0*t+2*np.pi/3)],
                              [-np.sin(lp.omega_0*t), - np.sin(lp.omega_0*t-2*np.pi/3), -   np.sin(lp.omega_0*t+2*np.pi/3)],
                              [1/2,                     1/2,                                1/2]])

        # Get currents from the states
        self.get_currents(self.x)
        
        # populate the assumed vector
        self.v = v = np.zeros((3,self.i_d_v.shape[1]))
        v[0,:] = self.i_d_v[0,:]
        v[1,:] = self.i_q_v[0,:]

        # Converting to phase current
        self.i_abc = abc = np.zeros((3,6000))
        for i, t in enumerate(self.t):
            self.i_abc[:,i] = abc[:,i] = np.linalg.inv(dq0[:,:,i]) @ v[:,i]

                        
        return abc

    def get_currents(self,x):
        # Sorting states into direct and quadrature axes
        xd,xq = x[[0,2,3]],x[[1,4]]

        # Finding currents in direct axis
        self.i_d,self.i_fd,self.i_kd = self.i_d_v = np.linalg.inv(self.lp.LD_mat) @ xd

        # Finding currents in quadrature axis
        self.i_q,self.i_kq = self.i_q_v = np.linalg.inv(self.lp.LQ_mat) @ xq
                    
        return


    
        
        
        