import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

np.set_printoptions(threshold=np.inf)  # 使得python print没有省略号
np.set_printoptions(suppress=True)  # 使得小数不以科学计数法的方式输出
A = 0.1
alpha = 0.33


class Household(object):
    def __init__(self,
                 e=0.5,  # Frisch elasticity of the labor supply
                 R=1e-8,  # transfer payment
                 chi=0.01,  # constant coefficient of disutility of labor supply
                 tau=0.0001,  # linear tax rate
                 dep=0.01,  # depreciation rate
                 r=0.02,  # interest rate
                 w=0.2,  # wages
                 rho=0.09,  # discount factor
                 gamma=2,  # relative risk aversion
                 sig1=0.2,  # volatility of Brownian Motion of capital
                 a_min=1e-6,  # minimum asset amount
                 a_max=8000,  # maximum asset amount
                 a_size=1000,  # number of asset grid points
                 var=0.07,  # variance of the stationary distribution of log(x)
                 corr=0.9,  # corr[log(x_t+1),log(x_t)]
                 x_size=300,  # number of productivity grid points
                 delta=1000.0):
        # Initialize values, and set up grids over a and z
        self.r, self.w, self.dep = r, w, dep
        self.rho, self.e, self.gamma, self.chi = rho, e, gamma, chi
        self.R, self.tau = R, tau
        self.a_min, self.a_max, self.a_size, self.sig1 = a_min, a_max, a_size, sig1
        self.x_size, self.var, self.corr = x_size, var, corr
        self.x_mean = np.exp(var / 2)
        self.theta = -np.log(corr)
        self.sig2 = 2 * self.theta * self.var  # volatility of Brownian Motion of log productivity
        self.x_min = self.x_mean * 0.001
        self.x_max = self.x_mean * 100
        self.dx = (self.x_max - self.x_min) / (self.x_size - 1)
        self.dx2 = self.dx ** 2
        self.da = (self.a_max - self.a_min) / (self.a_size - 1)
        self.x_vals = np.linspace(self.x_min, self.x_max, self.x_size)
        self.a_vals = np.linspace(self.a_min, self.a_max, self.a_size)  # 创建等差数列

        self.mu = (-self.theta * np.log(self.x_vals) + self.sig2 / 2) * self.x_vals  # Drift (from Ito's formula)
        self.s2 = self.sig2 * self.x_vals ** 2  # Volitility

        self.n = self.a_size * self.x_size
        self.delta = delta

        # The higher boundary coefficient in the wealth dimension
        self.upsilon = -self.gamma * self.sig1 ** 2 * self.a_max / 2
        ###### ADDED TO MATCH LABOR SUPPLY IN .m

        # 转移支付
        self.transpay = self.R * np.ones((self.x_size, self.a_size))

        # Solve labor supply respectively
        self.l = np.power(w * (1 - self.tau) * self.x_vals / self.chi, self.e)

        # 平均劳动（理解stationary时x的分布后再来编。。）
        # 先求x的密度函数 注意到log(x)是期望为0方差为var的正态分布
        self.gx = np.exp(-np.log(self.x_vals) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var) * self.x_vals)
        self.l_ave = np.sum(self.x_vals * self.l * self.gx * self.dx)

        # Initial Guess of Value Function
        # concave and increasing w.r.t wealth
        self.v = np.power(self.transpay + np.tile(self.a_vals, (self.x_size, 1)) * (self.r - self.dep)
                          + self.w * np.tile((1 - self.tau) * self.l, (self.a_size, 1)).transpose() * np.tile(
            self.x_vals, (self.a_size, 1)).transpose()
                          - self.chi * np.power(np.tile(self.l, (self.a_size, 1)).transpose(), 1 + 1 / self.e) / (
                                      1 + 1 / self.e), 1 - self.gamma) \
                 / ((1 - self.gamma) * self.rho)

        # Build skill_transition, the matrix summarizing transitions due to the income shocks
        # This is analogous to the Q matrix in the discrete time version of the QuantEcon Aiyagari model

        # Preallocation
        self.v_old = np.zeros((self.x_size, self.a_size))
        self.g = np.zeros((self.x_size, self.a_size))
        self.dva = np.zeros((self.x_size, self.a_size - 1))
        self.dvx = np.zeros((self.x_size - 1, self.a_size))
        self.s = np.zeros((self.x_size, self.a_size))
        self.cf = np.zeros((self.x_size, self.a_size - 1))
        self.c0 = np.zeros((self.x_size, self.a_size))
        self.u = np.zeros((self.x_size, self.a_size))
        self.ssf = np.zeros((self.x_size, self.a_size))
        self.ssb = np.zeros((self.x_size, self.a_size))
        self.is_forward = np.zeros((self.x_size, self.a_size), 'bool')
        self.is_backward = np.zeros((self.x_size, self.a_size), 'bool')

    def reinitialize_v(self):
        """
        Reinitializes the value function if the value function became NaN
        """
        self.v = np.power(self.transpay + np.tile(self.a_vals, (self.x_size, 1)) * (self.r - self.dep)
                          + self.w * np.tile((1 - self.tau) * self.l, (self.a_size, 1)).transpose() * np.tile(
            self.x_vals, (self.a_size, 1)).transpose()
                          - self.chi * np.power(np.tile(self.l, (self.a_size, 1)).transpose(), 1 + 1 / self.e) / (
                                      1 + 1 / self.e), 1 - self.gamma) \
                 / ((1 - self.gamma) * self.rho)

    def disutility_function(self):
        '''
        Disutility Function
        Return the disutility of the whole economy (x_size*a_size)
        '''
        return self.chi * np.tile(self.l, (self.a_size, 1)).transpose() ** (1 + 1 / self.e) / (1 + 1 / self.e)

    def utility_function(self):
        '''
        Utility Function(to be completed)
        '''
        return (self.c0 - self.disutility_function()) ** (1 - self.gamma) / (1 - self.gamma)

    def solve_foc(self):
        '''
        Solve First Order Condition of HJB Equation
        Return the consumption in this iteration
        '''
        return (self.dva ** (-1 / self.gamma) + self.chi * np.power(np.tile(self.l, (self.a_size - 1, 1)).transpose(),
                                                                    1 + 1 / self.e) / (1 + 1 / self.e))

    def solve_bellman(self, maxiter=100, crit=1e-6):
        """
        This function solves the decision problem with the given parameters

        Parameters:
        -----------------
        maxiter : maximum number of iteration before haulting value function iteration
        crit : convergence metric, stops if value function does not change more than crit
        """
        dist = 100.0
        self.transpay = self.R * np.ones((self.x_size, self.a_size))
        for i in range(maxiter):
            # compute saving and consumption implied by current guess for value function, using upwind method
            self.dva = (self.v[:, 1:] - self.v[:, :-1]) / self.da  # dv/da
            self.cf = self.solve_foc()
            self.c0 = np.tile(self.a_vals, (self.x_size, 1)) * (self.r - self.dep) + self.w * np.tile(
                (1 - self.tau) * self.l, (self.a_size, 1)).transpose() * np.tile(self.x_vals,
                                                                                 (self.a_size, 1)).transpose()

            # computes savings with forward forward difference and backward difference
            self.ssf[:, :-1] = self.c0[:, :-1] - self.cf
            self.ssb[:, 1:] = self.c0[:, 1:] - self.cf
            # Note that the lower boundary conditions in the wealth dimension are handled implicitly as ssf will be zero at a_max and ssb at a_min
            self.is_forward = self.ssf > 0
            self.is_backward = self.ssb < 0
            # Update consumption based on forward or backward difference based on direction of drift
            self.c0[:, :-1] += (self.cf - self.c0[:, :-1]) * self.is_forward[:, :-1]
            self.c0[:, 1:] += (self.cf - self.c0[:, 1:]) * self.is_backward[:, 1:]
            # Now c0 is upwind consumption
            # Saving
            self.s = np.zeros((self.x_size, self.a_size))
            self.s[:, :-1] += self.ssf[:, :-1] * self.is_forward[:, :-1]
            self.s[:, 1:] += self.ssb[:, 1:] * self.is_backward[:, 1:]
            # The utility
            self.u = self.utility_function()
            # self.a_vals_use = np.hstack((0,self.a_vals[1:-1],0))

            # Build the matrix A that summarizes the evolution of the process for (a,x)
            # Matrix system
            # Construct matrix C_1
            self.y = -self.ssf * self.is_forward / self.da + self.ssb * self.is_backward / self.da
            self.x_1 = -self.ssb * self.is_backward / self.da + np.tile(
                (self.sig1 * self.a_vals) ** 2 / (2 * self.da ** 2), (self.x_size, 1))
            self.x_1 = self.x_1[:, 1:-1]
            self.x_0 = np.zeros((self.x_size, 1))
            self.x_2 = -self.ssb * self.is_backward / self.da - self.upsilon / self.da
            self.x_2 = self.x_2[:, -1, None]
            self.x = np.concatenate((self.x_1, self.x_2, self.x_0), axis=1)
            self.z_1 = self.ssf * self.is_forward / self.da + np.tile(
                (self.sig1 * self.a_vals) ** 2 / (2 * self.da ** 2), (self.x_size, 1))
            self.z_1 = self.z_1[:, 1:-1]
            self.z_0 = self.ssf * self.is_forward / self.da
            self.z_0 = self.z_0[:, 0, None]
            self.z_2 = np.zeros((self.x_size, 1))
            self.z = np.concatenate((self.z_2, self.z_0, self.z_1), axis=1)
            self.C_1 = sparse.spdiags(self.y.reshape(self.n), 0, self.n, self.n)
            self.C_1 += sparse.spdiags(self.x.reshape(self.n), -1, self.n, self.n)
            self.C_1 += sparse.spdiags(self.z.reshape(self.n), 1, self.n, self.n)

            # Construct matrix C_2
            self.kappa = -self.mu * (self.mu < 0) / self.dx + self.s2 / (2 * self.dx2)  # \chi_{j}
            self.zeta = self.mu * (self.mu > 0) / self.dx + self.s2 / (2 * self.dx2)  # \zeta_{j}
            self.nu = self.mu * (self.mu < 0) / self.dx - self.mu * (
                        self.mu > 0) / self.dx - self.s2 / self.dx2  # \nu_{j}
            self.nu[0] += self.kappa[0]
            self.nu[-1] += self.zeta[-1]
            self.nu = np.repeat(self.nu, self.a_size)
            self.kappa = np.repeat(self.kappa[1:], self.a_size)
            self.zeta = np.repeat(self.zeta[:-1], self.a_size)
            self.zeta = np.hstack((np.repeat(0, self.a_size), self.zeta))
            self.C_2 = sparse.spdiags(self.nu, 0, self.n, self.n)
            self.C_2 += sparse.spdiags(self.zeta, self.a_size, self.n, self.n)
            self.C_2 += sparse.spdiags(self.kappa, -self.a_size, self.n, self.n)

            # Construct matrix C_3
            self.epsilon_0 = np.zeros((self.x_size, 1))
            self.epsilon_1 = np.tile(-(self.sig1 * self.a_vals / self.da) ** 2, (self.x_size, 1))
            self.epsilon_1 = self.epsilon_1[:, 1:-1]
            self.epsilon_2 = np.full((self.x_size, 1), self.upsilon / self.da)
            self.epsilon = np.concatenate((self.epsilon_0, self.epsilon_1, self.epsilon_2), axis=1)
            self.C_3 = sparse.spdiags(self.epsilon.reshape(self.n), 0, self.n, self.n)

            self.A = self.C_1 + self.C_2 + self.C_3
            # Solve the system of linear equations corresponding to implicit finite difference scheme
            self.B = sparse.eye(self.n) * (1 / self.delta + self.rho) - self.A
            self.b = self.u.reshape(self.n, 1) + self.v.reshape(self.n, 1) / self.delta
            self.v_old = self.v.copy()
            self.v = spsolve(self.B, self.b).reshape(self.x_size, self.a_size)

            # Compute convergence metric and stop if it satisfies the convergence criterion
            dist = np.amax(np.absolute(self.v_old - self.v).reshape(self.n))
            if dist < crit:
                break

    def compute_stationary_distribution(self):
        """
        Solves for the stationary distribution given household decision rules

        Output:
        Capital level from the stationary distribution
        """
        # Matrix system
        # Construct matrix C_1
        self.y = -self.ssf * self.is_forward / self.da + self.ssb * self.is_backward / self.da - np.tile(
            (self.sig1 * self.a_vals) ** 2 / (self.da ** 2), (self.x_size, 1))
        self.x_1 = -self.ssb * self.is_backward / self.da + np.tile((self.sig1 * self.a_vals) ** 2 / (2 * self.da ** 2),
                                                                    (self.x_size, 1))
        self.x_1 = self.x_1[:, 1:]
        self.x_0 = np.zeros((self.x_size, 1))
        self.x = np.concatenate((self.x_1, self.x_0), axis=1)
        self.z_1 = self.ssf * self.is_forward / self.da + np.tile((self.sig1 * self.a_vals) ** 2 / (2 * self.da ** 2),
                                                                  (self.x_size, 1))
        # process the upper boundary of wealth
        self.y[:, -1] = self.y[:, -1] + self.z_1[:, -1]
        self.z_1 = self.z_1[:, :-1]
        self.z_2 = np.zeros((self.x_size, 1))
        self.z = np.concatenate((self.z_2, self.z_1), axis=1)
        self.C_1 = sparse.spdiags(self.y.reshape(self.n), 0, self.n, self.n)
        self.C_1 += sparse.spdiags(self.x.reshape(self.n), -1, self.n, self.n)
        self.C_1 += sparse.spdiags(self.z.reshape(self.n), 1, self.n, self.n)
        self.A = self.C_1 + self.C_2
        self.AT = self.A.transpose().tocsr()

        # The discretized Kolmogorov Forward equation AT*g=0 is an eigenvalue problem
        # AT is singular because one of the equation is the distribution adding
        # up to 1. Here we solve the eigenvalue problem by setting g(1,1)=0.1
        # and the equation is solved relative to that value.
        # Alternatively, one could use a routine for solving eigenvalue problems.
        b = np.zeros((self.n, 1))
        b[0] = 0.1
        self.AT.data[1:self.AT.indptr[1]] = 0
        self.AT.data[0] = 1.0
        self.AT.indices[0] = 0
        self.AT.eliminate_zeros()
        self.g = spsolve(self.AT, b).reshape(self.x_size, self.a_size)

        # Since g was solved taking one of g(1,1) as given, g needs to be
        # renormalized to add up to 1
        self.g = self.g / np.sum(self.g)
        return np.sum(self.g * (np.tile(self.a_vals, (self.x_size, 1))))

    def set_prices(self, r, w):
        """
        Resets prices
        Calling the method will resolves the Bellman Equation.

        Parameters:
        -----------------
        r : Interest rate
        w : wage
        """
        self.r, self.w = r, w
        self.solve_bellman()


# Define Cobb-Douglas Production


def r_to_w(r):
    return A * (1 - alpha) * (alpha * A / r) ** (alpha / (1 - alpha))


def rd(am, K):
    return A * alpha * (am.l_ave / K) ** (1 - alpha)


def prices_to_capital_stock(r0):
    """
    Map prices to the induced level of capital stock.

    Parameters:
    ----------
    r : float
        The interest rate
    """
    w0 = r_to_w(r0)
    am = Household(r=r0, w=w0)
    am.solve_bellman()
    print('completed stage')
    # Compute the stationary distribution and capital
    return am.compute_stationary_distribution()


# Lorenz Curve
def LorenzandGini(wealth, density):#输入wealth列表和密度
    density = density[np.argsort(wealth)]#按wealth大小调整density
    density = np.hstack((0, density))
    wealth = np.sort(wealth)
    wealth = np.hstack((0, wealth))
    wealth_wgt = wealth * density
    wealth_wgt_cum = np.cumsum(wealth_wgt)
    density_cum = np.cumsum(density)
    lorenz = wealth_wgt_cum / np.sum(wealth_wgt)
    Gini = 1 - 2 * np.sum(lorenz * density)
    return lorenz, Gini, density_cum


def percentile(lorenz, start, end, density_cum):
    i = 0
    while density_cum[i] * 100 - start < -1e-10 :
        i += 1
    q1 = lorenz[i]
    while density_cum[i] * 100 - end < -1e-10:
        i += 1
    q2 = lorenz[i]
    q = q2 - q1
    return q
def percentileresult(lorenz, Gini, density_cum):
    q0020 = percentile(lorenz, 0, 20, density_cum)
    q2040 = percentile(lorenz, 20, 40, density_cum)
    q4060 = percentile(lorenz, 40, 60, density_cum)
    q6080 = percentile(lorenz, 60, 80, density_cum)
    q8090 = percentile(lorenz, 80, 90, density_cum)
    q9095 = percentile(lorenz, 90, 95, density_cum)
    q9599 = percentile(lorenz, 95, 99, density_cum)
    q99100 = percentile(lorenz, 99, 100, density_cum)
    #percentileresult = np.round(100*np.array([q0020, q2040, q4060, q6080, q8090, q9095, q9599, q99100, Gini/100]),3)
    percentileresult = np.round(np.array([q0020, q2040, q4060, q6080, q8090, q9095, q9599, q99100, Gini]),5)
    return percentileresult

if __name__ == '__main__':

    print('start version3-2')

    # Supply and Demand curve
    num_points = 10
    r_vals = np.linspace(0.011, 0.015, num_points)
    # Compute supply of capital
    k_vals = np.empty(num_points)
    # Use multiprocessing
    # Init multiprocessing.Pool()
    pool = mp.Pool(processes=3)
    k_vals = pool.map(prices_to_capital_stock, [r for r in r_vals])
    pool.close()
    # init a new household set
    print('k_vals:')
    print(k_vals)
    World_0 = Household()
    World_0.solve_bellman()
    print('complete final stage')
    # Plot supply and demand of capital
    fig_SD, ax_SD = plt.subplots(figsize=(11, 8))
    ax_SD.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
    ax_SD.plot(k_vals, rd(World_0, k_vals), lw=2, alpha=0.6, label='demand for capital')
    ax_SD.grid()
    ax_SD.set_xlabel('capital')
    ax_SD.set_ylabel('interest rate')
    ax_SD.legend(loc='upper right')
    plt.show()

    # Find the equilibrium distribution of assets
    # Set parameters for bisection method
    crit = 1e-6
    r_min = 0.012
    r_max = 0.014
    r = 0.013

    # Bisection loop
    for i in range(100):
        World_0.set_prices(r, r_to_w(r))
        r_new = rd(World_0, World_0.compute_stationary_distribution())
        if np.absolute(r_new - r) < crit:
            break
        elif r_new > r:
            r_min = r
            r = (r_max + r_min) / 2
            print("too small")
        else:
            r_max = r
            r = (r_max + r_min) / 2
            print("too large")
    print("completed equilibrium stage")
    # Plot stationary distribution at the equilibrium
    n1 = 5  # Determine the max productivity level to show in the plot
    n2 = 100 # Determine the max asset level to show in the plot
    x = World_0.x_vals[0:n1]
    y = World_0.a_vals[0:n2]
    X, Y = np.meshgrid(x,y,indexing='ij')
    Z = World_0.g[0:n1, 0:n2]
    fig_eq = plt.figure(figsize=(11, 8))
    ax_eq = fig_eq.add_subplot(111, projection='3d')
    ax_eq.plot_surface(X, Y ,Z , lw=2, alpha=0.6, cmap=cm.jet)
    ax_eq.grid()
    ax_eq.set_xlabel('productivity position')
    ax_eq.set_ylabel('asset position')
    ax_eq.set_zlabel('distribution')
    # Plot equilibrium consumption
    n1 = 10  # Determine the max productivity level to show in the plot
    n2 = 999  # Determine the max asset level to show in the plot
    x = World_0.x_vals[0:n1]
    y = World_0.a_vals[0:n2]
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = World_0.c0[0:n1, 0:n2]
    fig_eq_c = plt.figure(figsize=(11, 8))
    ax_eq_c = fig_eq_c.add_subplot(111, projection='3d')
    ax_eq_c.plot_surface(X, Y, Z, lw=2, alpha=0.6, cmap=cm.jet)
    ax_eq_c.grid()
    ax_eq_c.set_xlabel('productivity position')
    ax_eq_c.set_ylabel('asset position')
    ax_eq_c.set_zlabel('consumption')
    # Plot equilibrium saving
    n1 = 10  # Determine the max productivity level to show in the plot
    n2 = 999  # Determine the max asset level to show in the plot
    x = World_0.x_vals[0:n1]
    y = World_0.a_vals[0:n2]
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = World_0.s[0:n1, 0:n2]
    fig_eq_s = plt.figure(figsize=(11, 8))
    ax_eq_s = fig_eq_s.add_subplot(111, projection='3d')
    ax_eq_s.plot_surface(X, Y, Z, lw=2, alpha=0.6, cmap=cm.jet)
    ax_eq_s.grid()
    ax_eq_s.set_xlabel('productivity position')
    ax_eq_s.set_ylabel('asset position')
    ax_eq_s.set_zlabel('saving')

    fig_eq_s_partial, ax_eq_s_partial = plt.subplots(figsize=(11, 8))
    n = 999  # Determine the max asset level to show in the plot
    ax_eq_s_partial.plot(World_0.a_vals[0:n], World_0.s[0, 0:n], lw=2, alpha=0.6, label='saving 1')
    ax_eq_s_partial.plot(World_0.a_vals[0:n], World_0.s[10, 0:n], lw=2, alpha=0.6, label='saving 2')
    ax_eq_s_partial.plot(World_0.a_vals[0:n], World_0.s[40, 0:n], lw=2, alpha=0.6, label='saving 3')
    ax_eq_s_partial.plot(World_0.a_vals[0:n], World_0.s[60, 0:n], lw=2, alpha=0.6, label='saving 4')
    ax_eq_s_partial.plot(World_0.a_vals[0:n], World_0.s[90, 0:n], lw=2, alpha=0.6, label='saving 5')
    ax_eq_s_partial.grid()
    ax_eq_s_partial.set_xlabel('asset position')
    ax_eq_s_partial.set_ylabel('saving')
    ax_eq_s_partial.legend(loc='upper right')
    plt.show()

    # lorenz of wealth
    wealth = World_0.a_vals
    density = np.sum(World_0.g, axis=0)
    lorenz, Gini, density_cum = LorenzandGini(wealth, density)
    plt.figure
    plt.plot(density_cum, lorenz)
    plt.plot(density_cum, density_cum)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Wealth')
    plt.show()
    print(percentileresult(lorenz, Gini, density_cum))


    # lorenz of income
    income = np.tile(World_0.a_vals, (World_0.x_size, 1)) * (World_0.r - World_0.dep) \
            + World_0.w * np.tile((1 - World_0.tau) * World_0.l, (World_0.a_size, 1)).transpose() \
             * np.tile(World_0.x_vals,(World_0.a_size, 1)).transpose()
    income = income.flatten()
    density = World_0.g.flatten()
    lorenz, Gini, density_cum = LorenzandGini(income, density)
    plt.figure
    plt.plot(density_cum, lorenz)
    plt.plot(density_cum, density_cum)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Income')
    plt.show()
    print(percentileresult(lorenz, Gini, density_cum))