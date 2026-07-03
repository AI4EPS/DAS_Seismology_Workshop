# Module containing the definition of inverse problems where the ADMM method is used
import pyOperator as pyOp
import pyVector as pyVec
from pyLinearSolver import LCGsolver, LSQRsolver
from pySparseSolver import ISTAsolver
from pyProblem import Problem, ProblemL1Lasso, ProblemL2LinearReg, ProblemL2Linear, ProblemLinearReg
from pySolver import Solver
from pySparseSolver import _soft_thresh
from pyStopper import BasicStopper
from sys_util import logger as logger_class
import numpy as np
from math import isnan
import matplotlib.pyplot as plt

# Check for avoid Overflow or Underflow
zero = 10 ** (np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)


class ADMMsolver(Solver):
    """Alternate Directions of Multipliers Method (ADMM)"""
    
    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_linear=5, linear_solver='CG',
                 rho=None, rho_update=True, rho_tol=10., rho_scale=2., rho_save=False,
                 warm_start=False, tol_abs=1e-3, tol_rel=1e-4):
        """
        Constructor of ADMM Solver for Generalized Lasso [boyd2010distributed 6.4.1]
        .. math ::
            1/2 |Op x - d|_2^2 + Σ_i εL2_i |R2_i x - dr|_2^2 + λ |z|_1
                
                subject to Ax - z = 0
                where A = R1 / εL1, λ = max(εL1)
                
        :param stopper          : stopper object
        :param logger           : logger object
        :param niter_linear     : int; number of iterations for solving the linear problem [5]
        :param rho              : float; penalty parameter rho (if None it is initialized as 2*gamma+.1)
        :param rho_update       : bool; update rho automatically
        :param rho_tol          : float; norm ratio between residuals for updating rho [10]
        :param rho_scale        : float; scaling factor for updating rho [2]
        :param warm_start       : bool; linear solver uses previous solution [False]
        :param linear_solver    : str; linear solver to be used [CG, SD, LSQR]
        :param rho_save         : bool; save rho value through the iterations [False]
        :param tol_abs          : float; tolerance for absolute eps in stopping criterion [1e-3]
        :param tol_rel          : float; tolerance for relative eps in stopping criterion [1e-4]
        """
        # Calling parent construction
        super(ADMMsolver, self).__init__()
        
        # Defining stopper object
        self.stopper = stopper
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # Logger for internal linear solver
        self.niter_linear = niter_linear

        self.warm_start = warm_start  # see boyd2010distributed 4.3.3
        # Stop criterion
        self.tol_abs = abs(tol_abs)
        self.tol_rel = abs(tol_rel)
        
        # Logger for internal linear solver
        self.logger_lin_solv = None
        if logger is not None:
            if "/" in logger.file.name:
                folder = "/".join(logger.file.name.split("/")[:-1]) + "/"
            else:
                folder = ""
            filename = "lin_inv_" + logger.file.name.split("/")[-1]
            self.logger_lin_solv = logger_class(folder + filename)
        
        if linear_solver == 'CG':
            self.linear_solver = LCGsolver(BasicStopper(niter=self.niter_linear),
                                           steepest=False, logger=self.logger_lin_solv)
        elif linear_solver == 'SD':
            self.linear_solver = LCGsolver(BasicStopper(niter=self.niter_linear),
                                           steepest=True, logger=self.logger_lin_solv)
        elif linear_solver == 'LSQR':
            self.linear_solver = LSQRsolver(BasicStopper(niter=self.niter_linear),
                                            logger=self.logger_lin_solv)
        else:
            raise ValueError('ERROR! Solver has to be CG, SD or LSQR')
        
        self.rho = rho  # ADMM penalty parameter
        self.mu = rho_tol  # norm ratio between residuals for updating rho (μ in boyd2010distributed)
        self.tau = rho_scale  # scaling factor for updating rho (τ in boyd2010distributed)
        self.rho_update = rho_update  # whether to update rho
        self.primal = None  # primal residual vector (r = A x + B z - c)
        self.dual = None  # dual residual vector   (s = rho A.H B r)
        self.rho_history = [] if rho_save and rho_update else None
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"
    
    def _update_rho(self):
        """update penalty parameter rho as suggested in boyd2010distributed (3.13)"""
        if self.rho_history is not None:  # save rho value
            self.rho_history.append(self.rho)
        
        if self.primal.norm(2) > self.mu * self.dual.norm(2):
            self.rho *= self.tau
        elif self.dual.norm(2) > self.mu * self.primal.norm(2):
            self.rho /= self.tau
        else:
            pass
    
    def _init_rho(self, lamda):
        # considering f(x) + λ g(z)
        self.rho = 2 * lamda + .1
    
    def run(self, problem, verbose=False, inner_verbose=False, restart=False):
        
        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.nregsL1 == 0:
            raise ValueError('ERROR! Provide at least one L1 regularizer!')
        
        self.create_msg = verbose or self.logger
        # overriding save_grad variable
        self.save_grad = False
        # reset stopper before running the inversion
        self.stopper.reset()
        
        # initialization
        admm_mdl = problem.model.clone().zero()
        # max L1reg weight (for both rho and sparse problem setup)
        lamda = max(problem.epsL1)
        # A is the Vstack of the L1 reg operators, scaled by their respective eps
        # we also divide by gamma as gamma becomes the lambda value for the FISTA problem
        A = problem.regL1_op * [eps / lamda for eps in problem.epsL1]
        # Constraint variable
        z = A.range.clone().zero()
        # Scaled dual variable
        u = A.range.clone().zero()
        # primal residual for constraint: r = Ax - z
        self.primal = A.range.clone().zero()
        # dual residual for optimality:   s = A.H * r
        self.dual = A.domain.clone().zero()
        # other vectors
        z_minus_u = A.range.clone().zero()
        Ax = A.range.clone().zero()
        Ax_plus_u = A.range.clone().zero()
        
        # init rho
        if self.rho is None:
            self._init_rho(lamda)
        
        # Linear Problem
        # 1/2 | Op x -  d   |_2^2
        # εL2 | R2 x -  dr  |_2^2
        # ρ/2 | A  x - (z-u)|_2^2
        if problem.nregsL2 != 0:
            opL2 = pyOp.Vstack(problem.op, np.sqrt(problem.epsL2) * problem.regL2_op)
            dataL2 = pyVec.superVector(problem.data, problem.dataregsL2)
        else:
            opL2 = problem.op
            dataL2 = problem.data
        
        linear_problem = ProblemL2Linear(model=admm_mdl.clone(),
                                         data=pyVec.superVector(dataL2, z_minus_u),  # WATCH OUT! data.vecs[-1] must be updated
                                         op=pyOp.Vstack(opL2, self.rho * A),  # WATCH OUT! op.ops[-1].const must be updated with self.rho
                                         minBound=problem.minBound,
                                         maxBound=problem.maxBound,
                                         boundProj=problem.boundProj)
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            admm_mdl = self.restart.retrieve_vector("admm_mdl")
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
        else:
            outer_iter = 0
            if self.create_msg:
                msg = 90 * '#' + '\n'
                msg += "\tALTERNATING DIRECTION METHOD OF MULTIPLIERS ALGORITHM log file\n\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                if problem.nregsL2 != 0:
                    msg += "\tL2 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL2_op.ops]) + "\n"
                    msg += "\tL2 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL2]) + "\n"
                msg += "\tL1 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL1_op.ops]) + "\n"
                msg += "\tL1 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL1]) + "\n"
                msg += "\tPenalty parameter:\t\t%.2e\n" % self.rho
                msg += 90 * '#' + '\n'
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
        
        # Main iteration loop
        while True:
            obj0 = problem.get_obj(admm_mdl)
            
            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.obj_terms[0],
                                           obj0 - problem.obj_terms[0],
                                           problem.get_rnorm(admm_mdl))
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog("\n" + msg)
             
            if self.logger_lin_solv:
                self.logger_lin_solv.addToLog("\n\t\t\tOuter iteration: %s"
                                              % (str(outer_iter).zfill(self.stopper.zfill)))
            
            if isnan(obj0):
                raise ValueError("Objective function values NaN!")
            
            if obj0 <= zero:
                print("Objective function is numerically zero! Stop the inversion")
                break
            
            self.save_results(outer_iter, problem, force_save=False)
            
# ======================================================================================================================
            ####        MAIN BLOCK      ####
            plt.figure(figsize=(12, 8))
            
            # 1) update x by solving linear problem
            z_minus_u.copy(z).scaleAdd(u, 1., -1.)
            linear_problem.op.ops[-1].const = self.rho
            if not self.warm_start:
                linear_problem.model.zero()
            linear_problem.setDefaults()
            plt.subplot(231)
            plt.plot(linear_problem.model.getNdArray(), label=r'$x_k$')
            self.linear_solver.run(linear_problem, verbose=inner_verbose)
            plt.plot(linear_problem.model.getNdArray(), label=r'$x_{k+1}$')
            plt.legend(), plt.title(r'x at k=%d' % outer_iter)
            plt.subplot(234), plt.plot(linear_problem.data.getNdArray()[-1][-1], label=r'dataL1 = $z_k - u_k$'), plt.legend()
            
            # 2) update z = S(Ax + u)
            plt.subplot(232)
            plt.plot(z.getNdArray()[-1], label=r'$z_k$')
            A.forward(False, linear_problem.model, Ax)
            Ax_plus_u.copy(Ax).scaleAdd(u, 1., 1.)
            z.copy(_soft_thresh(Ax_plus_u, thresh=lamda/self.rho))
            plt.plot(z.getNdArray()[-1], label=r'$z_{k+1}$')
            plt.plot(Ax.getNdArray()[-1], label=r'$Ax_{k+1}$')
            plt.legend(), plt.title(r'z at k=%d, λ/ρ=%.2e' % (outer_iter, lamda / self.rho))
            plt.subplot(235), plt.plot(Ax_plus_u.getNdArray()[-1], label=r'$Ax_{k+1}+u_k$')
            plt.legend()
            
            # 3) update residuals:
            # r = Ax - z
            self.primal.copy(Ax).scaleAdd(z, 1., -1.)
            # s = -rho A.H * r
            A.adjoint(False, self.dual, self.primal)
            self.dual.scale(-self.rho)
            plt.subplot(236)
            plt.plot(self.primal.getNdArray()[-1], label=r'$r=Ax-z$')
            plt.plot(self.dual.getNdArray(), label=r'$s=A^Hr$')
            plt.title('Residuals at k=%d, ρ=%.2e' % (outer_iter, self.rho)), plt.legend()
            # update penalty parameter and scaled dual variable
            rho_old = self.rho
            self._update_rho()
            plt.subplot(233), plt.plot(u.getNdArray()[-1], label=r'$u_k$')
            u.__add__(self.primal).scale(rho_old / self.rho)
            plt.plot(u.getNdArray()[-1], label=r'$u_{k+1}$')
            plt.legend(), plt.title(r'u at k=%d' % outer_iter)
            
            plt.show()
# ======================================================================================================================
            # update ADMM model
            admm_mdl.copy(linear_problem.model)
            
            outer_iter += 1
            
            obj1 = problem.get_obj(admm_mdl)
            
            # stopping criteria, see boyd2010distributed (3.12)
            eps_prim = np.sqrt(A.range.size) * self.tol_abs + self.tol_rel * max(Ax.norm(), z.norm())
            eps_dual = np.sqrt(A.domain.size) * self.tol_abs + self.tol_rel * self.rho * (A.H * u).norm()
            prim_norm = self.primal.norm()
            dual_norm = self.dual.norm()
            if prim_norm <= eps_prim:
                if self.create_msg:
                    msg = "Primal residual stopping criterion encoutered: %4e <= %.4e" % (prim_norm, eps_prim)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break
            if dual_norm <= eps_dual:
                if self.create_msg:
                    msg = "Dual residual stopping criterion encoutered: %4e <= %.4e" % (dual_norm, eps_dual)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break
                
            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.obj_terms[0],
                                       obj1 - problem.obj_terms[0],
                                       problem.get_rnorm(admm_mdl))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("admm_mdl", admm_mdl)
            
            if self.stopper.run(problem, outer_iter, initial_obj_value, verbose):
                break
        
        # writing last inverted model
        self.save_results(outer_iter, problem, force_save=True, force_write=True)
        
        # ending message and log file
        if self.create_msg:
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tADMM ALGORITHM log file end\n"
            msg += 90 * '#'
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog("\n" + msg)
        
        # Clear restart object
        self.restart.clear_restart()


def main():
    from sys import path
    path.insert(0, '.')
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    import pyNpOperator
    from pyProblem import ProblemLinearReg
    
    PLOT = True
    EXAMPLE = 'deconv'  # must be noisy, deconv, gaussian, medical or monarch
    
    if EXAMPLE == 'noisy':
        # data examples
        np.random.seed(1)
        nx = 101
        x = pyVec.vectorIC((nx,)).zero()
        x.getNdArray()[:nx // 2] = 10
        x.getNdArray()[nx // 2:3 * nx // 4] = -5
        
        Iop = pyOp.IdentityOp(x)
        TV = pyNpOperator.FirstDerivative(x)
        L = pyNpOperator.SecondDerivative(x)
        
        n = x.clone()
        n.getNdArray()[:] = np.random.normal(0, 1.0, nx)
        y = Iop * (x.clone() + n)
        
        dx = TV * x
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=2, label='x')
            plt.plot(y.getNdArray(), 'b', lw=2, label='y=x+n')
            plt.plot(dx.getNdArray(), ':k', lw=1, label='∂x')
            plt.legend()
            plt.title('Model, Data and Derivative')
            plt.show()
        
        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.)

        ADMM = ADMMsolver(BasicStopper(niter=30), niter_linear=10, niter_sparse=50)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=2, label='x')
            plt.plot(y.getNdArray(), 'b', lw=2, label='y=x+n')
            plt.plot(dx.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemADMM.model.getNdArray(), 'r', lw=1, label='x_inv')
            plt.plot((TV * problemADMM.model).getNdArray(), ':r', lw=1, label='∂(x_inv)')
            plt.legend()
            plt.title('ADMM inversion')
            plt.show()
    
    elif EXAMPLE == 'deconv':
        # 1D deconvolution for blocky signal
        nx = 201
        x = np.zeros((nx,), dtype=np.float32)
        x[20:30] = 10.
        x[50:75] = -5.
        x[100:150] = 2.5
        x[175:180] = 7.5
        x = pyVec.vectorIC(x)
        
        G = pyNpOperator.GaussianFilter(x, 2.0)
        y = G * x
        TV = pyNpOperator.FirstDerivative(x)
        dx = TV * x
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=2, label='x')
            plt.plot(y.getNdArray(), 'b', lw=2, label='y=Gx')
            plt.plot(dx.getNdArray(), 'k:', lw=1, label='∂x')
            plt.legend()
            plt.title('Model, Data and Derivative')
            plt.show()
        
        # ADMM
        w1 = .01
        mu = 10.
        tau = 2.
        niter_sparse = 50
        niter_linear = 10
        niter = 300
        problemADMM = ProblemLinearReg(x.clone().zero(), y, G, regsL1=TV, epsL1=w1)
        ADMM = ADMMsolver(BasicStopper(niter=niter), niter_linear=niter_linear,
                          logger=None, rho=None, rho_update=True, rho_tol=mu, rho_scale=tau,
                          warm_start=True, linear_solver='CG', rho_save=True, tol_abs=1e-3, tol_rel=1e-4)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=2, label='x')
            plt.plot(y.getNdArray(), 'b', lw=2, label='y=x+n')
            plt.plot(dx.getNdArray(), ':k', label='∂x')
            plt.plot(problemADMM.model.getNdArray(), 'r', label='x_inv')
            plt.plot((TV * problemADMM.model).getNdArray(), ':r', label='∂(x_inv)')
            plt.legend()
            plt.title('TV=%.e, μ=%.3f,τß=%.2f, niter=%d,%d,%d'
                      % (w1, mu, tau, niter, niter_linear, niter_sparse))
            plt.show()
    
    elif EXAMPLE == 'gaussian':
        x = pyVec.vectorIC(np.empty((301, 601))).set(0)
        x.getNdArray()[150, 300] = 1.0
        # x.getNdArray()[100, 200] = -5.0
        # x.getNdArray()[280, 400] = 1.0
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(x.getNdArray()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        G = pyNpOperator.GaussianFilter(x, [25, 15])
        y = G * x
        TV = pyNpOperator.TotalVariation(x)
        
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(y.getNdArray()), plt.colorbar()
            plt.title('Data')
            plt.show()

        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, G, regsL1=TV, epsL1=2.)
        
        ADMM = ADMMsolver(BasicStopper(niter=10), niter_linear=30, niter_sparse=100)
        ADMM.setDefaults()
        ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemADMM.model.getNdArray()), plt.colorbar()
            plt.title('ADMM')
            plt.show()

    elif EXAMPLE == 'medical':
        x = pyVec.vectorIC(np.load('../testdata/shepp_logan_phantom.npy', allow_pickle=True).astype(np.float32))
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        Blurring = pyNpOperator.GaussianFilter(x, [3, 3])
        TV = pyNpOperator.TotalVariation(x)
        y = Blurring * x
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(y.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Data')
            plt.show()
            
        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Blurring,
                                     regsL1=TV, epsL1=.1)

        ADMM = ADMMsolver(BasicStopper(niter=10), niter_linear=30, niter_sparse=10)
        ADMM.setDefaults(save_obj=True, save_model=True)
        ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemADMM.model.getNdArray(), cmap='bone'), plt.colorbar()
            plt.title('ADMM TV, ε=%.2e, %d iter'
                      % (problemADMM.epsL1[0], ADMM.stopper.niter))
            plt.show()
    
    elif EXAMPLE == 'monarch':
        x = pyVec.vectorIC(np.load('../testdata/monarch.npy', allow_pickle=True).astype(np.float32))
        np.random.seed(12345)
        sigma = 0.05
        y = x.clone()
        y.getNdArray()[:] += np.random.normal(0.0, sigma, y.shape)
        Op = pyOp.IdentityOp(x)
        TV = pyNpOperator.TotalVariation(x)
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.arr, cmap='gray'), plt.colorbar()
            plt.title('Model')
            plt.show()
            
            plt.figure(figsize=(5, 4))
            plt.imshow(y.arr, cmap='gray'), plt.colorbar()
            plt.title('Data, std=%.2f' % sigma)
            plt.show()
        
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Op, regsL1=TV, epsL1=.04)
        
        ADMM = ADMMsolver(BasicStopper(niter=200), niter_linear=30, niter_sparse=10)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemADMM.model.getNdArray(), cmap='gray'), plt.colorbar()
            plt.title(r'ADMM TV, $\varepsilon=%.2e$, %d iter'
                      % (problemADMM.epsL1[0], ADMM.stopper.niter))
            plt.show()
    
    else:
        raise ValueError("EXAMPLE has to be one of noisy, gaussian, medical or monarch")
    
    return 0


if __name__ == '__main__':
    main()
