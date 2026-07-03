import time
import numpy as np
import pyVector as pyVec
import pyOperator as pyOp
from pyProblem import ProblemL2LinearReg
from pyLinearSolver import LSQRsolver
from pyStopper import BasicStopper
import pylops.optimization.sparsity as pos


def _shrinkage(x, thresh):
    xabs = np.abs(x)
    return x / (xabs + 1e-10) * np.maximum(xabs - thresh, 0)


def SplitBregman(Op, RegsL1, data, niter_outer=3, niter_inner=5, RegsL2=None,
                 dataregsL2=None, mu=1., epsRL1s=None, epsRL2s=None,
                 tol=1e-10, tau=1., x0=None, restart=False,
                 show=False, **kwargs_lsqr):
    if show:
        tstart = time.time()
        print('Split-Bregman optimization\n'
              '---------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'niter_outer = %3d     niter_inner = %3d   tol = %2.2e\n'
              'mu = %2.2e         epsL1 = %s\t  epsL2 = %s     '
              % (Op.range.size, Op.domain.size,  # Op.shape[0], Op.shape[1],
                 niter_outer, niter_inner, tol,
                 mu, str(epsRL1s), str(epsRL2s)))
        print('---------------------------------------------------------\n')
        # head1 = '   Itn          x[0]           DFid            ObjFunc'
        # print(head1)
    
    # L1 regularizations
    nregsL1 = len(RegsL1)
    # b = [np.zeros(RegL1.shape[0]) for RegL1 in RegsL1]
    b = [np.zeros(RegL1.range.size) for RegL1 in RegsL1]
    d = b.copy()
    
    # L2 regularizations
    nregsL2 = 0 if RegsL2 is None else len(RegsL2)
    if nregsL2 > 0:
        Regs = RegsL2 + RegsL1
        if dataregsL2 is None:
            # dataregsL2 = [np.zeros(Op.shape[1])] * nregsL2
            dataregsL2 = [np.zeros(Op.domain.size)] * nregsL2
    else:
        Regs = RegsL1
        dataregsL2 = []
    
    # Rescale dampings
    epsRs = [np.sqrt(epsRL2s[ireg]/2) / np.sqrt(mu/2) for ireg in range(nregsL2)] + \
            [np.sqrt(epsRL1s[ireg]/2) / np.sqrt(mu/2) for ireg in range(nregsL1)]
    # xinv = np.zeros_like(np.zeros(Op.shape[1])) if x0 is None else x0
    xinv = np.zeros_like(np.zeros(Op.domain.size)) if x0 is None else x0
    # xold = np.inf * np.ones_like(np.zeros(Op.shape[1]))
    xold = np.inf * np.ones_like(np.zeros(Op.domain.size))
    
    itn_out = 0
    while np.linalg.norm(xinv - xold) > tol and itn_out < niter_outer:
        xold = xinv
        for _ in range(niter_inner):
            # Regularized problem
            dataregs = dataregsL2 + [d[ireg] - b[ireg] for ireg in range(nregsL1)]
            
            # xinv = RegularizedInversion(Op, Regs, data,
            #                             dataregs=dataregs,
            #                             epsRs=epsRs,
            #                             x0=x0 if restart else xinv,
            #                             **kwargs_lsqr)
            
            LinProb = ProblemL2LinearReg(model=pyVec.vectorIC((x0 if restart else xinv).reshape(Op.domain.shape)),
                                         data=data,
                                         op=Op,
                                         epsilon=epsRs[0],
                                         reg_op=Regs[0],
                                         prior_model=pyVec.vectorIC(dataregs[0].reshape(Regs[0].range.shape)))
            LinSolver = LSQRsolver(BasicStopper(niter=kwargs_lsqr['iter_lim']))
            LinSolver.run(LinProb)
            xinv_Vec = LinProb.model.clone()
            xinv = xinv_Vec.getNdArray()
            
            # Shrinkage
            # d = [_shrinkage(RegsL1[ireg] * xinv + b[ireg], epsRL1s[ireg]) for ireg in range(nregsL1)]
            d = [_shrinkage((RegsL1[ireg] * xinv_Vec).getNdArray().ravel() + b[ireg], epsRL1s[ireg]) for ireg in
                 range(nregsL1)]
        
        # Bregman update
        # b = [b[ireg] + tau * (RegsL1[ireg] * xinv - d[ireg]) for ireg in range(nregsL1)]
        b = [b[ireg] + tau * ((RegsL1[ireg] * xinv_Vec).getNdArray().ravel() - d[ireg]) for ireg in range(nregsL1)]
        
        itn_out += 1
        
        if show:
            # costdata = mu / 2. * np.linalg.norm(data - Op.matvec(xinv)) ** 2
            res = data.clone() - Op * xinv_Vec
            costdata = mu / 2. * res.norm() ** 2
            
            # costregL2 = 0 if RegsL2 is None else [epsRL2 * np.linalg.norm(dataregL2 - RegL2.matvec(xinv)) ** 2
            #                                       for epsRL2, RegL2, dataregL2 in zip(epsRL2s, RegsL2, dataregsL2)]
            costregL2 = 0 if RegsL2 is None else sum(
                [epsRL2 * (dataregL2 - (RegL2 * xinv_Vec).getNdArray().ravel()) ** 2
                 for epsRL2, RegL2, dataregL2 in zip(epsRL2s, RegsL2, dataregsL2)])
            
            # costregL1 = [np.linalg.norm(RegL1.matvec(xinv), ord=1) for epsRL1, RegL1 in zip(epsRL1s, RegsL1)]
            costregL1 = sum([epsRL1 * (RegL1 * xinv_Vec).norm(1) for epsRL1, RegL1 in zip(epsRL1s, RegsL1)])
            
            cost = costdata + np.sum(np.array(costregL2)) + np.sum(np.array(costregL1))
            iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e" \
                       % (str(itn_out).zfill(4), cost, costdata, costregL2 + costregL1, res.norm())
            # msg = '%6g  %12.5e       %10.3e        %9.3e' % \
            #       (np.abs(itn_out), xinv[0], costdata, cost)
            print(iter_msg)
    
    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (itn_out, time.time() - tstart))
        print('---------------------------------------------------------\n')
    # return xinv, itn_out
    return xinv_Vec, itn_out


if __name__ == '__main__':
    from sys import path
    
    path.insert(0, '.')
    import matplotlib.pyplot as plt
    import pyNpOperator
    from pyProblem import ProblemLinearReg, ProblemL2Linear
    from pySparseSolver import SplitBregmanSolver
    from pyLinearSolver import LCGsolver
    
    PLOT = True
    EXAMPLE = 'medical'  # must be noisy, deconv, gaussian, medical or monarch
    
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
        
        derivative = TV * x
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='d=x+n')
            plt.plot(derivative.getNdArray(), '.b', lw=2, label='∂x')
            plt.legend()
            plt.title('Model, Data and Derivative')
            plt.show()
        
        # # SplitBregman
        # x_inv, _ = SplitBregman(Op=Iop, RegsL1=[TV], data=y,
        #                         niter_outer=50, niter_inner=3, RegsL2=None,
        #                         dataregsL2=None, mu=0.01, epsRL1s=[.3], epsRL2s=None,
        #                         tol=1e-4, tau=1., x0=None, restart=False,
        #                         show=True, **dict(iter_lim=30))
        problemSB = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=2.)
        SB = SplitBregmanSolver(BasicStopper(niter=400),
                                niter_inner=1, niter_solver=200,
                                linear_solver='LSQR', breg_weight=1.)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        x_inv = problemSB.get_model()

        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='d=x+n')
            plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(x_inv.getNdArray(), 'r', lw=2, label='$x_{inv}$')
            plt.plot((TV * x_inv).getNdArray(), ':r', lw=2, label='∂($x_{inv}$)')
            plt.legend()
            plt.title('SB inversion')
            plt.show()
            # Objective function convergence
            # plt.figure(figsize=(5, 4))
            # plt.plot(np.log10(SB.obj / SB.obj[0]), 'r', lw=1, label='SplitBregman')
            # obj_true = problemSB.get_obj(x)
            # plt.plot([np.log10(obj_true / SB.obj[0])] * len(SB.obj), 'k--', lw=1, label='true solution obj value')
            # plt.legend()
            # plt.title('Convergence curve')
            # plt.show()
    
    elif EXAMPLE == 'deconv':
        # 1D deconvolution for blocky signal
        nx = 201
        x = pyVec.vectorIC(np.zeros((nx,), dtype=np.float32)).zero()
        x.getNdArray()[20:30] = 10.
        x.getNdArray()[50:75] = -5.
        x.getNdArray()[100:150] = 2.5
        x.getNdArray()[175:180] = 7.5
        
        G = pyNpOperator.GaussianFilter(x, 2.0)
        y = G * x

        if PLOT:
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.plot(x.getNdArray(), 'k',  label='x')
            plt.plot(y.getNdArray(), 'b',  label='d')
            plt.title("model and data")
            ax.autoscale(enable=True, axis='x', tight=True)
            plt.ylim(-5.5, 10.5)
            plt.legend()
            plt.show()

        TV = pyNpOperator.FirstDerivative(x)
        Iop = pyOp.IdentityOp(x)
        w1 = .1
        niter = 200
        niter_inner = 2
        niter_solver = 10
        breg = 1.
        # x_hybrid, _ = SplitBregman(G, [TV], y, niter_outer=niter, niter_inner=niter_inner,
        #                            mu=1.0, epsRL1s=[w1], epsRL2s=None, tau=breg,
        #                            show=True, **dict(iter_lim=niter_solver))

        problemSB = ProblemLinearReg(x.clone().zero(), y, G, regsL1=TV, epsL1=w1)
        SB = SplitBregmanSolver(BasicStopper(niter=niter),
                                niter_inner=niter_inner, niter_solver=niter_solver,
                                linear_solver='LSQR', breg_weight=breg)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        
        if PLOT:
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.plot(x.getNdArray(), 'k', label="true model")
            # plt.plot(x_pylops, 'g--', label="pyLops")
            # plt.plot(x_hybrid.getNdArray(), 'b--', label="Hybrid")
            plt.plot(problemSB.model.getNdArray(), 'r--', label="SB")
            plt.title('TV=%.e, ß=%.2f, niter=%d,%d,%d'
                      % (w1, breg, niter, niter_inner, niter_solver))
            plt.ylim(-5.5, 10.5)
            ax.autoscale(enable=True, axis='x', tight=True)
            plt.legend()
            plt.show()
            
    elif EXAMPLE == 'gaussian':
        x = pyVec.vectorIC(np.empty((301, 601))).set(0)
        x.getNdArray()[150, 300] = 1.0
        x.getNdArray()[100, 200] = -5.0
        x.getNdArray()[280, 400] = 1.0
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(x.getNdArray()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        G = pyNpOperator.GaussianFilter(x, [25, 15])
        y = G * x
        # y.scale(1./y.norm())
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(y.getNdArray()), plt.colorbar()
            plt.title('Data')
            plt.show()
        
        # SplitBregman
        I = pyOp.IdentityOp(x)
        problemSB = ProblemLinearReg(x.clone().zero(), y, G, regsL1=I, epsL1=10.)
        
        SB = SplitBregmanSolver(BasicStopper(niter=50), niter_inner=10, niter_solver=30,
                                linear_solver='LSQR', breg_weight=1., warm_start=False)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemSB.model.getNdArray()), plt.colorbar()
            plt.title('SplitBregman')
            plt.show()
    
    elif EXAMPLE == 'medical':
        x = pyVec.vectorIC(np.load('../testdata/shepp_logan_phantom.npy', allow_pickle=True).astype(np.float32))
        x = x.scale(1/255.)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        G = pyNpOperator.GaussianFilter(x, [3, 3])
        y = G * x
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(y.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Data')
            plt.show()
        
        # SplitBregman
        # the gradient of the image is 6e3
        D = pyNpOperator.TotalVariation(x)
        Dx = pyNpOperator.FirstDerivative(x, axis=1)
        Dz = pyNpOperator.FirstDerivative(x, axis=0)

        plt.imshow((D * x).getNdArray(), cmap='gray'), plt.colorbar(), plt.title('D'), plt.show()
        
        problemSB = ProblemLinearReg(x.clone().zero(), y, G, regsL1=[Dx, Dz], epsL1=[.001]*2)
        SB = SplitBregmanSolver(BasicStopper(niter=100), niter_inner=10, niter_solver=50,
                                linear_solver='LSQR', breg_weight=1., warm_start=True)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemSB.model.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title(r'SB TV, $\varepsilon=%.e$, %d iter' % (problemSB.epsL1[0], SB.stopper.niter))
            plt.show()
            plt.figure(figsize=(5, 4))
            plt.plot(np.log10(SB.obj / SB.obj[0]), 'r', lw=1, label='SplitBregman')
            obj_true = problemSB.get_obj(x)
            plt.plot([np.log10(obj_true / SB.obj[0])] * len(SB.obj), 'k--', lw=1, label='true solution obj value')
            plt.legend()
            plt.title('Convergence curve')
            plt.show()
    
    elif EXAMPLE == 'monarch':
        x = pyVec.vectorIC(np.load('../testdata/monarch.npy', allow_pickle=True).astype(np.float32))
        np.random.seed(12345)
        sigma = 0.05
        y = x.clone()
        y.getNdArray()[:] += np.random.normal(0.0, sigma, y.shape)
        Op = pyOp.IdentityOp(x)
        TV = pyNpOperator.TotalVariation(x)
        w1 = 1.
        niter = 10
        niter_inner = 10
        niter_solver = 10
        lambd = 1.
        breg = 1.
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.arr, cmap='gray'), plt.colorbar()
            plt.title('Model')
            plt.show()
            
            plt.figure(figsize=(5, 4))
            plt.imshow(y.arr, cmap='gray'), plt.colorbar()
            plt.title('Data, std=%.2f' % sigma)
            plt.show()
        
        Op_pylops = pyLopsInterface.ToPylops(Op)
        TV_pylops = pyLopsInterface.ToPylops(TV)
        y_pylops = Op_pylops * x.arr.ravel() + np.random.normal(0.0, sigma, x.shape).ravel()
        x_pylops, _ = pos.SplitBregman(Op=Op_pylops, RegsL1=[TV_pylops], data=y_pylops,
                                       niter_outer=niter, niter_inner=niter_inner,
                                       RegsL2=None, dataregsL2=None, mu=lambd,
                                       epsRL1s=[w1], epsRL2s=None,
                                       tol=1e-10, tau=breg, x0=None, restart=False,
                                       show=True, **dict(iter_lim=niter_solver))
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x_pylops.reshape(x.shape), cmap='gray'), plt.colorbar()
            plt.title(r'ADMM TV, ε=%.e' % w1)
            plt.show()
    
    else:
        raise ValueError("EXAMPLE has to be one of noisy, gaussian, medical or monarch")
