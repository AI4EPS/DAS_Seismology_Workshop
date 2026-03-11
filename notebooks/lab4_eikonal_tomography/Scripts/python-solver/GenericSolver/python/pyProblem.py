# Module containing the definition of abstract inverse problems
import pyVector as pyVec
import pyOperator as pyOp
from math import isnan


class Bounds:
    """Class used to enforce boundary constraints during the inversion"""

    def __init__(self, minBound=None, maxBound=None):
        """
        Bounds constructor
        minBound    = [None] - vector class; vector containing minimum values of the model vector
        maxBound    = [None] - vector class; vector containing maximum values of the model vector
        """
        self.minBound = minBound
        self.maxBound = maxBound
        if minBound is not None:
            self.minBound = minBound.clone()
        if maxBound is not None:
            self.maxBound = maxBound.clone()
        # If only the lower bound was provided we use the opposite of the lower bound to clip the values
        if self.minBound is not None and self.maxBound is None:
            self.minBound.scale(-1.0)
        return

    def apply(self, input_vec):
        """
        Function for applying the model bounds
        """
        if self.minBound is not None and self.maxBound is None:
            if not input_vec.checkSame(self.minBound):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.scale(-1.0)
            input_vec.clipVector(input_vec, self.minBound)
            input_vec.scale(-1.0)
        elif self.minBound is None and self.maxBound is not None:
            if not input_vec.checkSame(self.maxBound):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.clipVector(input_vec, self.maxBound)
        elif self.minBound is not None and self.maxBound is not None:
            if (not (input_vec.checkSame(self.minBound) and input_vec.checkSame(
                    self.maxBound))):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.clipVector(self.minBound, self.maxBound)
        return


class Problem:
    """Problem parent object"""

    # Default class methods/functions
    def __init__(self, minBound=None, maxBound=None, boundProj=None):
        """Default class constructor for Problem"""
        if minBound is not None or maxBound is not None:
            # Simple box bounds
            self.bounds = Bounds(minBound, maxBound)  # Setting the bounds of the problem (if necessary)
        elif boundProj is not None:
            # Projection operator onto the bounds
            self.bounds = boundProj
        # Setting common variables
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.dres_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        self.linear = False  # By default all problem are non-linear

    def __del__(self):
        """Default destructor"""
        return

    def setDefaults(self):
        """Default common variables for any inverse problem"""
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.dres_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        return

    def set_model(self, model):
        """Setting internal model vector"""
        if model.isDifferent(self.model):
            self.model.copy(model)
            self.obj_updated = False
            self.res_updated = False
            self.grad_updated = False
            self.dres_updated = False

    def set_residual(self, residual):
        """Setting internal residual vector"""
        # Useful for linear inversion (to avoid residual computation)
        if self.res.isDifferent(residual):
            self.res.copy(residual)
            # If residuals have changed, recompute gradient and objective function value
            self.grad_updated = False
            self.obj_updated = False
        self.res_updated = True
        return

    def get_model(self):
        """Accessor for model vector"""
        return self.model

    def get_dmodel(self):
        """Accessor for model vector"""
        return self.dmodel

    def get_rnorm(self, model):
        """Accessor for residual vector norm"""
        self.get_res(model)
        return self.get_res(model).norm()

    def get_gnorm(self, model):
        """Accessor for gradient vector norm"""
        return self.get_grad(model).norm()

    def get_obj(self, model):
        """Accessor for objective function"""
        self.set_model(model)
        if not self.obj_updated:
            self.res = self.get_res(self.model)
            self.obj = self.objf(self.res)
            self.obj_updated = True
        return self.obj

    def get_res(self, model):
        """Accessor for residual vector"""
        self.set_model(model)
        if not self.res_updated:
            self.fevals += 1
            self.res = self.resf(self.model)
            self.res_updated = True
        return self.res

    def get_grad(self, model):
        """Accessor for gradient vector"""
        self.set_model(model)
        if not self.grad_updated:
            self.res = self.get_res(self.model)
            self.grad = self.gradf(self.model, self.res)
            self.gevals += 1
            if self.linear:
                self.fevals += 1
            self.grad_updated = True
        return self.grad

    def get_dres(self, model, dmodel):
        """Accessor for dresidual vector (i.e., application of the Jacobian to Dmodel vector)"""
        self.set_model(model)
        if not self.dres_updated or dmodel.isDifferent(self.dmodel):
            self.dmodel.copy(dmodel)
            self.dres = self.dresf(self.model, self.dmodel)
            if self.linear:
                self.fevals += 1
            self.dres_updated = True
        return self.dres

    def get_fevals(self):
        """Accessor for number of objective function evalutions"""
        return self.fevals

    def get_gevals(self):
        """Accessor for number of gradient evalutions"""
        return self.gevals

    def objf(self, res):
        """Dummy objf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement objf for problem in the derived class!")

    def resf(self, model):
        """Dummy resf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement resf for problem in the derived class!")

    def dresf(self, model, dmodel):
        """Dummy dresf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement dresf for problem in the derived class!")

    def gradf(self, model, residual):
        """Dummy gradf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement gradf for problem in the derived class!")


class ProblemL2Linear(Problem):
    """Linear inverse problem of the form 1/2*|Lm-d|_2"""

    def __init__(self, model, data, op, grad_mask=None, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of linear problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - linear operator class; L operator
           grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
           minBound     = [None] - vector class; Minimum value bounds
           maxBound     = [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
           prec       	= [None] - linear operator class; Preconditioning matrix
        """
        # Setting the bounds (if any)
        super(ProblemL2Linear, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Residual vector
        self.res = data.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting linear operator
        self.op = op
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Preconditioning matrix
        self.prec = prec
        # Setting default variables
        self.setDefaults()
        self.linear = True
        return

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """Method to return residual vector r = Lm - d"""
        # Computing Lm
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing Lm - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector g = L'r = L'(Lm - d)"""
        # Computing L'r = g
        self.op.adjoint(False, self.grad, res)
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = Ldm"""
        # Computing Ldm = dres
        self.op.forward(False, dmodel, self.dres)
        return self.dres

    def objf(self, res):
        """Method to return objective function value 1/2|Lm-d|_2"""
        val = res.norm()
        obj = 0.5 * val * val
        return obj


class ProblemLinearSymmetric(Problem):
    """Linear inverse problem of the form 1/2m'Am - m'b"""

    def __init__(self, model, data, op, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Constructor of linear symmetric problem:
        model    	= [no default] - vector class; Initial model vector
        data     	= [no default] - vector class; Data vector
        op       	= [no default] - linear operator class; A symmetric operator (i.e., A = A')
        minBound		= [None] - vector class; Minimum value bounds
        maxBound		= [None] - vector class; Maximum value bounds
        boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        prec       	= [None] - linear operator class; Preconditioning matrix
        """
        # Setting the bounds (if any)
        super(ProblemLinearSymmetric, self).__init__(minBound, maxBound, boundProj)
        # Checking range and domain are the same
        if not model.checkSame(data) and not op.domain.checkSame(op.range):
            raise ValueError("Data and model vector live in different spaces!")
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Copying the pointer to data vector
        self.data = data
        # Residual vector
        self.res = data.clone()
        self.res.zero()
        # Gradient vector is equal to the residual vector
        self.grad = self.res
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting linear operator
        self.op = op
        # Preconditioning matrix
        self.prec = prec
        # Setting default variables
        self.setDefaults()
        self.linear = True

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """Method to return residual vector r = Am - b"""
        # Computing Lm
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing Lm - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector equal to residual one"""
        # Assigning g = r
        self.grad = self.res
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = Adm"""
        # Computing Ldm = dres
        self.op.forward(False, dmodel, self.dres)
        return self.dres

    def objf(self, res):
        """Method to return objective function value 1/2m'Am - m'b"""
        obj = 0.5 * (self.model.dot(res) - self.model.dot(self.data))
        return obj


class ProblemL2LinearReg(Problem):
    """Linear inverse problem regularized of the form 1/2*|Lm-d|_2 + epsilon^2/2*|Am-m_prior|_2"""

    def __init__(self, model, data, op, epsilon, grad_mask=None, reg_op=None, prior_model=None, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Constructor of linear regularized problem:
        model    	= [no default] - vector class; Initial model vector
        data     	= [no default] - vector class; Data vector
        op       	= [no default] - linear operator class; L operator
        epsilon      = [no default] - float; regularization weight
        grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
        reg_op       = [Identity] - linear operator class; A regularization operator
        prior_model  = [None] - vector class; Prior model for regularization term
        minBound		= [None] - vector class; Minimum value bounds
        maxBound		= [None] - vector class; Maximum value bounds
        boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        prec       	= [None] - linear operator class; Preconditioning matrix
        """
        # Setting the bounds (if any)
        super(ProblemL2LinearReg, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting a prior model (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            reg_op = pyOp.IdentityOp(self.model)
        # Checking if space of the prior model is consistent with range of
        # regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior model space no consistent with range of regularization operator")
        self.op = pyOp.stackOperator(op, reg_op)  # Modeling operator
        self.epsilon = epsilon  # Regularization weight
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Residual vector (data and model residual vectors)
        self.res = self.op.range.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = True
        # Preconditioning matrix
        self.prec = prec
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]

    def __del__(self):
        """Default destructor"""
        return

    def estimate_epsilon(self, verbose=False, logger=None):
        """
        Method returning epsilon that balances the first gradient in the 'extended-data' space or initial data residuals
        """
        msg = "Epsilon Scale evaluation"
        if verbose:
            print(msg)
        if logger:
            logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
        # Keeping the initial model vector
        prblm_mdl = self.get_model()
        mdl_tmp = prblm_mdl.clone()
        # Keeping user-predefined epsilon if any
        epsilon = self.epsilon
        # Setting epsilon to one to evaluate the scale
        self.epsilon = 1.0
        if self.model.norm() != 0.:
            prblm_res = self.get_res(self.model)
            msg = "	Epsilon balancing data and regularization residuals is: %.2e"
        else:
            prblm_grad = self.get_grad(self.model)  # Compute first gradient
            prblm_res = self.get_res(prblm_grad)  # Compute residual arising from the gradient
            # Balancing the first gradient in the 'extended-data' space
            prblm_res.vecs[0].scaleAdd(self.data)  # Remove data vector (Lg0 - d + d)
            if self.prior_model is not None:
                prblm_res.vecs[1].scaleAdd(self.prior_model)  # Remove prior model vector (Ag0 - m_prior + m_prior)
            msg = "	Epsilon balancing the data-space gradients is: %.2e"
        res_data_norm = prblm_res.vecs[0].norm()
        res_model_norm = prblm_res.vecs[1].norm()
        if isnan(res_model_norm) or isnan(res_data_norm):
            raise ValueError("Obtained NaN: Residual-data-side-norm = %.2e, Residual-model-side-norm = %.2e"
                             % (res_data_norm, res_model_norm))
        if res_model_norm == 0.:
            raise ValueError("Model residual component norm is zero, cannot find epsilon scale")
        # Resetting user-predefined epsilon if any
        self.epsilon = epsilon
        # Resetting problem initial model vector
        self.set_model(mdl_tmp)
        del mdl_tmp
        epsilon_balance = res_data_norm / res_model_norm
        # Resetting feval
        self.fevals = 0
        msg = msg % epsilon_balance
        if verbose:
            print(msg)
        if logger:
            logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        return epsilon_balance

    def resf(self, model):
        """Method to return residual vector r = [r_d; r_m]: r_d = Lm - d; r_m = epsilon * (Am - m_prior) """
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.data, 1., -1.)
        # Computing r_m = Am - m_prior
        if self.prior_model is not None:
            self.res.vecs[1].scaleAdd(self.prior_model, 1., -1.)
        # Scaling by epsilon epsilon*r_m
        self.res.vecs[1].scale(self.epsilon)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector g = L'r_d + epsilon*A'r_m"""
        # Scaling by epsilon the model residual vector (saving temporarily residual regularization)
        # g = epsilon*A'r_m
        self.op.ops[1].adjoint(False, self.grad, res.vecs[1])
        self.grad.scale(self.epsilon)
        # g = L'r_d + epsilon*A'r_m
        self.op.ops[0].adjoint(True, self.grad, res.vecs[0])
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = (L + epsilon * A)dm"""
        # Computing Ldm = dres_d
        self.op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.epsilon)
        return self.dres

    def objf(self, res):
        """Method to return objective function value 1/2|Lm-d|_2 + epsilon^2/2*|Am-m_prior|_2"""
        for idx in range(res.n):
            val = res.vecs[idx].norm()
            self.obj_terms[idx] = 0.5 * val * val
        return sum(self.obj_terms)


class ProblemL1Lasso(Problem):
    """Convex problem 1/2*| y - Am |_2 + lambda*| m |_1"""

    def __init__(self, model, data, op, op_norm=None, lambda_value=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of convex L1-norm LASSO inversion problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - linear operator class; L operator
           lambda_value	= [None] - Regularization weight. Not necessary for ISTC solver but required for ISTA and FISTA
           op_norm		= [None] - float; A operator norm that will be evaluated with the power method if not provided
           minBound		= [None] - vector class; Minimum value bounds
           maxBound		= [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(ProblemL1Lasso, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting linear operator
        self.op = op  # Modeling operator
        # Residual vector (data and model residual vectors)
        self.res = pyVec.superVector(op.range.clone(), op.domain.clone())
        self.res.zero()
        # Dresidual vector
        self.dres = None  # Not necessary for the inversion
        # Setting default variables
        self.setDefaults()
        self.linear = True
        if op_norm is not None:
            # Using user-provided A operator norm
            self.op_norm = op_norm  # Operator Norm necessary for solver
        else:
            # Evaluating operator norm using power method
            self.op_norm = self.op.powerMethod()
        self.lambda_value = lambda_value
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
        return

    def set_lambda(self, lambda_in):
        # Set lambda
        self.lambda_value = lambda_in
        return

    def objf(self, res):
        """Method to return objective function value 1/2*| y - Am |_2 + lambda*| m |_1"""
        # data term
        val = res.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val * val
        # model term
        self.obj_terms[1] = self.lambda_value * res.vecs[1].norm(1)
        return sum(self.obj_terms)

    # define function that computes residuals
    def resf(self, model):
        """ y - alpha * A m = rd (self.res[0]) and m = rm (self.res[1]);"""
        if model.norm() != 0.:
            self.op.forward(False, model, self.res.vecs[0])
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.data, -1., 1.)
        # Run regularization part
        self.res.vecs[1].copy(model)
        return self.res

    # function that projects search direction into data space (Not necessary for ISTC)
    def dresf(self, model, dmodel):
        """Linear projection of the model perturbation onto the data space. Method not implemented"""
        raise NotImplementedError("dresf is not necessary for ISTC; DO NOT CALL THIS METHOD")

    # function to compute gradient (Soft thresholding applied outside in the solver)
    def gradf(self, model, res):
        """- A'r_data (residual[0]) = g"""
        # Apply an adjoint modeling
        self.op.adjoint(False, self.grad, res.vecs[0])
        # Applying negative scaling
        self.grad.scale(-1.0)
        return self.grad


# TODO make it accept L2 reg problems
class ProblemLinearReg(Problem):
    def __init__(self, model, data, op, epsL1=None, regsL1=None, epsL2=None, regsL2=None, dataregsL2=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Linear Problem with both L1 and L2 regularizers:

        .. math ::
            1 / 2 |Op m - d|_2^2 +
            \sum_i epsL2_i |R2_i m - dr|_2^2 +
            \sum_i epsL1_i |R1_i m|_1

        :param model        : vector; initial model
        :param data         : vector; data
        :param op           : LinearOperator; data fidelity operator
        :param epsL1        : list; weights of L1 regularizers [None]
        :param regsL1       : list; L1 regularizers of class LinearOperator [None]
        :param epsL2        : list; weights of L2 regularizers [None]
        :param regsL2       : list; L2 regularizers of class LinearOperator [None]
        :param dataregsL2   : vector; prior model for L2 regularization term [None]
        :param minBound     : vector; minimum value bounds
        :param maxBound     : vector; maximum value bounds
        :param boundProj    : Bounds; object with a method "apply(x)" to project x onto some convex set
        """
        super(ProblemLinearReg, self).__init__(minBound, maxBound, boundProj)
        self.model = model
        self.dmodel = model.clone().zero()
        self.grad = self.dmodel.clone()
        self.data = data
        self.op = op

        self.minBound = minBound
        self.maxBound = maxBound
        self.boundProj = boundProj

        # L1 Regularizations
        self.regL1_op = None if regsL1 is None else pyOp.Vstack(regsL1)
        self.nregsL1 = self.regL1_op.n if self.regL1_op is not None else 0
        self.epsL1 = epsL1 if epsL1 is not None else []
        if type(self.epsL1) in [int, float]:
            self.epsL1 = [self.epsL1]
        assert len(self.epsL1) == self.nregsL1, 'The number of L1 regs and related weights mismatch!'

        # L2 Regularizations
        self.regL2_op = None if regsL2 is None else pyOp.Vstack(regsL2)
        self.nregsL2 = self.regL2_op.n if self.regL2_op is not None else 0
        self.epsL2 = epsL2 if epsL2 is not None else []
        if type(self.epsL2) in [int, float]:
            self.epsL2 = [self.epsL2]
        assert len(self.epsL2) == self.nregsL2, 'The number of L2 regs and related weights mismatch!'

        if self.regL2_op is not None:
            self.dataregsL2 = dataregsL2 if dataregsL2 is not None else self.regL2_op.range.clone().zero()
        else:
            self.dataregsL2 = None

        # At this point we should have:
        # - a list of L1 regularizers;
        # - a list of L1 weights (with same length of previous);
        # - a list of L2 regularizers (even empty is ok);
        # - a list of L2 weights (with same length of previous);
        # - a list of L2 dataregs (with same length of previous);

        # Last settings
        self.obj_terms = [None] * (1 + self.nregsL2 + self.nregsL1)
        self.linear = True
        # store the "residuals" (for computing the objective function)
        self.res_data = self.op.range.clone().zero()
        self.res_regsL2 = self.regL2_op.range.clone().zero() if self.nregsL2 != 0 else None
        self.res_regsL1 = self.regL1_op.range.clone().zero() if self.nregsL1 != 0 else None
        # this last superVector is instantiated with pointers to res_data and res_regs!
        self.res = pyVec.superVector(self.res_data, self.res_regsL2, self.res_regsL1)

        # flags for avoiding extra computations
        self.res_data_already_computed = False
        self.res_regsL1_already_computed = False
        self.res_regsL2_already_computed = False

        # TODO add compatibility with L2 problems and Lasso

    def __del__(self):
        """Default destructor"""
        return

    def objf(self, res):
        """
        Compute objective function based on the residual (super)vector

        .. math ::
            1 / 2 |Op m - d|_2^2 +
            \sum_i epsL2_i |R2_i m - dr|_2^2 +
            \sum_i epsL1_i |R1_i m|_1

        """
        res_data = res.vecs[0]
        res_regsL2 = res.vecs[1] if self.res_regsL2 is not None else None
        if self.res_regsL1 is not None:
            res_regsL1 = res.vecs[2] if self.res_regsL2 is not None else res.vecs[1]
        else:
            res_regsL1 = None

        self.obj_terms[0] = .5 * res_data.norm(2) ** 2  # data fidelity

        if res_regsL2 is not None:
            for idx in range(self.nregsL2):
                self.obj_terms[1 + idx] = self.epsL2[idx] * res_regsL2.vecs[idx].norm(2) ** 2
        if res_regsL1 is not None:
            for idx in range(self.nregsL1):
                self.obj_terms[1 + self.nregsL2 + idx] = self.epsL1[idx] * res_regsL1.vecs[idx].norm(1)

        return sum(self.obj_terms)

    def resf(self, model):
        """Compute residuals from current model"""

        # compute data residual: Op * m - d
        if model.norm() != 0:
            self.op.forward(False, model, self.res_data)  # rd = Op * m
        else:
            self.res_data.zero()
        self.res_data.scaleAdd(self.data, 1., -1.)  # rd = rd - d

        # compute L2 reg residuals
        if self.res_regsL2 is not None:
            if model.norm() != 0:
                self.regL2_op.forward(False, model, self.res_regsL2)
            else:
                self.res_regsL2.zero()
            if self.dataregsL2 is not None and self.dataregsL2.norm() != 0.:
                self.res_regsL2.scaleAdd(self.dataregsL2, 1., -1.)

        # compute L1 reg residuals
        if self.res_regsL1 is not None:
            if model.norm() != 0. and self.regL1_op is not None:
                self.regL1_op.forward(False, model, self.res_regsL1)
            else:
                self.res_regsL1.zero()

        return self.res


# Non-linear problem classes
class ProblemL2NonLinear(Problem):
    """Non-linear inverse problem of the form 1/2*|f(m)-d|_2"""

    def __init__(self, model, data, op, grad_mask=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of non-linear problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - non-linear operator class; f(m) operator
           grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
           minBound		= [None] - vector class; Minimum value bounds
           maxBound		= [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(ProblemL2NonLinear, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Residual vector
        self.res = data.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting non-linear and linearized operators
        if isinstance(op, pyOp.NonLinearOperator):
            self.op = op
        else:
            raise TypeError("Not provided a non-linear operator!")
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        return

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """Method to return residual vector r = f(m) - d"""
        self.op.nl_op.forward(False, model, self.res)
        # Computing f(m) - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector g = F'r = F'(f(m) - d)"""
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing F'r = g
        self.op.lin_op.adjoint(False, self.grad, res)
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = Fdm"""
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing Fdm = dres
        self.op.lin_op.forward(False, dmodel, self.dres)
        return self.dres

    def objf(self, res):
        """Method to return objective function value 1/2|f(m)-d|_2"""
        val = res.norm()
        obj = 0.5 * val * val
        return obj


class ProblemL2NonLinearReg(Problem):
    """
       Linear inverse problem regularized of the form
            1/2*|f(m)-d|_2 + epsilon^2/2*|Am - m_prior|_2
                or with a non-linear regularization
            1/2*|f(m)-d|_2 + epsilon^2/2*|g(m) - m_prior|_2
    """

    def __init__(self, model, data, op, epsilon, grad_mask=None, reg_op=None, prior_model=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of non-linear regularized problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - non-linear operator class; f(m) operator
           epsilon      = [no default] - float; regularization weight
           grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
           reg_op       = [Identity] - non-linear/linear operator class; g(m) regularization operator
           prior_model  = [None] - vector class; Prior model for regularization term
           minBound		= [None] - vector class; Minimum value bounds
           maxBound		= [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(ProblemL2NonLinearReg, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting a prior model (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            Id_op = pyOp.IdentityOp(self.model)
            reg_op = pyOp.NonLinearOperator(Id_op, Id_op)
        # Checking if space of the prior model is constistent with range of regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior model space no constistent with range of regularization operator")
        # Setting non-linear and linearized operators
        if not isinstance(op, pyOp.NonLinearOperator):
            raise TypeError("Not provided a non-linear operator!")
        # Setting non-linear stack of operators
        self.op = pyOp.VstackNonLinearOperator(op, reg_op)
        self.epsilon = epsilon  # Regularization weight
        # Residual vector (data and model residual vectors)
        self.res = self.op.nl_op.range.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
        return

    def __del__(self):
        """Default destructor"""
        return

    def estimate_epsilon(self, verbose=False, logger=None):
        """Method returning epsilon that balances the two terms of the objective function"""
        msg = "Epsilon Scale evaluation"
        if verbose:
            print(msg)
        if logger:
            logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
        # Keeping the initial model vector
        prblm_mdl = self.get_model()
        # Keeping user-predefined epsilon if any
        epsilon = self.epsilon
        # Setting epsilon to one to evaluate the scale
        self.epsilon = 1.0
        prblm_res = self.get_res(prblm_mdl)  # Compute residual arising from the gradient
        # Balancing the two terms of the objective function
        res_data_norm = prblm_res.vecs[0].norm()
        res_model_norm = prblm_res.vecs[1].norm()
        if isnan(res_model_norm) or isnan(res_data_norm):
            raise ValueError("Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s"
                             % (res_data_norm, res_model_norm))
        if res_model_norm == 0.:
            msg = "Trying to perform a linearized step"
            if verbose:
                print(msg)
            prblm_grad = self.get_grad(prblm_mdl)  # Compute first gradient
            # Gradient in the data space
            prblm_dgrad = self.get_dres(prblm_mdl, prblm_grad)
            # Computing linear step length
            dgrad0_res = prblm_res.vecs[0].dot(prblm_dgrad.vecs[0])
            dgrad0_dgrad0 = prblm_dgrad.vecs[0].dot(prblm_dgrad.vecs[0])
            if isnan(dgrad0_res) or isnan(dgrad0_dgrad0):
                raise ValueError("Obtained NaN: gradient-dataspace-norm = %s, gradient-dataspace-dot-residuals = %s"
                                 % (dgrad0_dgrad0, dgrad0_res))
            if dgrad0_dgrad0 != 0.:
                alpha = -dgrad0_res / dgrad0_dgrad0
            else:
                msg = "Cannot compute linearized alpha for the given problem! Provide a different initial model"
                if logger:
                    logger.addToLog(msg)
                raise ValueError(msg)
            # model=model+alpha*grad
            prblm_mdl.scaleAdd(prblm_grad, 1.0, alpha)
            if "bounds" in dir(self):
                self.bounds.apply(prblm_mdl)
            prblm_res = self.resf(prblm_mdl)
            # Recompute the new objective function terms
            res_data_norm = prblm_res.vecs[0].norm()
            res_model_norm = prblm_res.vecs[1].norm()
            # If regularization term is still zero, stop the solver
            if res_model_norm == 0.:
                msg = "Model residual component norm is zero, cannot find epsilon scale! Provide a different initial model"
                if logger:
                    logger.addToLog(msg)
                raise ValueError(msg)
        # Resetting user-predefined epsilon if any
        self.epsilon = epsilon
        epsilon_balance = res_data_norm / res_model_norm
        # Setting default variables
        self.setDefaults()
        self.linear = False
        msg = "	Epsilon balancing the the two objective function terms is: %.2e" % epsilon_balance
        if verbose:
            print(msg)
        if logger:
            logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        return epsilon_balance

    def resf(self, model):
        """
        Method to return residual vector r = [r_d; r_m]:
        r_d = f(m) - d;
        r_m = Am - m_prior or r_m = g(m) - m_prior
        """
        self.op.nl_op.forward(False, model, self.res)
        # Computing r_d = f(m) - d
        self.res.vecs[0].scaleAdd(self.data, 1., -1.)
        # Computing r_m = Am - m_prior
        if self.prior_model is not None:
            self.res.vecs[1].scaleAdd(self.prior_model, 1., -1.)
        # Scaling by epsilon epsilon*r_m
        self.res.vecs[1].scale(self.epsilon)
        return self.res

    def gradf(self, model, res):
        """
        Method to return gradient vector
        g = F'r_d + (epsilon*A'r_m or epsilon*G'r_m)
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # g = epsilon*A'r_m
        self.op.lin_op.ops[1].adjoint(False, self.grad, res.vecs[1])
        self.grad.scale(self.epsilon)
        # g = F'r_d + A'(epsilon*r_m)
        self.op.lin_op.ops[0].adjoint(True, self.grad, res.vecs[0])
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """
        Method to return residual vector
        dres = [F + epsilon * (A or G)]dm
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing Ldm = dres_d
        self.op.lin_op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.epsilon)
        return self.dres

    def objf(self, res):
        """
        Method to return objective function value
        1/2|f(m)-d|_2 + (epsilon^2/2*|Am-m_prior|_2 or epsilon^2/2*|g(m)-m_prior|_2)
        """
        # data term
        val = res.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val * val
        # model term
        val = res.vecs[1].norm()
        self.obj_terms[1] = 0.5 * val * val
        obj = self.obj_terms[0] + self.obj_terms[1]
        return obj


# Variable Projection Problem
class ProblemL2VpReg(Problem):
    """
       Non-linear inverse problem in which part of the model parameters define a quadratic function
       The non-linear component is solved using the variable-projection method (Golub and Pereyra, 1973)
       Problem form: phi(m) = 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2
    """

    def __init__(self, model_nl, lin_model, h_op, data, lin_solver, g_op=None, g_op_reg=None, h_op_reg=None,
                 data_reg=None, epsilon=None, minBound=None, maxBound=None, boundProj=None, prec=None,
                 warm_start=False):
        """
            Constructor for solving a inverse problem using the variable-projection method
            Required arguments:
            model_nl    = [no default] - vector class; Initial non-linear model component of the objective function
            lin_model   = [no default] - vector class; Initial quadritic (Linear) model component of the objective function (will be zeroed out)
            h_op   		= [no default] - Vp operator class; Variable projection operator
            data   		= [no default] - vector class; Data vector
            lin_solver	= [no default] - solver class; Linear solver to invert for linear component of the model
            Optional arguments:
            g_op   		= [None] - non-linear operator class; Fully non-linear additional operator
            g_op_reg   	= [None] - non-linear operator class; Fully non-linear additional operator for regularization term
            h_op_reg	= [None] - Vp operator class; Variable projection operator for regularization term
            data_reg   	= [None] - vector class; Data vector for regularization term
            epsilon 	= [None] - float; Regularization term weight (must be provided if a regularization is needed)
            minBound	= [None] - vector class; Minimum value bounds
            maxBound	= [None] - vector class; Maximum value bounds
            boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
            prec       	= [None] - linear operator class; Preconditioning matrix for VP problem
            warm_start  = [None] - boolean; Start VP problem from previous linearly inverted model
            ####################################################################################################################################
            Note that to save the results of the linear inversion the user has to specify the saving parameters within the setDefaults of the
            linear solver. The results can only be saved on files. To the prefix specified within the lin_solver f_eval_# will be added.
        """
        if not isinstance(h_op, pyOp.VpOperator):
            raise TypeError("ERROR! Not provided an operator class for the variable projection problem")
        # Setting the bounds (if any)
        super(ProblemL2VpReg, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model_nl
        self.dmodel = model_nl.clone()
        self.dmodel.zero()
        # Linear component of the inverted model
        self.lin_model = lin_model
        self.lin_model.zero()
        # Copying the pointer to data vector
        self.data = data
        # Setting non-linear/linear operator
        if not isinstance(h_op, pyOp.VpOperator):
            raise TypeError("ERROR! Provide a VpOperator operator class for h_op")
        self.h_op = h_op
        # Setting non-linear operator (if any)
        self.g_op = g_op
        # Verifying if a regularization is requested
        self.epsilon = epsilon
        # Setting non-linear regularization operator
        self.g_op_reg = g_op_reg
        # Setting non-linear/linear operator
        self.h_op_reg = h_op_reg
        # Setting data term in regularization
        self.data_reg = data_reg
        if self.h_op_reg is not None and self.epsilon is None:
            raise ValueError("ERROR! Epsilon value must be provided if a regularization term is requested.")
        # Residual vector
        if self.epsilon is not None:
            # Creating regularization residual vector
            res_reg = None
            if self.g_op_reg is not None:
                res_reg = self.g_op_reg.nl_op.range.clone()
            elif self.h_op_reg is not None:
                if not isinstance(h_op_reg, pyOp.VpOperator):
                    raise TypeError("ERROR! Provide a VpOperator operator class for h_op_reg")
                res_reg = self.h_op_reg.h_lin.range.clone()
            elif self.data_reg is not None:
                res_reg = self.data_reg.clone()
            # Checking if a residual vector for the regularization term was created
            if res_reg is None:
                raise ValueError("ERROR! If epsilon is provided, then a regularization term must be provided")
            self.res = pyVec.superVector(data.clone(), res_reg)
            # Objective function terms (useful to analyze each term)
            self.obj_terms = [None, None]
        else:
            self.res = data.clone()
        # Instantiating linear inversion problem
        if self.h_op_reg is not None:
            self.vp_linear_prob = ProblemL2LinearReg(self.lin_model, self.data, self.h_op.h_lin, self.epsilon,
                                                     reg_op=self.h_op_reg.h_lin, prior_model=self.data_reg,
                                                     prec=prec)
        else:
            self.vp_linear_prob = ProblemL2Linear(self.lin_model, self.data, self.h_op.h_lin, prec=prec)
        # Zeroing out the residual vector
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        # Linear solver for inverting quadratic component
        self.lin_solver = lin_solver
        self.lin_solver.flush_memory = True
        self.lin_solver_prefix = self.lin_solver.prefix
        self.vp_linear_prob.linear = True
        self.warm_start = warm_start
        return

    def __del__(self):
        """Default destructor"""
        return

    def estimate_epsilon(self, verbose=False, logger=None):
        """Method returning epsilon that balances the two terms of the objective function"""
        if self.epsilon is None:
            raise ValueError("ERROR! Problem is not regularized, cannot evaluate epsilon value!")
        if self.g_op_reg is not None and self.h_op_reg is None:
            # Problem is non-linearly regularized
            msg = "Epsilon Scale evaluation"
            if verbose: print(msg)
            if logger: logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
            # Keeping the initial model vector
            prblm_mdl = self.get_model()
            # Keeping user-predefined epsilon if any
            epsilon = self.epsilon
            # Setting epsilon to one to evaluate the scale
            self.epsilon = 1.0
            prblm_res = self.get_res(prblm_mdl)  # Compute residual arising from the gradient
            # Balancing the two terms of the objective function
            res_data_norm = prblm_res.vecs[0].norm()
            res_model_norm = prblm_res.vecs[1].norm()
            if isnan(res_model_norm) or isnan(res_data_norm):
                raise ValueError("ERROR! Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s" % (
                    res_data_norm, res_model_norm))
            if res_model_norm == 0.0:
                msg = "Model residual component norm is zero, cannot find epsilon scale! Provide a different initial model"
                if (logger): logger.addToLog(msg)
                raise ValueError(msg)
            # Resetting user-predefined epsilon if any
            self.epsilon = epsilon
            epsilon_balance = res_data_norm / res_model_norm
            # Resetting problem
            self.setDefaults()
            msg = "	Epsilon balancing the the two objective function terms is: %s" % (epsilon_balance)
            if verbose: print(msg)
            if logger: logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        elif self.h_op_reg is not None:
            # Setting non-linear component of the model
            self.h_op.set_nl(self.model)
            self.h_op_reg.set_nl(self.model)
            # Problem is linearly regularized (fixing non-linear part and evaluating the epsilon on the linear
            # component)
        return self.vp_linear_prob.estimate_epsilon(verbose, logger)

    def resf(self, model):
        """Method to return residual vector"""
        # Zero-out residual vector
        self.res.zero()
        ###########################################
        # Applying full non-linear modeling operator
        res = self.res
        if self.epsilon is not None: res = self.res.vecs[0]
        # Computing non-linear part g(m) (if any)
        if self.g_op is not None:
            self.g_op.nl_op.forward(False, model, res)
        # Computing non-linear part g_reg(m) (if any)
        if self.g_op_reg is not None:
            self.g_op_reg.nl_op.forward(False, model, self.res.vecs[1])

        ##################################
        # Setting data for linear inversion
        # data term = data - [g(m) if any]
        res.scaleAdd(self.data, -1.0, 1.0)
        # Setting data within first term
        self.vp_linear_prob.data = res

        # regularization data term = [g_reg(m) - data_reg if any]
        if self.data_reg is not None:
            self.res.vecs[1].scaleAdd(self.data_reg, 1.0, -1.0)
        # Data term for linear regularization term
        if "epsilon" in dir(self.vp_linear_prob):
            self.res.vecs[1].scale(-1.0)
            self.vp_linear_prob.prior_model = self.res.vecs[1]

        ##################################
        # Running linear inversion
        # Getting fevals for saving linear inversion results
        fevals = self.get_fevals()
        # Setting initial linear inversion model
        if not self.warm_start:
            self.lin_model.zero()
        self.vp_linear_prob.set_model(self.lin_model)
        # Setting non-linear component of the model
        self.h_op.set_nl(model)
        if self.h_op_reg is not None:
            self.h_op_reg.set_nl(model)
        # Resetting inversion problem variables
        self.vp_linear_prob.setDefaults()
        # Saving linear inversion results if requested
        if self.lin_solver_prefix is not None:
            self.lin_solver.setPrefix(self.lin_solver_prefix + "_feval%s" % fevals)

        # Printing non-linear inversion information
        if self.lin_solver.logger is not None:
            # Writing linear inversion log information if requested (i.e., a logger is present in the solver)
            msg = "NON_LINEAR INVERSION INFO:\n	objective function evaluation\n"
            msg += "#########################################################################################\n"
            self.lin_solver.logger.addToLog(msg + "Linear inversion for non-linear function evaluation # %s" % (fevals))
        self.lin_solver.run(self.vp_linear_prob, verbose=False)
        if self.lin_solver.logger is not None:
            self.lin_solver.logger.addToLog(
                "#########################################################################################\n")
        # Copying inverted linear optimal model
        self.lin_model.copy(self.vp_linear_prob.get_model())
        # Flushing internal saved results of the linear inversion
        self.lin_solver.flush_results()

        ##################################
        # Obtaining the residuals
        if (self.epsilon is not None) and not ("epsilon" in dir(self.vp_linear_prob)):
            # Regularization contains a non-linear operator only
            self.res.vecs[0].copy(self.vp_linear_prob.get_res(self.lin_model))
            self.res.vecs[1].scale(self.epsilon)
        else:
            self.res.copy(self.vp_linear_prob.get_res(self.lin_model))
        return self.res

    def gradf(self, model, res):
        """
           Method to return gradient vector
           grad= [G(m)' + H(m_nl;m_lin)'] r_d + epsilon * [G'(m_nl)' + H'(m_nl;m_lin)'] r_m
        """
        # Zero-out gradient vector
        self.grad.zero()
        # Setting the optimal linear model component and background of the Jacobian matrices
        self.h_op.set_lin_jac(self.lin_model)  # H(_,m_lin_opt)
        self.h_op.h_nl.set_background(model)  # H(m_nl,m_lin_opt)
        if self.h_op_reg is not None:
            self.h_op_reg.set_lin_jac(self.lin_model)  # H'(_,m_lin_opt)
            self.h_op_reg.h_nl.set_background(model)  # H'(m_nl,m_lin_opt)
        if self.g_op is not None:
            self.g_op.set_background(model)  # G(m_nl)
        if self.g_op_reg is not None:
            self.g_op_reg.set_background(model)  # G'(m_nl)
        # Computing contribuition from the regularization term (if any)
        if self.epsilon is not None:
            # G'(m_nl)' r_m
            if self.g_op_reg is not None:
                self.g_op_reg.lin_op.adjoint(False, self.grad, res.vecs[1])
            # H'(m_nl,m_lin_opt)' r_m
            if self.h_op_reg is not None:
                self.h_op_reg.h_nl.lin_op.adjoint(True, self.grad, res.vecs[1])
            # epsilon * [G'(m_nl)' + H'(m_nl,m_lin_opt)'] r_m
            self.grad.scale(self.epsilon)
        res = self.res if self.epsilon is None else self.res.vecs[0]
        # G(m_nl)' r_d
        if self.g_op is not None:
            self.g_op.lin_op.adjoint(True, self.grad, res)
        # H(m_nl,m_lin_opt)' r_d
        self.h_op.h_nl.lin_op.adjoint(True, self.grad, res)
        if self.lin_solver.logger is not None:
            self.lin_solver.logger.addToLog(
                "NON_LINEAR INVERSION INFO:\n	Gradient has been evaluated, current objective function value: %s;\n 	"
                "Stepping!" % (
                    self.get_obj(model)))
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres (Not currently supported)"""
        raise NotImplementedError(
            "ERROR! dresf is not currently supported! Provide an initial step-length value different than zero.")

    def objf(self, res):
        """Method to return objective function value 1/2*|g(m_nl) + h(m_nl)m_lin - d|_2 + epsilon^2/2*|g'(m_nl) + h'(m_nl)m_lin - d'|_2"""
        if "obj_terms" in dir(self):
            # data term
            val = res.vecs[0].norm()
            self.obj_terms[0] = 0.5 * val * val
            # model term
            val = res.vecs[1].norm()
            self.obj_terms[1] = 0.5 * val * val
            obj = self.obj_terms[0] + self.obj_terms[1]
        else:
            val = res.norm()
            obj = 0.5 * val * val
        return obj
