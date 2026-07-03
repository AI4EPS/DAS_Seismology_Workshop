import numpy as np
import pyOperator as pyOp
import pyVector as pyVec
try:
    import pylops
except ImportError:
    import os
    os.system("pip install --user pylops")
    import pylops


class ToPylops(pylops.LinearOperator):
    
    def __init__(self, domain, range, op):
        """
        Cast an Operator to pylops.LinearOperator
        :param op: `pyOperator.Operator` object (or child)
        """
        assert isinstance(op, pyOp.Operator), 'op has to be a pyOperator.Operator'
        self.shape = (range.size, domain.size)
        self.dtype = op.domain.getNdArray().dtype
        self.explicit = False
        
        self.op = op
        self.domain = domain.clone()
        self.range = range.clone()
    
    def _matvec(self, x):
        self.domain.getNdArray()[:] = x.reshape(self.domain.shape).astype(self.dtype)
        self.range.zero()
        self.op.forward(False, self.domain, self.range)
        return self.range.getNdArray().ravel()
    
    def _rmatvec(self, y):
        self.domain.zero()
        self.range.getNdArray()[:] = y.reshape(self.range.shape).astype(self.dtype)
        self.op.adjoint(False, self.domain, self.range)
        return self.domain.getNdArray().ravel()


class FromPylops(pyOp.Operator):
    
    def __init__(self, domain, range, op):
        """
        Cast a scipy LinearOperator to Operator
        :param op: `scipy.sparse.linalg.LinearOperator` class (or child, such as pylops.LinearOperator)
        """
        assert isinstance(op, pylops.LinearOperator), "op has to be a pylops.LinearOperator"
        self.name = op.__str__()
        self.op = op
        
        super(FromPylops, self).__init__(domain, range)
    
    def __str__(self):
        return self.name.replace('<', '').replace('>', '')
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        x = model.getNdArray().copy().ravel()
        y = self.op.matvec(x)
        data.getNdArray()[:] = y.reshape(data.shape)
        if add:
            data.scaleAdd(temp, 1., 1.)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        y = data.getNdArray().copy().ravel()
        x = self.op.rmatvec(y)
        model.getNdArray()[:] = x.reshape(model.shape)
        if add:
            model.scaleAdd(temp, 1., 1.)


if __name__ == '__main__':
    
    shape = (3, 5)
    x = pyVec.vectorIC(np.ones(shape))
    
    # test fromPylops
    d = np.arange(x.size)
    D = FromPylops(x, x, pylops.Diagonal(d))
    # D.dotTest(True)
    y = D * x
    
    # test ToPylops
    S = ToPylops(x, x, pyOp.scalingOp(x, 2.))
    S.op.dotTest(True)
    pylops.utils.dottest(S, x.size, x.size)
    z = S * x.arr.ravel()
    
    print(0)
