from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, target, FFN_CONFIGS=None):
        super().__init__()
        assert target in ["states", "outputs"]

        if target == "states":
            self.FFN = None
        else: #  target == "outputs"
            # self.FFN = ... TODO determine how this implementation should work...
            print(f"target value '{target}' has not been implemented yet. But will continue...")
            self.FFN = None
            # raise ValueError(f"target value '{target}' has not been implemented yet.")
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias)

    def forward(self, X, h0 = None):
        H, h = self.gru(X) if h0 is None else self.gru(X, h0)
        y_hat = H if self.FFN is None else self.FFN(H)
        return y_hat, h





# import torch
# from torch import nn
# from torch import optim

# import numpy as np

# def init_zeros(hidden_size):
#     return torch.zeros((hidden_size)).cuda(non_blocking=True)

# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, bias=False):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.sqrt_k = np.sqrt(1/hidden_size)
#         # activation functions
#         self.sig = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         # matrices and bias (if no bias, bias is set permanently to zero)
#         self.wz = self.uniform_init(nn.Linear(input_size, hidden_size, bias=False))
#         self.uz = self.uniform_init(nn.Linear(hidden_size, hidden_size, bias=False))
#         self.bz = self.uniform_init(nn.Parameter(init_zeros(hidden_size))) if bias else init_zeros(hidden_size)

#         self.wr = self.uniform_init(nn.Linear(input_size, hidden_size, bias=False))
#         self.ur = self.uniform_init(nn.Linear(hidden_size, hidden_size, bias=False))
#         self.br = self.uniform_init(nn.Parameter(init_zeros(hidden_size))) if bias else init_zeros(hidden_size)

#         self.wh = self.uniform_init(nn.Linear(input_size, hidden_size, bias=False))
#         self.uh = self.uniform_init(nn.Linear(hidden_size, hidden_size, bias=False))
#         self.bh = self.uniform_init(nn.Parameter(init_zeros(hidden_size))) if bias else init_zeros(hidden_size)

#     def forward(self, xt, h_prev):
#         zt    = self.sig(  self.wz(xt) + self.uz(h_prev)    + self.bz )
#         with torch.no_grad():
#             _zt = self._zt(xt, h_prev)
#         assert (zt == _zt).all()
#         rt    = self.sig(  self.wr(xt) + self.ur(h_prev)    + self.br )
#         with torch.no_grad():
#             _rt = self._rt(xt, h_prev)
#         assert (rt == _rt).all()
#         h_hat = self.tanh( self.wh(xt) + self.uh(rt*h_prev) + self.bh )
#         with torch.no_grad():
#             _h_hat = self._h_hat(xt, h_prev)
#         assert (h_hat == _h_hat).all()
#         ht    = (1-zt)*h_prev + zt*h_hat
#         with torch.no_grad():
#             _ht = self._ht(xt, h_prev)
#         return ht, ht
    
#     def uniform_init(self, layer):
#         torch.nn.init.uniform_(layer.weight, -self.sqrt_k, self.sqrt_k)
#         return layer

#     ### These next functions are for convenience for analyzing the GRU ###
#     def _zt(self, xt, h_prev):
#         return self.sig(self.wz(xt) + self.uz(h_prev) + self.bz )
    
#     def _rt(self, xt, h_prev):
#         return self.sig(self.wr(xt) + self.ur(h_prev) + self.br )

#     def _h_hat(self, xt, h_prev):
#         rt = self._rt(xt, h_prev)
#         return self.tanh( self.wh(xt) + self.uh(rt*h_prev) + self.bh)

#     def _ht(self, xt, h_prev, sep=False):
#         zt = self._zt(xt, h_prev)
#         h_hat = self._h_hat(xt, h_prev)
#         old = (1-zt)* h_prev
#         new =    zt * h_hat
#         ht= old + new
#         return old, new if sep else old + new


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, bias=False):
#         super().__init__()
#         self.gru = GRU(input_size, hidden_size, bias)

#     def forward(self, xt, h_prev=None):
#         assert len(xt.shape) == 2
#         ht = h_prev if h_prev != None else init_zeros(self.gru.hidden_size).cuda(non_blocking=True)

#         res = [None] * xt.shape[0]
#         for i in range(xt.shape[0]):
#             output, ht = self.gru(xt[i,:], ht)
#             res[i] = output
#         return torch.stack(res), ht