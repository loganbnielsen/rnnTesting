import numpy as np
import pandas as pd
import torch

import re

class LTI:
    def __init__(self, A, B, C, D, U, x0, steps, *args, **kwargs):
        B, C, D, U = self._preprocess(A, B, C, D, U, x0, steps, *args,**kwargs)
        X, Y = self._gen_data(A, B, C, D, U, x0, steps, *args, **kwargs)
        # constants
        self.A = A # always exists
        self.B = None if (B == 0).all() else B
        self.C = None if (C == 0).all() else C
        self.D = None if (D == 0).all() else D
        # time series
        self.U = None if (U == 0).all() else U
        self.X = X # always exists
        self.Y = None if ((Y == 0 ) | np.isnan(Y)).all() else Y

    def _preprocess(self, A, B, C, D, U, x0, steps, *args, **kwargs):
        steps = steps if not steps is None else len(U)
        # transfer matrices
        B = B if not B is None else np.zeros(1)
        C = C if not C is None else np.zeros(A.shape)
        D = D if not D is None else np.zeros(B.shape)
        # work out U
        u_dim = B.shape[1]
        U = U if U is not None else [None]*steps
        if (_steps_remaining := steps - len(U) ) > 0: # fill None if U is not long enough
            for i in range(_steps_remaining):
                U.append(None)
        assert not _steps_remaining < 0 # error if U is longer then steps
        U = [u if not u is None else np.zeros(u_dim) for u in U] # replace `None` with 0's
        U.append([np.NaN]*U[-1].shape[0])
        U = np.stack(U)
        return B, C, D, U


    def _gen_data(self, A, B, C, D, U, x0, steps, *args, **kwargs):
        X = [x0, *[None]*steps] # always one more X then U: x[t+1] = Ax[t]+Bu[t]
        Y = [None]*(steps+1) # Note that x[0] -> y[1] so y[0] DNE
        for i in range(steps):
            # print(A, X[i], B, U[i])
            # print(A.shape, X[i].shape, B.shape, U[i].shape)
            X[i+1] = A @ X[i] + B @ U[i]
            Y[i+1] = C @ X[i] + D @ U[i]
        Y[0] = np.array([np.NaN]*len(Y[-1]))
        return np.stack(X), np.stack(Y)

    @property
    def df(self):
        name_data =[(name, numpy2d) for name, numpy2d in
                                        [("Y",self.Y), ("X",self.X), ("U",self.U)]
                                    if not numpy2d is None]
        col_names = list()
        for name, numpy2d in name_data:
            for j in range(numpy2d.shape[1]):
                col_names.append(name+str(j))
        data = np.column_stack([numpy2d for _, numpy2d in name_data]) 

        return pd.DataFrame(data, columns=col_names)

    @property
    def torch(self, target="X"):
        """
            Returns Y, X, U in RNN indexed form:
                (e.g) Given a row  y_{t}, x_{t}, u_{t}
                      Let the RNN be f() then x_{t} = f(u_{t})
                      Whereas LTI indexing would be x_{t+1} = f(u_{t})
        """
        if not target in ["X", "Y"]:
            raise ValueError(f"Invalid target: {target}")
        df = self.df

        res = dict()
        column_prefix = ["U", "X", "Y"]
        for pre in column_prefix:
            col_names = df.columns[df.columns.str.match(pre, flags=re.IGNORECASE)]
            if col_names.size > 0:
                res[pre] = df[col_names]
                if pre == "U":
                    res[pre] = torch.tensor(df[col_names].shift().dropna().values)
                else:
                    res[pre] = torch.tensor(df.loc[1:None,col_names].values)
        x0 = torch.tensor(self.X[0])
        return res.get("Y"), res.get("X"), res.get("U"), x0

    def __repr__(self):
        return "\n".join([f"A =\n{self.A}",
                          f"B =\n{self.B}",
                          f"C = \n{self.C}",
                          f"D =\n{self.D}"])
    def __str__(self):
        return self.__repr__()



# @pd.api.extensions.register_dataframe_accessor("ts")
# class Functions:
#     def __init__(self, pandas_obj):
#         self._validate(pandas_obj)
#         self._obj = pandas_obj.sort_values("date")

#     @staticmethod
#     def _validate(obj):
#         _required_columns = ["date","ticker"]
#         for _col in _required_columns:
#             if _col not in obj.columns:
#                 raise AttributeError(f"Must have '{_col}'.")

#     def _add_cols(self, _delta_perc_cols):
#         cols = _delta_perc_cols.columns
#         self._obj[cols] = _delta_perc_cols
#         return self._obj


#     def create_delta_perc_vars(self, columns, lag=1, join=False, merge_date=False):
#         _vars = np.array(columns)
#         _lagged_cols = self.create_lagged_vars(columns, lag)
#         _delta_perc_cols = (self._obj[columns] -_lagged_cols.values) / _lagged_cols.values * 100
#         _delta_perc_cols.columns = np.char.add(f"delta{lag}_perc_" ,_vars)
#         res = self._add_cols(_delta_perc_cols) if join else _delta_perc_cols
#         if merge_date:
#             res['date'] = self._obj['date']
#         return res

#     def create_lagged_vars(self, columns, lag=1, join=False, merge_date=False):
#         _vars = np.array(columns)
#         _lagged_cols = self._obj.groupby("ticker")[_vars].shift(lag)
#         _lagged_cols.columns = np.char.add("lag_", _vars)
#         res = self._add_cols(_lagged_cols) if join else _lagged_cols
#         if merge_date:
#             res['date'] = self._obj['date']
#         return res

#     def split(self, ratio=[3/4, 1/8, 1/8]):
#         assert sum(ratio) == 1
#         splits = np.array(ratio)
#         obs = len(self._obj) * splits
#         cuts = np.cumsum(obs).astype(int)
#         frames = []
#         prev=None
#         for end in cuts:
#             frames.append(self._obj.iloc[prev:end])
#             prev = end
#         return frames