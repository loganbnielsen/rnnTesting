from RNN import RNN
from tqdm import tqdm

import torch
from torch import optim

import pickle
from os import path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# TODO when target == "output" make the graphs, indices, etc... have a Y on them instead of of H and be called "output".

def tensor_to_cuda(*tensors, non_blocking=True):
    def cuda(t):
        if t is not None:
            t = t.cuda(non_blocking=non_blocking).float()
        return t
    return tuple(cuda(t) for t in tensors)

def rel_space(x,i):
    x = x.squeeze()
    assert len(x.shape) == 2
    return x / x[:,i].reshape(-1,1)

def rel_space_dif(x, y, base):
    return rel_space(x,base) - rel_space(y,base)

def delta_percent(x):
    x = x.squeeze()
    assert len(x.shape) 
    prev = x.roll(shifts=1,dims=0)
    x, prev = x[1:], prev[1:]
    return (x - prev) / prev

def loss_func(y_hat, y_true, base=None, epoch=None):
    y_hat, y_true = y_hat.squeeze(), y_true.squeeze()
    if base == "direct":
        pass
    else:
        dimOut = y_hat.shape[-1]
        if base == "all":
            # TODO put this elsewhere?
            #####
            _rel_space = [None] * dimOut
            for _base in range(dimOut):
                _dif = rel_space_dif(y_hat, y_true, _base)
                _psuedo_mse = _dif.square().mean(axis=0).reshape(-1,1) # now H x 1: differences for each H relative to _base
                _rel_space[_base] = _psuedo_mse
            _rel_space = torch.cat(_rel_space,axis=1) # H x H
            mse = _rel_space[_rel_space != 0].mean() # take the mean of the non-zero values
            return mse
            #####
        elif isinstance(base, int):
            _base = base
        elif base == "random":
            _base = epoch % dimOut
        else:
            raise ValueError(f"Invalid argument base='{base}'." )
    return rel_space_dif(y_hat, y_true, _base).square().mean()

def as_df(dif, name, isState):
    square_dif = dif.square()
    mse = square_dif.mean(axis=0).reshape(-1,1)
    std = square_dif.std(axis=0).reshape(-1,1)

    df = pd.DataFrame(np.column_stack([mse,std]), columns=["mse","std"])

    _pre = "H" if isState else "Y"
    df.index = _pre+(df.index+1).astype(str)
    df = pd.DataFrame(df.stack())
    df.columns = [name]
    return df

def relative_spacing_metrics(_hat, _true, isState):
    _hat, _true = _hat.squeeze(), _true.squeeze()
    dimTarget = _hat.shape[-1]
    _rel_space = [None]*dimTarget
    for _base in range(dimTarget):
        _dif = rel_space_dif(_hat, _true, _base)
        _pre = "Base H" if isState else "Base Y"
        _df = as_df(_dif, _pre+str(_base), isState)
        _df.columns = pd.MultiIndex.from_product([["Relative Spacing"], _df.columns])
        _rel_space[_base] = _df
    _rel_space = pd.concat(_rel_space, axis=1)
    _rel_space = _rel_space.drop('std',axis=0,level=1).droplevel(1) # drop previous mse information
    tmp = _rel_space[_rel_space != 0] # mask self comparison
    _rel_space = pd.DataFrame(pd.concat([tmp.mean(axis=1),
                                        tmp.std(axis=1)],axis=1).rename({0:"mse",1:"std"},
                                        axis=1).stack())\
                             .rename({0:"Relative Spacing"},axis=1)
    return _rel_space

def state_values_metrics(state_hat, state_true, isState):
    state_hat, state_true = state_hat.squeeze(), state_true.squeeze()
    _dif = state_hat - state_true
    _value = as_df(_dif, "Value", isState)
    return _value

def state_delta_percent(state_hat, state_true, isState):
    state_hat, state_true = state_hat.squeeze(), state_true.squeeze()
    _dif = delta_percent(state_hat) - delta_percent(state_true)
    _delta_perc = as_df(_dif, "Delta Percent", isState)
    return _delta_perc

def all_state_metrics(state_hat, state_true, isState):
    state_hat, state_true = state_hat.squeeze(), state_true.squeeze()
    # state values
    _value = state_values_metrics(state_hat, state_true, isState)
    # relative spacing
    _rel_space = relative_spacing_metrics(state_hat, state_true, isState)
    # delta percent
    _delta_perc = state_delta_percent(state_hat, state_true, isState)
    return pd.concat([_value, _rel_space, _delta_perc],axis=1)


class Trainer:
    def __init__(self, TRAIN_CONFIGS, GRU_CONFIGS, FFN_CONFIGS=None):
        self.TRAIN_CONFIGS = TRAIN_CONFIGS
        self.GRU_CONFIGS = self._process_gru_configs(GRU_CONFIGS)
        self.model = RNN(target=TRAIN_CONFIGS['target'],**self.GRU_CONFIGS, FFN_CONFIGS=FFN_CONFIGS)
        self.epochs_trained = 0
        self.trained = False
        # Storage for later
        self.loss = self.val_loss = self.train_y_hat = self.train_y_true = self.val_y_hat = self.val_y_true = None

    def _process_gru_configs(self, GRU_CONFIGS):
        lti = self._load_data_source
        HIDDEN_SIZE = lti.A.shape[-1]
        INPUT_SIZE = 1 if lti.B is None else lti.U.shape[-1]
        GRU_IMPLIED_CONFIGS = {
            "hidden_size": lti.A.shape[-1],
            "input_size": 1 if lti.B is None else lti.U.shape[-1]
        }
        GRU_CONFIGS.update(GRU_IMPLIED_CONFIGS)
        return GRU_CONFIGS

    @property
    def _load_data_source(self):
        data_dir = self.TRAIN_CONFIGS.get("data_dir")
        lti_file = self.TRAIN_CONFIGS.get("lti_file")
        with open(path.join(data_dir,lti_file), "rb") as f:
            lti = pickle.load(f)
        return lti

    @property
    def _load_train_data(self):
        def unsqueeze(*args):
            return (_unsqueeze(M) for M in args)
        def _unsqueeze(M):
            if not M is None:
                M = M.unsqueeze(-2)
            return M
        lti = self._load_data_source
        Y, H, X, h0 = lti.torch
        _Y, _H, _X = unsqueeze(Y, H, X)
        _h0 = None if self.TRAIN_CONFIGS.get("init_h") == False else h0.reshape(self.GRU_CONFIGS["num_layers"],
                                                                             1,
                                                                             self.GRU_CONFIGS["hidden_size"])
        return _Y, _H, _X, _h0

    @property
    def fit(self):
        if self.trained == False:
            # get configs (for readability)
            nEpochs = self.TRAIN_CONFIGS['epochs']
            train_steps = self.TRAIN_CONFIGS['train_steps']
            init_h = self.TRAIN_CONFIGS['init_h']
            base = self.TRAIN_CONFIGS['base']
            # load data
            Y, H, X, h0 = tensor_to_cuda(*self._load_train_data)
            # split data
            if self.TRAIN_CONFIGS['target'] == 'states':
                y_train, y_val = H[:train_steps], H[train_steps:]
            elif self.TRAIN_CONFIGS['target'] == 'outputs':
                y_train, y_val = Y[:train_steps], Y[train_steps:]
            x_train, x_val = X[:train_steps], X[train_steps:]
            # prep model and optimizers
            self.model.cuda()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
            # trian
            loss = [None]*nEpochs
            val_loss = [None]*nEpochs
            pbar = tqdm(total=nEpochs, leave=False)
            for i in range(nEpochs):
                # reset gradient
                optimizer.zero_grad()
                # generate prediction
                y_hat, h_plus1 = self.model(x_train) if not init_h else self.model(x_train, h0)
                y_hat = y_hat.squeeze()
                # calculate loss
                l = loss_func(y_hat, y_train, base=self.TRAIN_CONFIGS['base'], epoch=i)
                loss[i] = l.item()
                # learn from loss
                l.backward()
                optimizer.step()
                scheduler.step(l.item())
                # validate
                with torch.no_grad():
                    val_y_hat, _ = self.model(x_val) if not init_h else self.model(x_val, h_plus1)
                    val_y_hat = val_y_hat.squeeze()
                    l = loss_func(val_y_hat, y_val, base=self.TRAIN_CONFIGS['base'], epoch=i)
                    val_loss[i] = l.item()
                # decorator
                pbar.set_description(f"Loss={loss[i]:.3f}. Val={val_loss[i]:.3f}")
                pbar.update(1)

            pbar.close()
            self.epochs_trained += nEpochs

            self.loss, self.val_loss            = loss, val_loss
            self.train_y_hat, self.train_y_true = y_hat.detach().cpu().squeeze(), y_train.detach().cpu().squeeze()
            self.val_y_hat, self.val_y_true     = val_y_hat.detach().cpu().squeeze(), y_val.detach().cpu().squeeze()
            self.trained=True
        else:
            # this shouldn't ever be reached. It's a safety.
            raise ValueError("Model has already been trained.")
        return (self.loss, self.val_loss), \
               (self.train_y_hat, self.train_y_true), \
               (self.val_y_hat, self.val_y_true)
    
    def pickle_save(self, trial_num):
        p = Trainer._pickle_path(self.TRAIN_CONFIGS, trial_num)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    # TODO remove
    # def _gen_relative_graphs(self, hat, true, dimH, val_begins, trial_num=0, fname_prefix=None, freq=10):
    #     Trainer.gen_relative_graphs(hat, true, dimH, val_begins, trial_num, self.TRAIN_CONFIGS.get("fig_dir"), fname_prefix, freq)

    @staticmethod
    def pickled_exists(TRAIN_CONFIGS, trial_num):
        p = Trainer._pickle_path(TRAIN_CONFIGS, trial_num)
        return path.exists(p)

    @staticmethod
    def _pickle_path(TRAIN_CONFIGS, trial_num):
        name = Trainer.model_name(TRAIN_CONFIGS, trial_num)
        if not np.char.endswith(name,".pickle"):
            name += ".pickle"
        model_dir = TRAIN_CONFIGS.get("model_dir")
        return path.join(model_dir,name)

    @staticmethod
    def _gen_relative_graphs(hat, true, dimOut, val_begins, trial_num, isState, fig_dir=None, fname_prefix=None, freq=10, pause=False):
        val_ends = hat.shape[0]
        palette ={"H1": "C0", "H2": "C1", "H3": "C2",
                  "Y1": "C0", "Y2": "C1", "Y3": "C2"}
        for _base in range(dimOut):
            _dif = rel_space_dif(hat, true, _base)
            df = pd.DataFrame(_dif)
            df = df.drop(_base, axis=1)
            _pre = "H" if isState else "Y"
            df.columns = _pre + (df.columns+1).astype(str)
            df.columns.name = "Hidden States" if isState else "Output Indices"
            df.index.name = "Itteration"
            df = df.stack()
            df.name = "Error"
            df = df.reset_index()
            _df = df[df['Itteration'] % freq == 0]
            plt.axhline(0,color="k", alpha=0.5)
            _hue = "Hidden States" if isState else "Output Indices"
            sns.lineplot(data=_df, x="Itteration", y="Error", hue=_hue, alpha=1, palette=palette)

            plt.title(f"Relative Difference (Base: {_pre}{_base+1})")
            plt.axvspan(val_begins, val_ends, facecolor="0.1", alpha=0.25)
            if not fname_prefix is None and not fig_dir is None:
                fname = fname_prefix+f"-relgraph-{_pre}{_base+1}-trial{trial_num}"
                f = path.join(fig_dir, fname)
                plt.savefig(path.join(fig_dir, fname))
            else:
                print(f"fname_prefix='{fname_prefix}'; fig_dir='{fig_dir}'")
            if pause:
                plt.show()
            else:
                plt.show(block=False)
            plt.clf()

    @staticmethod
    def model_name(TRAIN_CONFIGS, trial_num):
        fprefix = TRAIN_CONFIGS.get("lti_file").split(".")[0]
        name = fprefix + f"-trial{trial_num}"
        return name

    @staticmethod
    def load_trained(TRAIN_CONFIGS, trial_num):
        model_dir = TRAIN_CONFIGS.get("model_dir")
        name = Trainer.model_name(TRAIN_CONFIGS, trial_num)
        if not np.char.endswith(name,".pickle"):
            name += ".pickle"
        with open(path.join(model_dir, name), "rb") as f:
            trainer = pickle.load(f)
        return trainer
        
    
    def gen_relative_graphs(self, trial_num, freq=10, pause=False):
        train_hat, train_true, val_hat, val_true = self.train_y_hat, self.train_y_true, self.val_y_hat, self.val_y_true
        # derived
        dimOut = train_hat.shape[-1]
        val_begins = train_hat.shape[0]
        # combine predictions and true values
        hat = np.concatenate([train_hat, val_hat])
        true = np.concatenate([train_true, val_true])
        # graph
        fprefix = self.TRAIN_CONFIGS.get("lti_file").split(".pickle")[0]

        isState = self.TRAIN_CONFIGS.get("target") == "state"
        Trainer._gen_relative_graphs(hat, true, dimOut, val_begins, trial_num, isState, self.TRAIN_CONFIGS.get("fig_dir"), fprefix, freq=10, pause=False)
        
    @property
    def get_train_test_metrics(self):
        isState = self.TRAIN_CONFIGS.get("target") == "state"

        state_tups = [(self.train_y_hat, self.train_y_true), (self.val_y_hat, self.val_y_true)]
        train, test = [all_state_metrics(state_hat, state_true, isState) for state_hat, state_true in state_tups]

        return train, test

    def save_train_test_metrics(self, trial_num):
        metrics_dir = self.TRAIN_CONFIGS.get("metrics_dir")

        train, test = self.get_train_test_metrics
        _name = Trainer.model_name(self.TRAIN_CONFIGS, trial_num)

        train.to_csv(path.join(metrics_dir, _name+"-train.csv"))
        test.to_csv(path.join(metrics_dir, _name+"-val.csv"))

        return train, test