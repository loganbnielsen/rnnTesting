TRAIN_CONFIGS = {
    "trials": 10,
    "fig_dir": "figures",
    "data_dir": "data",
    "model_dir": "models",
    "metrics_dir": "metrics",
    "lti_file": "03-LTI-AB.pickle",
    "train_steps": 1000,
    "total_steps": 2000,
    "epochs": 2000,
    "target": "outputs", # options are "states"(i.e. x[t+1]) or "outputs" (i.e. y[t+1])
    "base": "all", # "random" => alternating base index. 0 => h_1, 1 => h_2, ... "direct" => no relative spacing
    "init_h": False
}
GRU_CONFIGS = {
    "num_layers": 2,
    "bias": False,
    # note: `input_size` and `hidden_size` are implied by the data file
}