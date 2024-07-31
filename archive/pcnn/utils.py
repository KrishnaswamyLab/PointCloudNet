import torch
import torch_geometric
import itertools
import os
import yaml
import pandas as pd

def compute_sparse_diffusion_operator(b):
    #L_sparse = torch_geometric.utils.get_laplacian(b.edge_index)
    A_sparse = torch_geometric.utils.to_torch_coo_tensor(b.edge_index, b.edge_weight)
    D = A_sparse.sum(1).to_dense()
    Dinv = torch.sparse.spdiags(1/D.squeeze(), offsets = torch.zeros(1).long(),shape = (len(D),len(D)))
    P_sparse = torch.sparse.mm(Dinv,A_sparse)
    return P_sparse


def get_scattering_indices(n):
    return [(i,j) for (i,j) in itertools.combinations(range(n),2) if i<j]


#RESULTS UTILS


def get_experiment_config(model_name, run_name):
    """
    Get the config of an experiment (with run name) - multirun
    """
    file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,"multirun.yaml") #if sweep params were provided in command line
    if not os.path.exists(file_path):    #if a sweeper config was used
        file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,"0",".hydra","hydra.yaml")
    
    with open(file_path, 'r') as file:
            exp_config = yaml.safe_load(file) 
    return exp_config

def get_sweep_variables(exp_config):
    """
    Get the variables that were swept in the experiment
    """
    if exp_config["hydra"]["sweeper"]["params"] is not None: # this is used if a sweeper config was used
        variables = [(k,[v_.strip() for v_ in v.split(",")]) for k,v in exp_config["hydra"]["sweeper"]["params"].items()]
    else:
        variables = [(s.split("=")[0],s.split("=")[1].split(",")) for s in exp_config["hydra"]["overrides"]["task"]]
    variables = {v[0]:v[1] for v in variables if len(v[1]) > 1}
    return variables

def get_all_results_exp(model_name, run_name, sweep_variables):
    dir_name = os.path.join("../logs/experiments/multiruns",model_name,run_name)
    run_ids = [ f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name,f))]

    df_list = []
    for run_id in run_ids:
        run_results = get_extended_run_results(model_name, run_name, run_id, sweep_variables)
        if run_results is not None:
            df_list.append(run_results)

    df_results = pd.concat(df_list)
    return df_results

def get_extended_run_results(model_name, run_name, run_id, sweep_variables):
    run_config = get_run_config(model_name, run_name, run_id)

    variables_from_run = extract_variables_from_run(sweep_variables, run_config)

    run_results = get_run_results(model_name, run_name, run_id)

    if run_results is not None:
        for var in variables_from_run.keys():
            run_results[var] = variables_from_run[var]

    return run_results

def extract_variables_from_run(variables, run_config):
    """
    Extract the values of the variables that were swept in the experiment, from the config of a specific run
    """
    extracted_variables = {}
    for conf_var in variables.keys():
        conf_value = None
        if conf_var == "data":
            splitted_conf_var = ["dataset_name"]
        else:
            splitted_conf_var = conf_var.split(".")
        for conf_ in splitted_conf_var:
            if conf_value is None:
                conf_value = run_config[conf_]
            else:
                conf_value = conf_value[conf_]
        ### THIS IS A FIX TO DISTINGUISH BETWEEN SWISS ROLL DATASETS - REMOVE IN NEXT ITERATION ---
        if conf_var == "data":
            if conf_value == "tree":
                if run_config["data"]["n_dim"] == 30:
                    conf_value = "tree_high"
        ### ---------------------------------------------------------------------------------------
        
        extracted_variables[conf_var] = conf_value
    return extracted_variables

def get_run_config(model_name, run_name, run_id):
    """
    Get the config of a specific run (with run id)
    """
    file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,run_id,".hydra","config.yaml")
    with open(file_path, 'r') as file:
        run_config = yaml.safe_load(file)
    return run_config

def get_run_results(model_name, run_name, run_id):
    """
    Get the results of a specific run (with run id)
    """
    dir_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,run_id)
    pkl_files = [f for f in os.listdir(dir_path) if "pkl" in f]
    if len(pkl_files)!=1:
        print("No PKL file found for {model_name} {run_name} {run_id}".format(model_name=model_name, run_name=run_name, run_id=run_id))
        print("Config for this run : ")
        print(get_run_config(model_name, run_name, run_id))
        return None
    else:
        pkl_file = pkl_files[0]
        return pd.read_pickle(os.path.join(dir_path,pkl_file))
    

def get_best(df,sweep_variables):
    metric = "val_acc"
    test_metric = "test_acc"

    df_m = df.groupby(list(sweep_variables.keys()))[[metric,test_metric]].mean().reset_index()
    df_s = df.groupby(list(sweep_variables.keys()))[[metric,test_metric]].std().reset_index()

    best_ix = df_m.loc[df_m[metric].argmax()]

    df_m_best = df_m.loc[[df_m[metric].argmax()]].copy()
    df_s_best = df_s.loc[[df_m[metric].argmax()]].copy()

    df_s_best.rename(columns = {x:x+"_std" for x in df_s_best.columns if x not in sweep_variables.keys()}, inplace = True)

    df_best = pd.merge(df_m_best,df_s_best,how = "inner", on = list(sweep_variables.keys()))
    return df_best