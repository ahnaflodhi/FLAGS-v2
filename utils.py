import random
import torch
import pickle
import os


def constrained_sum(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur.
    """
    divider = []
    while 1 in divider or len(divider) == 0:
        dividers = sorted(random.sample(range(1, total), n - 1))
        divider = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    return divider

def dataset_approve(dataset:'str'):
    if dataset == 'mnist': # Num labels will depend on the class in question
        location = '../data/'
        num_labels = 10
    elif dataset == 'cifar':
        location = '../data/'
        num_labels = 10
    elif dataset == 'fashion':
        location = '../data/'
        num_labels = 10
    return location, num_labels

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
#     print('model size: {:.3f}MB'.format(model_mb))
    return model_mb

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


def save_file(file_list, folder, status, flmode, modename, dataset, skew,  alpha, num_nodes, num_clusters, num_epochs, num_rounds, prop, agg_prop, starttime, regagg, gossagg):
    file_name = f'status_{str(modename).upper()}_{dataset.upper()}_a{str(alpha)}s{str(skew)}_n{str(num_nodes)}_c{str(num_clusters)}_e{str(num_epochs)}_r{str(num_rounds)}_prp{str(prop)}_ap{str(agg_prop)}_rg{str(regagg)}{str(gossagg)}_{starttime}'
    if file_list is not None:
        file_list.append(file_name)
    file_name = os.path.join(folder, file_name)
    saved_set = {}
    if modename != 'sgd':
        saved_set = { 'mode':modename,
                      'avgtrgloss' : flmode.avgtrgloss,
                      'avgtrgacc' : flmode.avgtrgacc,
                      'avgtestloss' : flmode.avgtestloss,
                      'avgtestacc' : flmode.avgtestacc,
                      'cluster_trgloss' : flmode.cluster_trgloss,
                      'cluster_trgacc' : flmode.cluster_trgacc,
                      'cluster_testloss' : flmode.cluster_testloss,
                      'cluster_testacc' : flmode.cluster_testacc,
                      'nodetrgloss' : {node: flmode.nodeset[node].trgloss for node in range(num_nodes)},
                      'nodetrgacc' : {node:flmode.nodeset[node].trgacc for node in range(num_nodes)},
                      'nodetestloss' : {node:flmode.nodeset[node].testloss for node in range(num_nodes)},
                      'nodetestacc' : {node:flmode.nodeset[node].testacc for node in range(num_nodes)},
                      'divergence_dict' : {node:flmode.nodeset[node].divergence_dict for node in range(num_nodes)},
                      'divegence_cov_dict' : {node:flmode.nodeset[node].divergence_conv_dict for node in range(num_nodes)},
                      'divergence_fc_dict' : {node:flmode.nodeset[node].divergence_fc_dict for node in range(num_nodes)},
                      'neighborhood' : {node:flmode.nodeset[node].neighborhood for node in range(num_nodes)},
                      'ranked_nhood' : {node:flmode.nodeset[node].ranked_nhood for node in range(num_nodes)},
                      'node_degree' : {node:flmode.nodeset[node].degree for node in range(num_nodes)}
                      }
    elif modename == 'sgd':
        saved_set = {'avgtrgloss' : flmode.avgtrgloss,
                      'avgtrgacc' : flmode.avgtrgacc,
                      'avgtestloss' : flmode.avgtestloss,
                      'avgtestacc' : flmode.avgtestacc}
                    
    with open(file_name, 'wb') as ffinal:
        pickle.dump(saved_set, ffinal)


def merge_files(target_folder, file_list, inter_files_list):
    saved_state = {}
    mode_list = []
    for file in file_list:
        with open(os.path.join(target_folder, file),  'rb') as f:
            state = pickle.load(f)
        mode_list.append(state['mode'])
        saved_state[state['mode']] = {x:state[x] for x in state.keys() if x != 'mode'}
    saved_state['mode_list'] = mode_list

    file_name = '_'.join(file.split('_')[2:])
    
    with open (os.path.join(target_folder, file_name), 'wb') as final_saved:
        pickle.dump(saved_state, final_saved)
    print('All files Merged')

    for file in file_list:
        os.remove(os.path.join(target_folder, file))
    print('All unmerged final files removed')

    for inter_file in inter_files_list:
        os.remove(inter_file)
    print('All inter_files removed')
