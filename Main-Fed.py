import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

#The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys, argparse
import pickle
import time
import gc
from utils import dataset_approve, save_file, model_size
from data_utils import * # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from env_sysmodel import system_model, FL_Modes
from devices import Nodes, Servers
from plots import plot_testacc

parser = argparse.ArgumentParser()
parser.add_argument('-b', type = int, default = 8, help = 'Batch size for the dataset')
parser.add_argument('-t', type = int, default = 8, help = 'Batch size for the test dataset')
parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist, cifar10 or fashion')
parser.add_argument('-n', type = int, default = 40, help='Number of nodes')
parser.add_argument('-c', type = int, default = 7, help='Number of clusters')
parser.add_argument('-ser', type = int, default = 3, help='Number of servers')
parser.add_argument('-e', type = int, default = 1, help='Number of epochs')
parser.add_argument('-r', type = int, default = 30, help='Number of federation rounds')
parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
parser.add_argument('-s', type = int, default = 50, help = ' Shard size for Non-IID distribution')
parser.add_argument('-prop', type = float, default = 1.0, help = 'Proportion of nodes chosen for server aggregation : 0.0-1.0')
parser.add_argument('-aggprop', type = float, default = 1.0, help = 'Aggregation-Proportion: Proportion of nodes in neighborhood for D2D aggregation : 0.0-1.0')
parser.add_argument('-dist', type = str, default = 'niid', help = 'Data distribution mode (IID, non-IID, 1-class and 2-class non-IID: iid, niid, niid1 or niid2.')
args = parser.parse_args()

dataset = args.d
batch_size = args.b
nodes = args.n
clusters = args.c
epochs = args.e
rounds = args.r
overlap_factor = args.o
shards =args.s
dist_mode = args.dist
test_batch_size = args.t
prop = args.prop
agg_prop = args.aggprop
servers = args.ser

modes_list = {'d2d':None, 'chd2d':None, 'hch_d2d': None, 'gossip':None, 'hgossip':None, 'cfl': None, 'sgd' : None}

def D2DFL():
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    location, num_labels, in_ch = dataset_approve(dataset)    
    base_model = Net(num_labels, in_ch, dataset)
    
    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset, location)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist
    if dist == 'iid':
        train_dist = data_iid(traindata, num_nodes)
    elif dist == 'niid':
        train_dist = data_noniid(traindata, num_nodes, shard_size)
    elif dist == 'niid1':
        skew = 1
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    elif dist == 'niid2':
        skew = 2
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    
    # Uniform Test distribution for each node. The testing may be carried out on the entire datset
    test_dist = data_iid(testdata, num_nodes)
    
    # Step 4: Create Environment
    env = system_model(num_nodes, num_clusters, num_servers)
    
    # Create Base Parameter Dictionary for Modes
    base_params = { 'dataset' : dataset, 'num_epochs' : num_epochs, 'num_rounds' : num_rounds, 
                   'num_nodes' : num_nodes, 'dist' : dist, 'base_model' : base_model,'num_labels' : num_labels, 
                   in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 
                   'testdata' : testdata, 'testdata_dist' : test_dist, 'batch_size' : batch_size,
                   'nhood' : env.neighborhood_map, 'env_Lp' : env.Lp, 'num_clusters' : num_clusters,
                   'num_servers': env.num_servers}
    
    d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    hfl_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    chd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    hch_d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': True, 'inter_ch_agg_flg': True}
    gossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hgossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    cfl_flg = {'d2d_agg_flg' : 'CServer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    
    # Step-5: Create Modes and combine mode params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode
    
    for mode in modes.keys():
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes
            modes[mode] = FL_Modes(**mode_params[mode])
            # Check Hierarchical Aggregation Flag
            if modes[mode].hserver_agg_flg == True:
            # Create Hierarchical Servers
                modes[mode].form_serverset(env.num_servers, num_labels, in_ch, dataset)

        elif mode == 'sgd':
            modes[mode] = Servers(0, base_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01)
            sgd_trainloader = DataLoader(traindata, batch_size = batch_size)
            sgd_testloader =  DataLoader(testdata)
            
    ### Step-6: Initiate FL Process
    for rnd in range(num_rounds):
        for mode in modes.keys():
            model_sizes = 0
            print(f'Starting with mode {mode} in round-{rnd}')
            
            if mode != 'sgd':
                ### Move Mode-models to cuda
                for node in modes[mode].nodeset:
                    node.model.to('cuda')
                if hasattr(modes[mode], 'serverset'):
                    for server in modes[mode].serverset:
                        server.model.to('cuda')
                modes[mode].cfl_model.to('cuda')
                
                # Initiate Local Training on models
                modes[mode].update_round()
                
                # Perform Testing on Locally trained/fine-tuned models
                modes[mode].test_round(env.cluster_set)
                
                # Share models with neighbors
                # Add noise / Share partials / 
                
                # Perform Neighborhood analysis and update weights assigned to neighbors
                modes[mode].ranking_round(rnd, mode)
                
                #4-Aggregate from neighborhood  using the weights obtained in the previous step
                print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                if modes[mode].d2d_agg_flg == 'D2D':
                    modes[mode].nhood_aggregate_round(agg_prop)

                elif modes[mode].d2d_agg_flg == 'Random':
                    modes[mode].random_aggregate_round()

                elif modes[mode].d2d_agg_flg == 'CServer':
                    modes[mode].cfl_aggregate_round(prop)
                    
                # 5- Cluster operations: 
                if modes[mode].ch_agg_flg == True:
                    print(f'Entering Cluster Head Aggregation for mode-{mode} in round-{rnd}')
                    for i in range(env.num_clusters):
                        modes[mode].clshead_aggregate_round(env.cluster_heads[i], env.cluster_set[i], agg_prop)

                if modes[mode].inter_ch_agg_flg == True:
                    modes[mode].inter_ch_aggregate_round(env.cluster_heads)
                    
                # Should not be executed for Clustered D2D-FL
                if modes[mode].hserver_agg_flg == True: 
                    print(f'Entering Hierarchical Aggregation for mode-{mode} in round-{rnd}')
                    assigned_nodes = []
                    for i in range(env.num_servers):
                        for cluster_id in env.server_groups[i]:
                            assigned_nodes += env.cluster_set[cluster_id] 
                        modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset, assigned_nodes, prop)

                    #Final Server Aggregation
                    modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)
    
    