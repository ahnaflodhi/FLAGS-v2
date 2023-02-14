import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# # #The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys, argparse
import pickle
import time
import gc
from get_args import arg_parser
from utils import dataset_approve, save_file, model_size, merge_files
from data_utils import * # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from env_sysmodel import system_model, FL_Modes
from devices import Nodes, Servers
import torchvision.models as models

args = arg_parser()
dataset = args.d
batch_size = args.b
nodes = args.n
clusters = args.c
epochs_min = args.emin
epochs_max = args.emax
rounds = args.r
overlap_factor = args.o
skew = args.s
test_batch_size = args.t
prop = args.prop
agg_prop = args.aggprop
servers = args.ser
modeltype = args.model
alpha = args.a
stale = args.D
cos_lim = args.sim
regagg = args.regagg
gossagg = args.gossagg

# 'd2d': d2d_flags, 'hd2d': hd2d_flags, 'hfl': hfl_flags, 'chd2d':chd2d_flags, 'intch': intch_flags, 'intchd2d':intchd2d_flags, 'gossip':gossip_flags, 'hgossip':hgossip_flags, 'cfl':cfl_flags, 'sgd':None
# modes_list = {'d2d':None, 'hd2d':None, 'hfl':None, 'chd2d':None, 'intch': None, 'intchd2d': None, 'intch':None, 'gossip' : None, 
                # 'hgossip':None,  'cfl': None, 'sgd' : None}
# modes_list = {'d2d':None, 'd2dsel': None, 'd2dlay':None, 'd2dstalemodel': None}
# modes_list = {'d2dstalemodel':None, 'd2dlay':None, 'd2dsel':None, 'd2dstalelay':None, 'd2d':None}
modes_list = {'d2dsel':None}

def D2DFL(model_type, dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_servers, num_rounds, min_epochs, max_epochs, alpha, skew, 
            overlap, prop, agg_prop, stale_lim, cos_lim, regagg, gossagg):
    
    files_list = [] # Record of all results saved. Used to merge into one file the end.
    inter_files_list = []

    # Step 1: Define parameters for the environment, dataset and dataset distribution
    starttime = time.strftime("%Y%m%d_%H%M")
    location, num_labels = dataset_approve(dataset)
    
    if model_type == 'shallow':
        if dataset == 'mnist' or dataset == 'fashion':
            in_ch = 1
        elif dataset  == 'cifar':
            in_ch = 3
        base_model = Net(num_labels, in_ch, dataset)
    
    else:
        in_ch = 3
        if model_type == 'vgg16':
            base_model = models.vgg16(num_classes = num_labels)
    
        elif model_type ==  'alexnet':
            base_model = models.alexnet()
            base_model.classifier[6] = nn.Linear(in_features=4096, out_features= num_labels, bias=True)
            
        elif model_type == 'resnet':
            base_model = models.resnet18()
            base_model.fc = nn.Linear(in_features=512, out_features=num_labels, bias=True)
    
    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset, location, in_ch)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist
    if skew == 0:
        train_dist = data_noniid(traindata, num_labels, num_nodes, alpha)
    else:
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew)
    
    # Uniform Test distribution for each node. The testing may be carried out on the entire datset
    test_dist = data_iid(testdata, num_labels, num_nodes)
    
    # Step 4: Create Environment
    env = system_model(num_nodes, num_clusters, num_servers)
    
    # Create Base Parameter Dictionary for Modes
    base_params = { 'dataset' : dataset, 'num_epochs' : max_epochs, 'num_rounds' : num_rounds, 
                   'num_nodes' : num_nodes, 'base_model' : base_model,'num_labels' : num_labels, 
                   'in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 
                   'testdata' : testdata, 'testdata_dist' : test_dist, 'batch_size' : batch_size,
                   'nhood' : env.neighborhood_map, 'env_Lp' : env.Lp, 'num_clusters' : num_clusters,
                   'num_servers': env.num_servers}
    
    # Flags will only be used for the modes defined at the outset of the main file
    # Prospective Flags: IF XX_Op: False, corressponding Selective_XX: False (XX -> D2D, CH, HSer) 
    # FLAGS: Device_Op: Selective_D2D, CH_Op, Selective-CH, HSer_Op, Selective_HSer
    # Device_Op: False, D2D, Random, Cls_D2D, Cls_Random, CServer
    # CH_Op: False, CH_agg, iCH, iCH_Random
    # HSer_Op: False, Default, iHSer, iHSer_Random
    # Selective_XX: False, defautl, model_sel, layer_sel, stale_global, current_global (XX -> D2D, CH, HSer) 

    # Flags will only be used for the modes defined at the outset of the main file
    d2d_flags = {'Device_Op' :'D2D', 'Selective_D2D':'default', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':False, 'Selective_HSer':False}
    d2dsel_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'model_sel', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':False, 'Selective_HSer':False}
    d2dlay_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'layer_sel', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':False, 'Selective_HSer':False}
    d2dstalemodel_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'stalemodel_global', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':False, 'Selective_HSer':False}
    d2dstalelay_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'stalelay_global', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':False, 'Selective_HSer':False}
    
    hfl_flags = {'Device_Op' : False, 'Selective_D2D': False, 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':'default', 'Selective_HSer': 'default'}
    hd2d_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'default', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':'default', 'Selective_HSer': 'default'}
    hgossip_flags = {'Device_Op' : 'Random', 'Selective_D2D': 'default', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op':'default', 'Selective_HSer': 'default'}

    chd2d_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'default', 'CH_Op': 'CH_agg', 'Selective_CH': 'default', 'HSer_Op': False, 'Selective_HSer': False}
    intch_flags = {'Device_Op' : False, 'Selective_D2D': False, 'CH_Op': 'iCH', 'Selective_CH': 'default', 'HSer_Op': False, 'Selective_HSer': False}
    intchd2d_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'default', 'CH_Op': 'iCH', 'Selective_CH': 'default', 'HSer_Op': False, 'Selective_HSer': False}
    intchgossip_flags = {'Device_Op' : 'D2D', 'Selective_D2D': 'default', 'CH_Op': 'iCH_Random', 'Selective_CH': 'default', 'HSer_Op': False, 'Selective_HSer': False}
        
    gossip_flags = {'Device_Op' : 'Random', 'Selective_D2D': 'default', 'CH_Op': False, 'Selective_CH': False, 'HSer_Op': False, 'Selective_HSer': False}
    cfl_flags = {'Device_Op' : 'CSer', 'Selective_D2D': 'default', 'CH_Op': False, 'selective_CH': False, 'Hser_Op': False, 'Selective_HSer': False}

    #d2dgossip_flags = {'d2d_agg_flg' : 'Dual', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    flag_dict = {'d2d': d2d_flags, 'd2dsel':d2dsel_flags, 'd2dlay':d2dlay_flags, 'd2dstalemodel': d2dstalemodel_flags, 'd2dstalelay': d2dstalelay_flags, 
                'hd2d': hd2d_flags, 'hfl': hfl_flags, 'hgossip':hgossip_flags, 
                'chd2d':chd2d_flags, 'intch': intch_flags, 'intchd2d':intchd2d_flags, 'intchgossip':intchgossip_flags,
                'gossip':gossip_flags,'cfl':cfl_flags}
    
    # Step-5: Create Modes and combine mode params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode

        
    modelist = [mode for mode in modes.keys()]
    agg_cycle = regagg + gossagg # Cycle for alternating between Regular and Gosssip aggregation

    for mode in modelist:
        file_args = {'folder': None, 'status': None, 'flmode': None, 'modename': mode, 'dataset':dataset, 'skew':skew, 'alpha':alpha, 'num_nodes':num_nodes, 
                    'num_clusters':num_clusters, 'num_epochs':max_epochs, 'num_rounds': num_rounds, 'starttime':starttime, 'prop':prop, 
                    'agg_prop' : agg_prop, 'regagg':regagg, 'gossagg':gossagg}
        
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes
            modes[mode] = FL_Modes(**mode_params[mode])
            # Check Hierarchical Aggregation Flag
            if modes[mode].HSer_Op != False:
            # Create Hierarchical Servers
                modes[mode].form_serverset(env.num_servers, num_labels, in_ch, dataset)
            
            # Start Federation Protocol
            for rnd in range(num_rounds):
                # For usage with stale global models    
                ref_rnd = (rnd // stale_lim) * stale_lim

                # Initiate Local Training on models
                print(f'Update Round {rnd}- Mode {mode}') 
                modes[mode].update_round(min_epochs, max_epochs)
            
                # # Perform Testing on Locally trained/fine-tuned models
                modes[mode].test_round(env.cluster_set)

                # Share partials / layers

                #4-Aggregate from neighborhood  using the weights obtained in the previous step
                print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                modes[mode].cfl_aggregate_round(rnd, prop, modes[mode].D2D_Op)

                # Sim _Check round
                # modes[mode].sim_check(ref_rnd)
                
                # Entering D2D Aggregation
                if modes[mode].D2D_Op != False:       
                    # If D2D Aggregation, choose selectivity criterion             
                    if modes[mode].D2D_Op == 'D2D':
                        print(f'Entering D2D Aggregation for {rnd}-rnd', '\n')
                        if modes[mode].Selective_D2D  == 'default':
                            modes[mode].nhood_aggregate_round(agg_prop)
                        elif modes[mode].Selective_D2D == 'model_sel':
                            modes[mode].selective_aggregate_round(cos_lim)
                        elif modes[mode].Selective_D2D == 'layer_sel':
                            modes[mode].layerwise_aggregate_round(cos_lim)
                        elif modes[mode].Selective_D2D == 'stalemodel_global':
                            modes[mode].selective_aggregate_round(cos_lim, ref_rnd = ref_rnd)
                        elif modes[mode].Selective_D2D == 'stalelay_global':
                            modes[mode].layerwise_aggregate_round(cos_lim, ref_rnd = ref_rnd)
                    elif modes[mode].d2d_op == 'Random':
                        modes[mode].random_aggregate_round()

                # 5- Cluster operations: 
                if modes[mode].CH_Op != False:
                    print(f'Entering Cluster Head Aggregation for mode-{mode} in round-{rnd}')
                    if modes[mode].CH_Op == 'CH_agg':
                        for i in range(env.num_clusters):
                            modes[mode].clshead_aggregate_round(env.cluster_heads[i], env.cluster_set[i], agg_prop) # Added 0.9 instead of agg_prop

                    elif modes[mode].CH_Op == 'iCH':
                        modes[mode].inter_ch_aggregate_round(env.cluster_heads, 5)
                    
                    elif modes[mode].ch_opCH_Op == 'iCH_Random':
                        pass
                    
                # Should not be executed for Clustered D2D-FL
                if modes[mode].HSer_Op != False: 
                    print(f'Entering Hierarchical Aggregation for mode-{mode} in round-{rnd}')
                    for i in range(env.num_servers):
                        assigned_nodes = []
                        for cluster_id in env.server_groups[i]:
                            assigned_nodes += env.cluster_set[cluster_id] 
                        modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset, assigned_nodes, modes[mode].weights, agg_prop)

                    #Final Server Aggregation
                    modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)

                # Interim Record
                if rnd % 5 == 0:
                    file_args['folder'] = './'
                    file_args['status'] = 'inter'
                    file_args['flmode'] = modes[mode]
                    save_file(inter_files_list, **file_args)
                        
            file_args['folder'] = './Results'
            file_args['status'] = 'Final'
            file_args['flmode'] = modes[mode]
            save_file(files_list, **file_args)
                     
            for node in modes[mode].nodeset:
                node.model.to('cpu')

            if hasattr(modes[mode], 'serverset'):
                for server in modes[mode].serverset:
                    server.model.to('cpu')
            modes[mode].cfl_model.to('cpu')
            torch.cuda.empty_cache()
                        
            del modes[mode]            
            gc.collect()

        elif mode == 'sgd':
            modes[mode] = Servers(0, base_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01, momentum = 0.9)
            sgd_lambda = lambda epoch: 0.004 * epoch
#             sgd_scheduler = torch.optim.lr_scheduler.LambdaLR(sgd_optim, lr_lambda= sgd_lambda)
            sgd_trainloader = DataLoader(traindata, batch_size = 32)
            sgd_testloader =  DataLoader(testdata)
            for rnd in range(num_rounds):
                node_update(modes[mode].model.cuda(), sgd_optim, sgd_trainloader, modes[mode].avgtrgloss,
                            modes[mode].avgtrgacc, 0, num_rounds)
                loss, acc = test(modes[mode].model, sgd_testloader)
                modes[mode].avgtestloss.append(loss)
                modes[mode].avgtestacc.append(acc)
            
            file_args['folder'] = './Results'
            file_args['status'] = 'Final'
            file_args['flmode'] = modes[mode]
            save_file(files_list, **file_args)
                        
            del modes[mode]            
            gc.collect()

    merge_files('./Results', files_list, inter_files_list)
            
            
## Main Function
if __name__ == "__main__":
#     dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist
# def D2DFL(model_type, dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_servers, num_rounds, min_epochs, max_epochs, alpha, skew, 
            # overlap, prop, agg_prop, stale_lim, cos_lim, regagg, gossagg):
        mode_state = D2DFL(modeltype, dataset, batch_size, test_batch_size, modes_list,  nodes, clusters, servers, rounds, epochs_min, epochs_max, alpha, skew, 
        overlap_factor, prop, agg_prop, stale, cos_lim, regagg, gossagg)