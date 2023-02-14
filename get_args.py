import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type = int, default = 8, help = 'Batch size for the dataset')
    parser.add_argument('-t', type = int, default = 8, help = 'Batch size for the test dataset')
    parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist, cifar10 or fashion')
    parser.add_argument('-n', type = int, default = 40, help='Number of nodes')
    parser.add_argument('-c', type = int, default = 7, help='Number of clusters')
    parser.add_argument('-ser', type = int, default = 3, help='Number of servers')
    parser.add_argument('-emin', type = int, default = 1, help='Min Number of epochs')
    parser.add_argument('-emax', type = int, default = 4, help='Max Number of epochs')
    parser.add_argument('-sim', type = float, default = 0.9, help='Cos Similarity Threshold')
    parser.add_argument('-r', type = int, default = 50, help='Number of federation rounds')
    parser.add_argument('-D', type = int, default = 5, help='Delay induced for Stale Global Models')
    parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
    parser.add_argument('-s', type = int, default = 0, help = ' Skew for Exteme Non-IID distribution. Min: 1 - Max: Num of classes') # Positive skew overrides alpha value
    parser.add_argument('-a', type = float, default = 1.0, help = 'Alpha parameter (+ve only) for Dirichlet based non-IID distribution')
    parser.add_argument('-prop', type = float, default = 0.9, help = 'Proportion of nodes chosen for server aggregation : 0.0-1.0')
    parser.add_argument('-aggprop', type = float, default = 0.9, help = 'Aggregation-Proportion: Proportion of nodes in neighborhood for D2D aggregation : 0.0-1.0')
    parser.add_argument('-model', type = str, default = 'shallow', help = 'Define base model type to run the experiments')
    parser.add_argument('-regagg', type = int, default = 4, help = 'Number of rounds for normal aggregation')
    parser.add_argument('-gossagg', type = int, default = 1, help = 'Number of gossip aggregation rounds')
    
    
    args = parser.parse_args()
    return args