from SELFRec import SELFRec
from util.conf import ModelConf
import time

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','MF','FedGNN','FedMF','PerFedRec','PerFedRec_plus']
    ssl_graph_models = ['SGL', 'SimGCL', 'XSimGCL']
    sequential_baselines= []
    ssl_sequential_models = []

    print('=' * 80)
    print('''  _____          ______       _ _____                       
 |  __ \        |  ____|     | |  __ \            _     _   
 | |__) |__ _ __| |__ ___  __| | |__) |___  ___ _| |_ _| |_ 
 |  ___/ _ \ '__|  __/ _ \/ _` |  _  // _ \/ __|_   _|_   _|
 | |  |  __/ |  | | |  __/ (_| | | \ \  __/ (__  |_|   |_|  
 |_|   \___|_|  |_|  \___|\__,_|_|  \_\___|\___|            
                                                        
                                                            ''')
    print('=' * 80)

    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)


    
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--model', type=str, default='PerFedRec_plus')
    parser.add_argument("--dataset", type=str, default='kindle', help='kindle/yelp/gowalla')
    parser.add_argument('--emb', type=str, default='64')
    parser.add_argument('--pretrain_epoch', type=str, default='5')
    parser.add_argument('--noise_scale', type=str, default='0.1')
    parser.add_argument('--clip_value', type=str, default='0.5')
    parser.add_argument('--pretrain_noise', type=str, default='0.1')
    parser.add_argument('--pretrain_nclient', type=str, default='256')

    args = parser.parse_args()

    model = args.model

    print(model)


    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)

    if args.dataset == 'kindle':
        args.dataset = 'kindle_test'
    elif args.dataset == 'yelp':
        args.dataset = 'yelp_test'


    conf.__setitem__('training.set',f'./dataset/{args.dataset}/train.txt')
    conf.__setitem__('valid.set',f'./dataset/{args.dataset}/valid.txt')
    conf.__setitem__('test.set',f'./dataset/{args.dataset}/test.txt')
    conf.__setitem__('embedding.size', args.emb )
    conf.__setitem__('noise_scale', args.noise_scale )
    conf.__setitem__('clip_value', args.clip_value )
    conf.__setitem__('pretrain_noise', args.pretrain_noise )
    conf.__setitem__('pretrain_nclient', args.pretrain_nclient )
    conf.__setitem__('pretrain_epoch', args.pretrain_epoch )

    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
