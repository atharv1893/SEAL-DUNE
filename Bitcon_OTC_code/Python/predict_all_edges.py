import os, sys, random, pickle, math, shutil
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, '/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/Python/data')
os.makedirs('/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/Python', exist_ok=True)
#C:mnt\c\Users\athar\Documents\IIIT-Bangalore\SEAL-master\Bitcon_OTC_code\pytorch_DGCNN\lib\build\dll
os.makedirs('mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/pytorch_DGCNN/lib/build/dll', exist_ok=True)
if os.path.exists('mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/pytorch_DGCNN/lib/build/dll/libgnn.so'):
    shutil.copy('mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/pytorch_DGCNN/lib/build/dl', 'mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/pytorch_DGCNN/lib/build/dl')
# for p in ['/mnt/data/DRIVE/__init__.py','/mnt/data/DRIVE/Python/__init__.py']:
#     open(p,'a').close()
# shutil.copy('/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/Python/util_functions.py','/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/Python/util_functions.py')
# # legacy code passes sets to random.sample; adapt for modern Python
_original_sample = random.sample
def sample_compat(population, k, *, counts=None):
    if isinstance(population, (set, frozenset)):
        population = sorted(population)
    return _original_sample(population, k) if counts is None else _original_sample(population, k, counts=counts)
random.sample = sample_compat
sys.argv = ['predict_all_edges']
import Bitcon_OTC_code.Python.seal_bitcoin as seal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
base='/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code'
prefix='/mnt/c/Users/athar/Documents/IIIT-Bangalore/SEAL-master/Bitcon_OTC_code/Python/output/SEAL_FG_DIRECTION/seal_hop2_15_hidden128_seed21'
meta=pickle.load(open(prefix+'.metadata.pkl','rb'))
seal.set_seed(int(meta['seed']))
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seal.cmd_args.mode='gpu' if device.type=='cuda' else 'cpu'
all_df,node_map=seal.load_bitcoin_alpha(base+'/db/soc-sign-bitcoinalpha.csv')
if len(all_df)!=meta['n_edges'] or node_map!=meta['node_map']:
    raise ValueError('Source CSV does not match checkpoint metadata')
train=pd.read_csv(prefix+'.train_edges.csv',header=None,names=['src','dst','rating','timestamp'])
train['src']=train.src.astype(int); train['dst']=train.dst.astype(int); train['rating']=train.rating.astype(float); train['timestamp']=train.timestamp.astype(int); train['weight']=train.rating/10.0
n_nodes=int(meta['n_nodes'])
A=seal.build_structural_graph(train,n_nodes,directed=True)
directed_weight_map=seal.build_directed_weight_map(train)
use_direction = bool(meta.get('use_direction_features', True))
direction = seal.node_direction_stats(train, n_nodes) if use_direction else None
pair_dim = 8 if use_direction else 0
seal.cmd_args.feat_dim = int(meta['feat_dim'])
node_attr_dim = int(meta.get('node_attr_dim', 6 if use_direction else 0))
model = seal.SEALWeightRegressor(
    int(meta['feat_dim']), pair_dim, int(meta['hidden']), True, 0.5,
    meta.get('output_activation', 'scaled_tanh'), node_attr_dim
)
model.to(device)
model.load_state_dict(torch.load(prefix + '.model.pth', map_location=device, weights_only=True)); model.eval()
all_preds = []
chunk_size = 3000
for start in range(0, len(all_df), chunk_size):
    chunk = all_df.iloc[start:start + chunk_size].reset_index(drop=True)
    graphs = seal.build_graph_dataset(
        chunk, A, int(meta['hop']), meta['max_nodes_per_hop'],
        direction_stats=direction, use_direction_features=use_direction,
        directed_weight_map=directed_weight_map,
        desc=f'Edges {start}-{start+len(chunk)-1}'
    )
    with torch.no_grad():
        inference_batch_size=300
        for j in range(0,len(graphs),inference_batch_size):
            pp,_,_=model(graphs[j:j+inference_batch_size])
            all_preds.extend(pp.detach().cpu().numpy().tolist())
    print(f'Predicted {len(all_preds):,}/{len(all_df):,}',flush=True)
preds=np.asarray(all_preds,dtype=float)
inverse={int(v):int(k) for k,v in node_map.items()}
train_pairs=set(zip(train.src.astype(int),train.dst.astype(int)))
pairs=list(zip(all_df.src.astype(int),all_df.dst.astype(int)))
is_train=np.array([x in train_pairs for x in pairs])
out=pd.DataFrame({'source_original':[inverse[int(x)] for x in all_df.src], 'target_original':[inverse[int(x)] for x in all_df.dst], 'true_rating':all_df.rating.to_numpy(float), 'predicted_rating_continuous':preds, 'predicted_rating_rounded':np.clip(np.rint(preds),-10,10).astype(int), 'absolute_error':np.abs(all_df.rating.to_numpy(float)-preds), 'split_membership':np.where(is_train,'training_edge_in_sample','held_out_edge')})
out_path=base+'/soc_sign_bitcoinalpha_all_edge_predictions.csv'; out.to_csv(out_path,index=False)
def metrics(part):
    yt=part.true_rating.to_numpy(float); yp=part.predicted_rating_continuous.to_numpy(float)
    return {'edges':len(part),'MAE_rating':mean_absolute_error(yt,yp),'RMSE_rating':math.sqrt(mean_squared_error(yt,yp)),'PCC':pearsonr(yt,yp).statistic if len(part)>1 and np.std(yt)>0 and np.std(yp)>0 else float('nan'),'within_±1_rounded_pct':100*np.mean(np.abs(part.predicted_rating_rounded-yt)<=1),'within_±2_rounded_pct':100*np.mean(np.abs(part.predicted_rating_rounded-yt)<=2),'within_±3_rounded_pct':100*np.mean(np.abs(part.predicted_rating_rounded-yt)<=3)}
summary=pd.DataFrame([{'subset':'all_edges_mixed_validity',**metrics(out)},{'subset':'training_edges_in_sample_NOT_valid_evaluation',**metrics(out[out.split_membership=='training_edge_in_sample'])},{'subset':'held_out_edges',**metrics(out[out.split_membership=='held_out_edge'])}])
summary.to_csv(base+'/soc_sign_bitcoinalpha_all_edge_prediction_summary.csv',index=False)
print('\nSaved:',out_path); print(summary.to_string(index=False))
