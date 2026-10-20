import os, sys, json, time, math, random, argparse, pathlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N, D, Z = 737280, 4096, 10

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

class BetaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4096,1200),nn.ReLU(),nn.Linear(1200,1200),nn.ReLU())
        self.mu = nn.Linear(1200,10)
        self.logvar = nn.Linear(1200,10)
        self.decoder = nn.Sequential(nn.Linear(10,1200),nn.Tanh(),nn.Linear(1200,1200),nn.Tanh(),
                                     nn.Linear(1200,1200),nn.Tanh(),nn.Linear(1200,4096))
    def encode(self,x):
        h = self.encoder(x.reshape(-1,D))
        return self.mu(h), self.logvar(h)
    def decode(self,z):return self.decoder(z)
    def forward(self,x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5*lv)*torch.randn_like(mu)
        return self.decode(z), mu, lv

def initialize(m):
    if isinstance(m,nn.Linear):
        bound = 1/math.sqrt(m.in_features)
        nn.init.uniform_(m.weight,-bound,bound)
        nn.init.uniform_(m.bias,-bound,bound)

def loss_parts(logits, x, mu, lv, beta):
    reconstruction = F.binary_cross_entropy_with_logits(logits,x.reshape(-1,D),reduction='none').sum(dim=1)
    kl = (-0.5*(1+lv-mu.square()-lv.exp())).sum(dim=1)
    return (reconstruction + beta*kl).mean(), reconstruction.mean(), kl.mean()

def save_checkpoint(path, epoch, model, opt, shufgen, beta, seed, history, mode):
    record = dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                  shufgen_state=shufgen.get_state(), cpu_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state(), beta=beta, seed=seed,
                  history=history, data_mode=mode, batch_size=128)
    tmp = path.with_suffix('.tmp')
    torch.save(record,tmp)
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    for key, typ in [('data',str),('out',str),('beta',float),('seed',int),('epochs',int),('batch',int)]:
        ap.add_argument('--'+key, required=True, type=typ)
    ap.add_argument('--benchmark',action='store_true')
    args=ap.parse_args()
    assert args.batch==128, 'Protocol fixed at batch=128'
    assert torch.cuda.device_count()==1, 'Worker must see exactly one GPU'
    device=torch.device('cuda:0'); seed_everything(args.seed)
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model=BetaVAE().to(device);model.apply(initialize)
    optimizer=torch.optim.Adagrad(model.parameters(),lr=0.01,lr_decay=0,
                                   weight_decay=0,initial_accumulator_value=0,eps=1e-10)
    shufgen=torch.Generator(device=device).manual_seed(args.seed+1)
    # Load original binary images, preserving their canonical Cartesian order.
    print('Loading complete dSprites...',flush=True)
    with np.load(args.data,allow_pickle=False) as d:
        arr=d['imgs']
        if arr.shape!=(N,64,64) or arr.dtype not in (np.uint8,np.bool_):
            raise ValueError('Unexpected image data')
        imgs=torch.from_numpy(np.ascontiguousarray(arr.reshape(N,D),dtype=np.uint8))
    mode='cpu_uint8'
    if torch.cuda.mem_get_info()[0] > imgs.numel()+2*1024**3:
        try:
            imgs=imgs.to(device)
            mode='gpu_uint8'
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mode='cpu_uint8'
    print('Image cache:',mode,'|',imgs.shape,'|',imgs.dtype,flush=True)
    ckpt=out/'last.pt'; history=[]; start=0
    if not args.benchmark and ckpt.exists():
        try: previous=torch.load(ckpt,map_location=device,weights_only=False)
        except TypeError:previous=torch.load(ckpt,map_location=device)
        if previous['beta']!=args.beta or previous['seed']!=args.seed or previous['batch_size']!=args.batch:
            raise RuntimeError('Checkpoint configuration mismatch')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        shufgen.set_state(previous['shufgen_state'])
        torch.set_rng_state(previous['cpu_rng']);torch.cuda.set_rng_state(previous['cuda_rng'])
        start=previous['epoch'];history=previous['history']
        print('Resuming after epoch',start,flush=True)
    if args.benchmark:
        # Warm-up plus timed train steps on current hardware; does NOT save a checkpoint.
        torch.cuda.synchronize()
        durations=[]
        for i in range(25):
            ids=torch.randint(0,N,(args.batch,),device=device,generator=shufgen)
            x=imgs[ids].to(device,dtype=torch.float32)
            t=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits,mu,lv=model(x);total,_,_=loss_parts(logits,x,mu,lv,args.beta)
            total.backward();optimizer.step()
            torch.cuda.synchronize()
            if i>=5:durations.append(time.perf_counter()-t)
        result=dict(beta=args.beta,seconds_per_step=float(np.median(durations)),
                    estimated_train_hours=args.epochs*math.ceil(N/args.batch)*float(np.median(durations))/3600,
                    cache_mode=mode,steps_measured=len(durations))
        (out/'benchmark.json').write_text(json.dumps(result,indent=2))
        print('BENCHMARK',json.dumps(result),flush=True)
        return
    for epoch in range(start,args.epochs):
        model.train(); t0=time.perf_counter()
        order=torch.randperm(N,device=device,generator=shufgen)
        accum=np.zeros(3,dtype=np.float64)
        for start_idx in range(0,N,args.batch):
            ids=order[start_idx:start_idx+args.batch]
            # GPU cache and CPU fallback preserve exact same input samples.
            if mode=='cpu_uint8':
                x=imgs[ids.cpu()].to(device=device,dtype=torch.float32)
            else:
                x=imgs[ids].to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits,mu,lv=model(x);total,rec,kl=loss_parts(logits,x,mu,lv,args.beta)
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss at epoch '+str(epoch+1))
            total.backward();optimizer.step()
            accum += np.array([total.item(),rec.item(),kl.item()])*len(ids)
        elapsed=time.perf_counter()-t0
        row=dict(epoch=epoch+1,objective=accum[0]/N,reconstruction_bce=accum[1]/N,
                 kl=accum[2]/N,epoch_seconds=elapsed,beta=args.beta)
        history.append(row)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print('EPOCH',json.dumps(row),flush=True)
        if epoch+1<=3 or (epoch+1)%5==0 or epoch+1==args.epochs:
            save_checkpoint(ckpt,epoch+1,model,optimizer,shufgen,args.beta,args.seed,history,mode)
            print('CHECKPOINT',epoch+1,flush=True)
    print('COMPLETE',args.epochs,flush=True)

if __name__=='__main__':main()
