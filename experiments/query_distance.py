"""Task-singular Query routing with constrained class-level publication completion."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from angular_distance import Problem,angular_scores
from classification_completion import readout
from clip_benchmark import digest,merge
from conditional_task_distance import diagnose,optimize
from covariance_noise import dataset
from cross_module_distance import extended_parameters,final_hidden,normalized_input
from functional_distance import ClassCompletion
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader,save_json
from task_probe_distance import checked_models,gelu


def query_parameters(model,head):
    p=extended_parameters(model,head);attn=model.model.visual.transformer.resblocks[11].attn;d=attn.embed_dim
    assert not attn.batch_first and attn.dropout==0 and attn.bias_k is None and attn.bias_v is None
    return p|{'Q':attn.in_proj_weight[:d].detach().cpu().double(),
              'qb':attn.in_proj_bias[:d].detach().cpu().double(),'heads':attn.num_heads,'head_dim':d//attn.num_heads}


def attend(query,keys,values,head_dim):
    n,heads=keys.shape[:2];q=query.reshape(n,heads,head_dim)
    probability=torch.einsum('nhd,nhsd->nhs',q,keys).div(head_dim**.5).softmax(-1)
    return torch.einsum('nhs,nhsd->nhd',probability,values).reshape(n,-1)


@torch.no_grad()
def capture_query(model,training,indices):
    block=model.model.visual.transformer.resblocks[11];attn=block.attn
    groups={k:[] for k in ['q_input','keys','values','o_input','r','h','actual']};active={};old_linear=F.linear
    def before_attention(_m,args):
        x=args[0];n=x.shape[1];d=x.shape[-1];heads=attn.num_heads
        projected=old_linear(x,attn.in_proj_weight,attn.in_proj_bias)
        groups['q_input'].append(x[0].detach().cpu())
        for name,start in [('keys',d),('values',2*d)]:
            value=projected[:,:,start:start+d].permute(1,0,2).reshape(n,len(x),heads,d//heads).permute(0,2,1,3)
            groups[name].append(value.detach().cpu())
    def observe(value,weight,bias=None):
        if weight.data_ptr()==active['pointer']:
            active['calls']+=1;groups['o_input'].append(value.reshape(-1,active['batch'],value.shape[-1])[0].detach().cpu())
        return old_linear(value,weight,bias)
    hooks=[attn.register_forward_pre_hook(before_attention),
           block.ln_2.register_forward_pre_hook(lambda _m,args:groups['r'].append(args[0][0].detach().cpu())),
           model.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:groups['h'].append(args[0].detach().cpu()))]
    try:
        model.cuda().eval();active['pointer']=attn.out_proj.weight.data_ptr();F.linear=observe
        for images,_ in make_loader(training,indices,64):
            active.update(batch=len(images),calls=0);groups['actual'].append(model(images.cuda()).detach().cpu());assert active['calls']==1
    finally:
        F.linear=old_linear
        for hook in hooks:hook.remove()
        model.cpu()
    return {k:torch.cat(v).double() for k,v in groups.items()}


class QueryProblem(Problem):
    def __init__(self,target,cfg):
        self.target,self.cfg=target,cfg;self.r=cfg['probe_rank'];self.a_size=self.r**2;self.radius=cfg['parameter_budget'];self.form='matrix'
        parent=Path(cfg['margin_reference'])/(target+'_diagnostic.pt');self.parent_sha=digest(parent)
        assert self.parent_sha==json.loads(parent.with_name(target+'.json').read_text())['diagnostic_sha256']
        previous=torch.load(parent,map_location='cpu');self.prior_confirmation_indices=previous['indices']['confirmation']
        self.anchor,self.expert,self.head,self.hashes=checked_models(target);self.p_cpu=query_parameters(self.expert,self.head);self.p=on_device(self.p_cpu)
        assert torch.equal(self.p_cpu['B'],previous['parameter_reference']['B'])
        if cfg.get('cache_reference'):
            root=Path(cfg['cache_reference']);meta=json.loads((root/'mechanism.json').read_text())
            assert digest(root/'calibration.pt')==cfg['calibration_sha256']==meta['calibration_sha256']
            assert meta['checkpoint_sha256']==self.hashes and meta['parent_sha256']==self.parent_sha
            cache=torch.load(root/'calibration.pt',map_location='cpu')
            proxy=merge(self.anchor,self.expert,self.anchor,cfg['proxy_beta']);self.pp=on_device(query_parameters(proxy,self.head));del proxy
            self.complete(cache,cache['basis_Q'],cache['basis_B'],cache['indices'])
            save_json(Path(cfg['output'])/'mechanism.json',{'target':target,'cache_reference':str(root),'calibration_sha256':cfg['calibration_sha256'],'checkpoint_sha256':self.hashes,'new_model_forwards':0})
            return
        training=dataset(target,'data',self.expert.val_preprocess,True);indices={k:previous['indices'][k] for k in ['fit','scale']}
        cache={g:capture_query(self.expert,training,indices[k]) for g,k in [('native','fit'),('scale','scale')]}
        proxy=merge(self.anchor,self.expert,self.anchor,cfg['proxy_beta']);self.pp=on_device(query_parameters(proxy,self.head))
        cache['proxy']=capture_query(proxy,training,indices['fit']);del proxy
        checks=[]
        for group in ['native','scale','proxy']:
            data=on_device(cache[group]);p=self.pp if group=='proxy' else self.p
            a=attend(data['q_input']@p['Q'].T+p['qb'],data['keys'],data['values'],p['head_dim'])
            error=float((a-data['o_input']).abs().max());actual=data['actual']/data['actual'].norm(dim=1,keepdim=True)@p['head'].T
            score_error=float((scores(data['h'],p)-actual).abs().max());assert error<5e-4 and score_error<5e-4
            checks.append({'group':group,'n':len(a),'attention_input_maximum_error':error,'score_maximum_error':score_error})
        original=query_parameters(self.anchor,self.head);task=self.p_cpu['Q']-original['Q'];u,s,vh=torch.linalg.svd(task,full_matrices=False)
        modes={'u':u.cuda(),'vh':vh.cuda()};importance={};fd=[]
        for group in ['native','scale']:
            data=on_device({k:v[:cfg['selection_n']] for k,v in cache[group].items()});p=self.p
            query=(data['q_input']@p['Q'].T+p['qb']).detach().requires_grad_(True)
            base=attend(query,data['keys'],data['values'],p['head_dim']).detach();rounding=data['h']-final_hidden(data['r'],p)
            def classify(q):
                r=data['r']+(attend(q,data['keys'],data['values'],p['head_dim'])-base)@p['O'].T
                return angular_scores(scores(rounding+final_hidden(r,p),p))
            y=classify(query);jac=torch.stack([torch.autograd.grad(y[:,c].sum(),query,retain_graph=True)[0] for c in range(y.shape[1])],1)
            response=(jac@modes['u'])*(data['q_input']@modes['vh'].T)[:,None,:]
            winner=y.argmax(1);counts=torch.bincount(winner,minlength=y.shape[1]);weights=1./counts[winner].double()/int((counts>0).sum())
            importance[group]=(response.square().sum(1)*weights[:,None]).sum(0).detach()
            with torch.no_grad():
                for i in [0,int(importance[group].argmax()),len(s)-1]:
                    dq=(data['q_input']@modes['vh'][i])[:,None]*modes['u'][:,i];h=cfg['selection_derivative_step']
                    measured=(classify(query+h*dq)-classify(query-h*dq))/(2*h)
                    error=float((measured-response[:,:,i]).norm()/response[:,:,i].norm().clamp_min(1e-20));assert error<1e-3
                    fd.append({'group':group,'mode':i,'relative_error':error})
        chosen=importance['native'].argsort(descending=True)[:self.r].cpu();basis={'u':u[:,chosen],'s':s[chosen],'vh':vh[chosen],'selected':chosen}
        self.complete(cache,basis,previous['basis']['B'],indices)
        out=Path(cfg['output']);torch.save(self.cal,out/'calibration.pt')
        save_json(out/'mechanism.json',{'target':target,'site':'block11 Query weight and class-output B completion','calibration_sha256':digest(out/'calibration.pt'),'checkpoint_sha256':self.hashes,'parent_sha256':self.parent_sha,'cache_checks':checks,'selection_read_only_gradient_calls':2*self.proxy['teacher'].shape[1],
                 'selected_query_modes':chosen.tolist(),'fit_response_fraction':float(importance['native'][chosen.cuda()].sum()/importance['native'].sum()),'scale_response_fraction':float(importance['scale'][chosen.cuda()].sum()/importance['scale'].sum()),
                 'fit_scale_top32_overlap':len(set(chosen.tolist())&set(importance['scale'].argsort(descending=True)[:self.r].cpu().tolist())),
                 'unit_mode_derivative_checks':fd,'target_training_forward_instances':3*len(indices['fit']),'future_partner_used':False,'test_used':False,'new_weights':0})

    def complete(self,cache,basis,b_basis,indices):
        self.basis={'A':on_device(basis),'B':on_device(b_basis)};self.na,self.nb=self.p['Q'].norm(),self.p['B'].norm()
        self.class_space,self.class_geometry=readout(self.p,self.cfg['readout_relative_tolerance']);self.output_basis=self.class_space['Q'];self.b_rows=self.output_basis.shape[1];self.b_width=self.p['B'].shape[1]
        self.cal={**cache,'indices':indices,'basis_Q':basis,'basis_B':b_basis,'output_basis':self.output_basis.cpu()}
        self.native=[self.prepare(cache[g],self.p,True) for g in ['native','scale']];self.proxy=self.prepare(cache['proxy'],self.pp,False)
        teacher=self.proxy['teacher'];self.winner=teacher.argmax(1);self.rows=torch.arange(len(teacher),device='cuda')
        counts=torch.bincount(self.winner,minlength=teacher.shape[1]);self.weights=1./counts[self.winner].double()/int((counts>0).sum())
        self.unit=self.gap(angular_scores(teacher)).median().detach();assert self.unit>0
        self.initial=-self.cfg['parameter_budget']*torch.diag(self.basis['A']['s'])/self.basis['A']['s'].norm()

    @torch.no_grad()
    def prepare(self,cache_cpu,p,probes):
        data=on_device(cache_cpu);r,x=data['r'],data['q_input'];n=self.cfg['probe_contexts'];count=2*self.r
        rounding=data['h']-final_hidden(r,p);zero=attend(x@p['Q'].T+p['qb'],data['keys'],data['values'],p['head_dim'])
        rr=[r];xx=[x];cc=[rounding];aa=[zero];context=[torch.arange(len(r),device='cuda')];offset=[torch.zeros((len(r),p['B'].shape[1]),device='cuda',dtype=torch.float64)]
        if probes:
            hidden=gelu(normalized_input(r,p)@p['A'].T+p['a']);shifts=[]
            for inputs,v in [(x,self.basis['A']['vh']),(hidden,self.basis['B']['vh'])]:
                rms=(inputs@v.T).square().mean(0).sqrt();shifts.append((self.cfg['finite_step']*torch.tensor([-1.,1.],device='cuda')[:,None,None]*rms[None,:,None]*v[None,:,:]).reshape(count,-1))
            sq,sb=shifts
            rr.extend([r[:n].repeat_interleave(count,0)]*2);xx.extend([(x[:n,None,:]+sq[None,:,:]).reshape(n*count,-1),x[:n].repeat_interleave(count,0)])
            cc.extend([rounding[:n].repeat_interleave(count,0)]*2);aa.extend([zero[:n].repeat_interleave(count,0)]*2);context.extend([torch.arange(n,device='cuda').repeat_interleave(count)]*2)
            offset.extend([torch.zeros((n*count,p['B'].shape[1]),device='cuda',dtype=torch.float64),sb.repeat(n,1)])
        packed={'r':torch.cat(rr),'q_input':torch.cat(xx),'rounding':torch.cat(cc),'base_attention':torch.cat(aa),'context_indices':torch.cat(context),'keys':data['keys'],'values':data['values'],'b_offset':torch.cat(offset)}
        packed['task_input']=packed['q_input']@self.basis['A']['vh'].T
        packed['teacher']=self.predict(packed,p,torch.zeros(self.a_size+self.b_rows*self.b_width,device='cuda',dtype=torch.float64))
        return packed

    def routed_residual(self,packed,p,co,factor=1.):
        dq=factor*self.na*(packed['task_input']@co.reshape(self.r,self.r).T)@self.basis['A']['u'].T
        keys,values=packed['keys'],packed['values']
        if 'context_indices' in packed:keys,values=keys[packed['context_indices']],values[packed['context_indices']]
        changed=attend(packed['q_input']@p['Q'].T+p['qb']+dq,keys,values,p['head_dim'])
        return packed['r']+(changed-packed['base_attention'])@p['O'].T

    def predict(self,packed,p,theta,factor=1.):
        r=self.routed_residual(packed,p,theta[:self.a_size],factor);z=gelu(normalized_input(r,p)@p['A'].T+p['a'])+packed['b_offset']
        cb=theta[self.a_size:].reshape(self.b_rows,self.b_width)
        h=packed['rounding']+r+z@p['B'].T+p['b']+factor*self.nb*(z@cb.T)@self.output_basis.T
        return scores(h,p)


class QueryCompletion(ClassCompletion):
    problem_class=QueryProblem
    real_fields=('r','task_input','rounding','q_input','keys','values','base_attention')
    confirmation_capture=staticmethod(capture_query)
    def state(self,co):
        p=self.problem;r=p.routed_residual(self.real,p.p,co);z=gelu(normalized_input(r,p.p)@p.p['A'].T+p.p['a'])
        return z,self.real['rounding']+r+z@p.p['B'].T+p.p['b']

    def publication_deltas(self,co,theta):
        p=self.problem
        return {'Q':p.na*p.basis['A']['u']@co.reshape(p.r,p.r)@p.basis['A']['vh'],
                'B':p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get('mode')=='optimize':optimize(args.target,cfg,problem_class=QueryCompletion)
    else:diagnose(args.target,cfg,problem_class=QueryCompletion)
