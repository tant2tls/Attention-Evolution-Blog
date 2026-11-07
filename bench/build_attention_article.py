"""Build the H100 article from frozen CSVs; never modifies report evidence.

python bench/build_attention_article.py
python bench/build_attention_article.py --check
For staging: --repo REPOSITORY --output DESTINATION
"""
import argparse
import collections
import csv
import hashlib
import gzip
import math
import html
import json
import re
import statistics
from pathlib import Path
from attention_diagrams import diagrams

STUDY = '2026-11-03_h100x8_vllm033dev450_cu130_ep'
ORDER = ['llama-31-70b', 'mimo-v26', 'v4-0731', 'qwen-38-fp8', 'glm-53', 'v41']
NAMES = ['Llama 3.1 70B', 'MiMo V2.6', 'DeepSeek V4 0731', 'Qwen3.8 FP8', 'GLM-5.3', 'DeepSeek V4.1']
COLORS = ['#3e7057','#b7783d','#5076a6','#7b6aa6','#228b8b','#ba6b85']
TYPES = ['Full GQA', 'Window + global', 'Compressed + sparse', 'GDN + sparse QSA', 'KDA + sparse MLA', 'Shared compressed KV']
CATS = ['ordinary_kv','window_kv','shared_compressed','sparse_index_tail','recurrent']
MEMFIELDS = ['ordinary_kv_gib','window_kv_gib','shared_compressed_gib','sparse_index_tail_gib','recurrent_state_gib']


def build(repo):
    report = repo/'report'/STUDY
    def rows(name):
        with (report/name).open(encoding='utf-8',newline='') as f:
            return list(csv.DictReader(f))
    ledger=json.loads((report/'manifest_hashes.json').read_text())
    for name,digest in ledger.items():
        p=report/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Evidence hash mismatch: {name}'
    manifest=json.loads((report/'manifest.json').read_text())
    assert manifest['vllm_commit']=='17e675965d54bef8a3eb77ea92c6bae195ef793c'
    assert manifest['torch']=='2.14.0+cu130'
    arch={r['model']:r for r in rows('architecture.csv')}
    summary=rows('summary.csv')
    overhead=rows('overhead.csv')
    layers=rows('memory_layers.csv')
    data={'study':STUDY,'models':[], 'contexts':[1024,16384,65536,131072], 'sources':{}, 'ledger_entries':len(ledger)}
    for name in ['manifest.json','architecture.csv','memory.csv','memory_layers.csv','decode.csv','prefill.csv','summary.csv','overhead.csv']:
        data['sources'][name]=hashlib.sha256((report/name).read_bytes()).hexdigest()
    groups={}
    supported=0
    for experiment,file in [('memory','memory.csv'),('decode','decode.csv'),('prefill','prefill.csv')]:
        grouped=collections.defaultdict(list)
        for r in rows(file):
            if r['status']!='measured' or (experiment=='memory' and r['snapshot_phase']!='settled_decode'):continue
            key=(r['model'],int(r['context_tokens']),int(r.get('target_batch') or 1))
            grouped[key].append(r)
        for key,rs in grouped.items():
            assert len(rs)==3 and {r['repeat'] for r in rs}=={'1','2','3'},(experiment,key)
            mid=key[0]
            for r in rs:
                assert r['revision']==manifest['models'][mid]['revision']
                assert r['config_id']==manifest['models'][mid]['config_id']
                p=report/r['manifest']
                assert p.exists() or Path(str(p)+'.gz').exists(),str(p)
                if experiment=='decode':
                    assert r['target_batch']==r['actual_batch'] and int(r['warmup_steps'])>=50 and int(r['measured_steps'])>=50
            metric={'memory':'total_attention_state_gib','decode':'attention_ms_per_step','prefill':'attention_ms_per_chunk'}[experiment]
            sm=[x for x in summary if x['model']==key[0] and int(x['context_tokens'])==key[1] and x['experiment']==experiment and x['metric']==metric and (experiment!='decode' or int(x['target_batch'])==key[2])]
            assert len(sm)==1 and int(sm[0]['valid_repeats'])==3
            assert abs(float(sm[0]['median'])-statistics.median(float(r[metric]) for r in rs))<1e-7
        groups[experiment]=grouped
        supported+=len(grouped)
    assert supported==138,supported
    def stats(rs,field):
        vs=[float(r[field]) for r in rs]
        return {'median':statistics.median(vs),'min':min(vs),'max':max(vs)}
    for mid,name,color,kind in zip(ORDER,NAMES,COLORS,TYPES):
        a=arch[mid]
        inv=json.loads((report/f'raw/{mid}/architecture/inventory.json').read_text())
        config=json.loads((report/f'raw/{mid}/architecture/config.json').read_text())
        text=config.get('text_config',config)
        selected=json.loads((report/f'config/{mid}__selected.json').read_text())
        m={'id':mid,'name':name,'color':color,'kind':kind,'layers':int(a['layers']),
           'total':float(a['total_params_b']),'active':float(a['active_decode_params_b']),
           'lookup':float(a['lookup_table_params_b']),'mtp':float(a['mtp_draft_params_b']),
           'repo':manifest['models'][mid]['repo'],'revision':a['revision'],
           'weight_dtype':a['dtype'],'kv':a['kv_state_dtype'],
           'dtype_evidence':selected['state_dtype_evidence'],'scales':selected['kv_scale_source'],
           'config_id':selected['config_id'],'attention':inv['attention'],'memory':{},'decode':{},'prefill':{},'layer_data':[]}
        for c in data['contexts']:
            key=(mid,c,1)
            if key in groups['memory']:
                rs=groups['memory'][key]
                d=stats(rs,'total_attention_state_gib')
                d.update({k:statistics.median(float(r[f]) for r in rs) for k,f in zip(CATS,MEMFIELDS)})
                assert abs(sum(d[k] for k in CATS)-d['median'])<1e-6
                d['live_tokens']=statistics.median(float(r['actual_live_tokens']) for r in rs)
                d['raw']=[r['manifest'] for r in rs]
                m['memory'][str(c)]=d
            for b in [1,2,4,8]:
                key=(mid,c,b)
                if key not in groups['decode']:continue
                rs=groups['decode'][key];d=stats(rs,'attention_ms_per_step')
                d['model']=stats(rs,'model_ms_per_step')['median']
                ov=next(o for o in overhead if o['model']==mid and o['experiment']=='decode' and int(o['context_tokens'])==c and int(o['target_batch'])==b)
                d['sensitivity']=float(ov['attention_lower_bound_ms'])
                d['overhead_pct']=float(ov['overhead_pct'])
                d['raw']=[r['manifest'] for r in rs]
                m['decode'][f'{c}:{b}']=d
            key=(mid,c,1)
            if key in groups['prefill']:
                rs=groups['prefill'][key];d=stats(rs,'attention_ms_per_chunk')
                assert all(int(r['new_chunk_tokens'])==(1024 if c==1024 else 8192) for r in rs)
                d['model']=stats(rs,'model_ms_per_chunk')['median'];d['positions']=json.loads(rs[0]['executed_positions_by_layer'])
                d['raw']=[r['manifest'] for r in rs]
                m['prefill'][str(c)]=d
        for li in range(m['layers']):
            if mid=='llama-31-70b':typ='Full GQA'
            elif mid=='mimo-v26':typ='Full GQA' if li in inv['attention']['detail']['full_layers'] else 'Window'
            elif mid.startswith('v4'):
                ratio=inv['attention']['detail']['compress_ratios'][li]
                typ='Window' if ratio==0 else f'Window + {ratio}× compressed' if ratio>1 else 'Window + sparse'
            else:typ='Recurrent' if text['layer_types'][li]=='linear_attention' else 'Sparse attention'
            layer={'index':li,'type':typ,'memory':{},'references':[]}
            for c in data['contexts']:
                sums=collections.defaultdict(float)
                bycat=collections.defaultdict(lambda:collections.defaultdict(float))
                for r in layers:
                    match=re.search(r'layers\.(\d+)\.',r['layer_id'])
                    if r['model']!=mid or r['context_tokens']!=str(c) or r['snapshot_phase']!='settled_decode' or not match or int(match[1])!=li:continue
                    if r['shared_reference']:
                        if r['shared_reference'] not in layer['references']:layer['references'].append(r['shared_reference'])
                    if r['state_category'] not in CATS:continue
                    mult=8 if 'x8' in r['rank'] else 1
                    val=float(r['unique_owned_mib'])*mult
                    sums[r['repeat']]+=val
                    bycat[r['state_category']][r['repeat']]+=val
                if sums:layer['memory'][str(c)]={'mib':statistics.median(sums.values()),'categories':{cat:statistics.median(v.values()) for cat,v in bycat.items()}}
            m['layer_data'].append(layer)
        for c,v in m['memory'].items():
            assert abs(sum(l['memory'][c]['mib'] for l in m['layer_data'])/1024-v['median'])<1e-5,(mid,c)
        # Split occupied recurrent pages using the recorded runtime tensor shapes and observed live blocks.
        specs = {}
        if mid in ('qwen-38-fp8', 'glm-53'):
            sess = selected['accepted_sessions'][0]
            relative = f'raw/{mid}/sessions/{sess}/state_buffers.json.gz'
            buffers = json.load(gzip.open(report/relative, 'rt', encoding='utf-8'))
            specs = next(rank for rank in buffers if rank['rank'] == 0)['layers']
            data['sources'][relative] = hashlib.sha256((report/relative).read_bytes()).hexdigest()
        for c, state in m['memory'].items():
            parts = {k: state[k] for k in CATS}
            if specs:
                byrepeat = collections.defaultdict(lambda: collections.defaultdict(float))
                for row in layers:
                    if row['model'] != mid or row['context_tokens'] != c or row['snapshot_phase'] != 'settled_decode' or row['state_category'] != 'recurrent':
                        continue
                    sp = specs[row['layer_id']]['spec']
                    assert sp['__cls__'] == 'MambaSpec' and len(sp['shapes']) == 2
                    assert len(sp['shapes'][0]) == 2 and len(sp['shapes'][1]) == 3
                    widths = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4}
                    conv, matrix = [math.prod(shape)*widths[dtype] for shape,dtype in zip(sp['shapes'],sp['dtypes'])]
                    page = int(row['page_bytes'])
                    padding = page-conv-matrix
                    assert padding >= 0
                    mult = 8 if 'x8' in row['rank'] else 1
                    live = int(row['live_blocks'])*mult/(1024**3)
                    for key,value in [('sconv',conv),('recurrent_matrix',matrix),('state_padding',padding)]:
                        byrepeat[row['repeat']][key] += value*live
                split = {key: statistics.median(v[key] for v in byrepeat.values()) for key in ['sconv','recurrent_matrix','state_padding']}
                assert abs(sum(split.values())-parts.pop('recurrent')) < 1e-6, (mid,c,split)
                parts.update(split)
            assert abs(sum(parts.values())-state['median']) < 1e-6
            state['storage_parts'] = {k:v for k,v in parts.items() if v > 0}
            state['storage_split_basis'] = ('Occupied state blocks; short-conv/matrix/padding split calculated from recorded MambaSpec shapes, dtypes, page bytes and observed live blocks.' if specs else 'Occupied state blocks from measured per-layer allocation categories.')
        data['models'].append(m)
    data['supported_points']=supported
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args();repo=args.repo;out=args.output or repo
    data=build(repo)
    here=Path(__file__).parent
    template=(here/'attention_article_template.html').read_text(encoding='utf-8')
    js=(here/'attention_article.js').read_text(encoding='utf-8')
    serialized=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    result=template.replace('@@DATA@@',serialized).replace('@@SCRIPT@@',js)
    for model in data['models']:
        result=result.replace('@@MEM:'+model['id']+'@@',f"{model['memory']['65536']['median']:.2f}")
        result=result.replace('@@DEC:'+model['id']+'@@',f"{model['decode']['65536:8']['median']:.2f}")
        result=result.replace('@@PRE:'+model['id']+'@@',f"{model['prefill']['65536']['median']:.1f}")
    assert '@@' not in result
    outputs={'index.html':result,'assets/attention/study-data.json':json.dumps(data,ensure_ascii=False,indent=1)+'\n'}
    outputs.update({'assets/attention/'+k+'.svg':v for k,v in diagrams().items()})
    for name,content in outputs.items():
        p=out/name
        if args.check:assert p.exists() and p.read_text(encoding='utf-8')==content,f'Generated file stale: {name}'
        else:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8',newline='\n')
    print(f"{'Checked' if args.check else 'Built'} H100 article: {len(data['models'])} models, {data['supported_points']} supported points, {data['ledger_entries']} evidence hashes; no report files changed.")

if __name__=='__main__':main()
