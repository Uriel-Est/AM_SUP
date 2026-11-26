"""AM_SUP — 24, revisão computacional. Executar diretamente no Positron.
Requer polars, numpy, pandas, scipy, scikit-learn >= 1.2, joblib.
Não altera a Gold nem os resultados da versão anterior.
"""
from __future__ import annotations
import gc
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy import sparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# ---------------- Configuração ----------------
GOLD_DIR = None
OUT_DIR = Path('./reports/modelos_lineares_regularizados_v2')
BASELINES_DIR = Path('./reports/baselines_longitudinais')
ANOS = [2020, 2021, 2022, 2023]
RETRO = 2024
SEED = 42
AMOSTRA_TUNING = 200_000
ESTRATEGIAS = ['completo', 'expostos']
TUNING_SPECS = ['M4', 'M8']
ABLACAO_SPECS = [f'M{i}' for i in range(10)]
# Ridge alpha = n*lambda, para manter a intensidade de regularização
# aproximadamente comparável entre amostra e base completa.
RIDGE_LAMBDAS = [1e-5, 1e-4, 1e-3]
LASSO_ALPHAS = [0.001, 0.01, 0.1]
ENET_ALPHAS = [0.001, 0.01, 0.1]
ENET_RATIO = 0.5
SGD_EPOCAS_TUNING = 30
SGD_EPOCAS_FULL = 20
SGD_ETA = 0.005
SGD_TOL = 1e-3
RIDGE_TOL = 1e-4
RIDGE_MAX_ITER = 300
CLIP_NUM = 12.0
SALVAR_MODELOS = True
SALVAR_PREVISOES = True
# Se True, permite selecionar SGD que alcançou o teto de épocas,
# desde que finito; isso não é certificado de convergência.
ACEITAR_SGD_ORCAMENTO = True
VERSAO = '24-v2.1'
KEYS = ['CO_IES','CO_CURSO','NU_ANO_INGRESSO','ANO_ORIGEM']
Y = 'TADA_FUTURA'
A = 'ANO_ALVO'
CONTROLES = ['exposicao','QT_INGRESSANTE','QT_PERMANENCIA','idade_coorte']
REFS = ['hist_ies_curso_idade_risco','hist_curso_risco']
BENCH = {'mae_medio':3.564263690214786,'rmse_medio':7.62925703444876,
         'r2_medio':0.3121081630771244,'mae_2024':1.935115045104516,
         'rmse_2024':4.799383562367058,'r2_2024':0.27151678313937044}
logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',force=True)
log = logging.getLogger('24-v2')


def exigir(ok, msg):
    if not ok:
        raise ValueError(msg)


def uniq(xs):
    return list(dict.fromkeys(xs))


def jsonstr(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)


def atomic_parquet(df,path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    df.write_parquet(tmp,compression='zstd')
    os.replace(tmp,path)


def atomic_joblib(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    joblib.dump(obj,tmp,compress=3)
    os.replace(tmp,path)


def localizar():
    rel=Path('fluxo_educacao_superior/modelagem_longitudinal')
    candidates=[] if GOLD_DIR is None else [Path(GOLD_DIR)]
    for p in [Path.cwd(),*[Path(s) for s in sys.path if s]]:
        candidates += [p/'data/gold'/rel,p/'gold'/rel,p/rel,p/'modelagem_longitudinal',p]
    for p in candidates:
        if all((p/n).is_file() for n in ['features_longitudinais.parquet','alvos_longitudinais.parquet','09_auditoria_vazamento.json']):
            return p.resolve()
    raise FileNotFoundError('Gold longitudinal não encontrada. Configure GOLD_DIR se necessário.')


def contrato(p):
    c=json.loads((p/'09_auditoria_vazamento.json').read_text(encoding='utf-8'))
    exigir(c['status']=='APROVADO' and c['chaves']==KEYS and c['teste_sintetico'], 'Contrato 09 inválido')
    t={v['corte']:v for v in c['testes_invariancia']}
    exigir(all(t.get(a,{}).get('truncamento') and t.get(a,{}).get('perturbacao') for a in [2020,2022,2023]),'Invariância não aprovada')
    f=set(c['features_autorizadas']); cat=set(c['categoricas']); num=set(c['numericas'])
    proib=set(c['metadados_historicos_proibidos']+c['alvos_e_controles_proibidos'])
    exigir(not f&proib and not cat&num and cat|num==f,'Contrato de features inconsistente')
    return c


def specs(c):
    f=c['features_autorizadas']
    m0=['TADA','QT_INGRESSANTE','ANO_ORIGEM','NU_ANO_INGRESSO','CO_REGIAO','CO_UF','CO_CINE_AREA_GERAL','TP_CATEGORIA_ADMINISTRATIVA','TP_ORGANIZACAO_ACADEMICA','TP_GRAU_ACADEMICO','TP_MODALIDADE_ENSINO']
    m1=m0+['idade_coorte','exposicao','zero_estrutural','QT_PERMANENCIA','log_QT_PERMANENCIA']
    m2=m1+['TAP','TDA','TCA','QT_DESISTENCIA','QT_CONCLUINTE','log_QT_INGRESSANTE','log_QT_DESISTENCIA','log_QT_CONCLUINTE','risco_atual']
    m3=m2+[v for v in f if any(s in v for s in ['_lag1','_lag2','_lag3','_delta1','_media3','_inclinacao2'])]
    m4=m3+['CO_MUNICIPIO','CO_CINE_ROTULO']; m5=m4+['CO_IES']; m6=m5+['CO_CURSO']
    h=[v for v in f if v.startswith('hist_')]
    d=dict(zip([f'M{i}' for i in range(10)],[m0,m1,m2,m3,m4,m5,m6,m4+h,m6+h,f]))
    for k,v in d.items():
        d[k]=uniq(v); exigir(set(v)<=set(f),f'Feature não autorizada em {k}')
    return d


def abrir_base(p,c):
    f=pl.scan_parquet(p/'features_longitudinais.parquet'); y=pl.scan_parquet(p/'alvos_longitudinais.parquet')
    exigir(set(c['features_autorizadas']+c['metadados_historicos_proibidos'])<=set(f.collect_schema().names()),'Schema da Gold incompatível')
    for lf,n in [(f,c['n_features_linhas']),(y,c['n_alvos'])]:
        a=lf.select(pl.len().alias('n'),pl.struct(KEYS).n_unique().alias('u'),pl.any_horizontal([pl.col(k).is_null() for k in KEYS]).sum().alias('null')).collect().row(0)
        exigir(a==(n,n,0),f'Chaves/dimensões inválidas: {a}')
    v=f.select([(pl.col(k)>pl.col('ANO_ORIGEM')).fill_null(False).sum().alias(k) for k in c['metadados_historicos_proibidos']]).collect().row(0)
    exigir(sum(v)==0,'Históricos futuros')
    b=f.select(c['features_autorizadas']).join(y.select(KEYS+[A,Y]),on=KEYS,how='inner',validate='1:1')
    a=b.select(pl.len().alias('n'),(pl.col(A)!=pl.col('ANO_ORIGEM')+1).fill_null(True).sum().alias('h'),(~pl.col(Y).is_finite()|~pl.col(Y).is_between(0,100)).fill_null(True).sum().alias('y'),(~pl.col('exposicao').is_finite()|~pl.col('exposicao').is_between(0,100)).fill_null(True).sum().alias('e'),((pl.col('exposicao')==0)&(pl.col(Y)>0)).sum().alias('z'),(~pl.col('QT_INGRESSANTE').is_finite()|(pl.col('QT_INGRESSANTE')<=0)).fill_null(True).sum().alias('ninv'),((pl.col('exposicao')-100*pl.col('QT_PERMANENCIA')/pl.col('QT_INGRESSANTE')).abs()>1e-4).fill_null(True).sum().alias('exp')).collect().row(0)
    exigir(a[0]==c['n_alvos'] and sum(a[1:])==0,f'Base inválida: {a}')
    log.info('Gold auditada: %s pares; %s features',a[0],len(c['features_autorizadas']))
    return b


def grade():
    return ([{'algoritmo':'ridge','alpha':a,'l1_ratio':None} for a in RIDGE_LAMBDAS]+
            [{'algoritmo':'lasso_sgd','alpha':a,'l1_ratio':None} for a in LASSO_ALPHAS]+
            [{'algoritmo':'elasticnet_sgd','alpha':a,'l1_ratio':ENET_RATIO} for a in ENET_ALPHAS])


def task_id(stage,ano,spec,strategy,config):
    return jsonstr([stage,ano,spec,strategy,config])


def fingerprint(p,c,s):
    # Hash integral dos arquivos: detecta inclusive troca da Gold com mesmo tamanho.
    hashes={}
    for name in ['09_auditoria_vazamento.json','features_longitudinais.parquet','alvos_longitudinais.parquet']:
        h=hashlib.sha256()
        with (p/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        hashes[name]=h.hexdigest()
    cfg={'versao':VERSAO,'hashes':hashes,'specs':s,'anos':ANOS,'retro':RETRO,'seed':SEED,'amostra':AMOSTRA_TUNING,'estrategias':ESTRATEGIAS,'tuning_specs':TUNING_SPECS,'ablacao_specs':ABLACAO_SPECS,'grade':grade(),'epocas':[SGD_EPOCAS_TUNING,SGD_EPOCAS_FULL],'eta':SGD_ETA,'tol_sgd':SGD_TOL,'tol_ridge':RIDGE_TOL,'max_ridge':RIDGE_MAX_ITER,'clip':CLIP_NUM,'aceitar_sgd':ACEITAR_SGD_ORCAMENTO,'salvar_modelos':SALVAR_MODELOS,'salvar_previsoes':SALVAR_PREVISOES,'libs':[sklearn.__version__,np.__version__,pd.__version__,pl.__version__,scipy.__version__]}
    return hashlib.sha256(jsonstr(cfg).encode()).hexdigest(),cfg


class Checkpoint:
    def __init__(self,path,fp):
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS ajustes (id TEXT PRIMARY KEY, etapa TEXT, payload TEXT NOT NULL)')
        old=self.db.execute("SELECT v FROM meta WHERE k='fingerprint'").fetchone()
        exigir(old is None or old[0]==fp,'Checkpoint incompatível. Use outra OUT_DIR; não misture configurações.')
        with self.db: self.db.execute("INSERT OR IGNORE INTO meta VALUES ('fingerprint',?)",(fp,))
    def has(self,k):
        return self.db.execute('SELECT 1 FROM ajustes WHERE id=?',(k,)).fetchone() is not None
    def save(self,k,stage,rows):
        with self.db: self.db.execute('INSERT OR REPLACE INTO ajustes VALUES (?,?,?)',(k,stage,jsonstr(rows)))
    def rows(self,stage):
        return [r for (v,) in self.db.execute('SELECT payload FROM ajustes WHERE etapa=? ORDER BY id',(stage,)) for r in json.loads(v)]
    def count(self,stage):
        return self.db.execute('SELECT COUNT(*) FROM ajustes WHERE etapa=?',(stage,)).fetchone()[0]
    def close(self): self.db.close()


def fold(b,c,features,ano,strategy,stage):
    cols=uniq(KEYS+features+CONTROLES+[A,Y]); parts={p['ano_alvo']:p for p in c['particoes']}
    tr=b.filter(pl.col(A)<ano); te=b.filter(pl.col(A)==ano)
    exigir(tr.select(pl.len()).collect().item()==parts[ano]['n_treino'],'Treino incompatível')
    if strategy=='expostos': tr=tr.filter(pl.col('QT_PERMANENCIA')>0)
    if stage=='tuning':
        # Amostra proporcional por ano-alvo, sem consultar resultados futuros.
        keys=tr.select(KEYS+[A]).collect().sort(KEYS)
        if keys.height>AMOSTRA_TUNING:
            groups=keys.partition_by(A,maintain_order=True)
            quotas=[int(AMOSTRA_TUNING*g.height/keys.height) for g in groups]
            for i in range(AMOSTRA_TUNING-sum(quotas)): quotas[i%len(quotas)]+=1
            keys=pl.concat([g.sample(n=q,seed=SEED+int(g[A][0]),shuffle=True) for g,q in zip(groups,quotas) if q>0]).select(KEYS)
            tr=tr.join(keys.lazy(),on=KEYS,how='semi')
        del keys
    train=tr.select(cols).collect().sort(KEYS); test=te.select(cols).collect().sort(KEYS)
    exigir(train.height>0 and test.height==parts[ano]['n_avaliacao'],'Fold inválido')
    exigir(train[A].max()<test[A].min(),'Vazamento temporal')
    return train,test


def pandas_x(df,features,c):
    nums=set(c['numericas']); d={}
    for v in features:
        if v in nums:
            a=df[v].cast(pl.Float32,strict=False).to_numpy().copy(); a[~np.isfinite(a)]=np.nan; d[v]=a
        else: d[v]=df[v].cast(pl.String).fill_null('__AUSENTE__').to_numpy()
    return pd.DataFrame(d,columns=features,copy=False)


def preprocess(features,c):
    n=[v for v in features if v in c['numericas']]; a=[v for v in features if v in c['categoricas']]
    exigir(set(n+a)==set(features),'Tipos não autorizados')
    # Centralizar APENAS o bloco numérico, jamais a matriz one-hot.
    num=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scaler',StandardScaler()),('clip',FunctionTransformer(np.clip,kw_args={'a_min':-CLIP_NUM,'a_max':CLIP_NUM},validate=False,feature_names_out='one-to-one'))])
    cat=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
    return ColumnTransformer([('num',num,n),('cat',cat,a)],remainder='drop',sparse_threshold=1.0)


def csr(x):
    x=sparse.csr_matrix(x,dtype=np.float32); x.sum_duplicates(); x.sort_indices()
    exigir(np.isfinite(x.data).all(),'Matriz não finita')
    return x


def matrices(tr,te,features,c):
    pre=preprocess(features,c)
    x=pandas_x(tr,features,c); xt=csr(pre.fit_transform(x)); del x; gc.collect()
    x=pandas_x(te,features,c); xv=csr(pre.transform(x)); del x; gc.collect()
    exigir(xt.shape[1]==xv.shape[1],'Encoding incompatível')
    mib=(xt.data.nbytes+xt.indices.nbytes+xt.indptr.nbytes)/2**20
    log.info('CSR treino=%s teste=%s | %.0f MiB',xt.shape,xv.shape,mib)
    return xt,xv,pre,mib


def arr(df,k): return df[k].cast(pl.Float64).to_numpy()


def fit(config,x,y,stage):
    alg=config['algoritmo']; alpha=config['alpha']; epochs=SGD_EPOCAS_TUNING if stage=='tuning' else SGD_EPOCAS_FULL
    if alg=='ridge':
        model=Ridge(alpha=alpha*x.shape[0],solver='lsqr',tol=RIDGE_TOL,max_iter=RIDGE_MAX_ITER)
    else:
        model=SGDRegressor(loss='squared_error',penalty='l1' if alg=='lasso_sgd' else 'elasticnet',alpha=alpha,l1_ratio=config['l1_ratio'] or .5,max_iter=epochs,tol=SGD_TOL,n_iter_no_change=3,learning_rate='invscaling',eta0=SGD_ETA,power_t=.25,average=True,shuffle=True,random_state=SEED,early_stopping=False)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning); model.fit(x,y.astype(np.float32,copy=False))
    n_iter=int(np.max(model.n_iter_)) if getattr(model,'n_iter_',None) is not None else None
    warning=any(issubclass(w.category,ConvergenceWarning) for w in ws)
    cap=RIDGE_MAX_ITER if alg=='ridge' else epochs
    budget=warning or (n_iter is not None and n_iter>=cap)
    exigir(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Coeficientes não finitos')
    return model,{'tempo_treino':time.perf_counter()-start,'n_iter':n_iter,'aviso_convergencia':warning,'atingiu_limite':budget,'convergiu':not budget,'elegivel':not budget or (alg!='ridge' and ACEITAR_SGD_ORCAMENTO),'alpha_efetivo':alpha*x.shape[0] if alg=='ridge' else alpha,'n_coeficientes':int(model.coef_.size),'n_coeficientes_nao_zero':int(np.count_nonzero(model.coef_))}


def variants(p,e):
    return {'bruta':p,'clip_0_100':np.clip(p,0,100),'zero_estrutural':np.where(e==0,0,np.clip(p,0,100)),'limitada_exposicao':np.clip(p,0,e)}


def metrics(y,p,e,n):
    exigir(len(p)==len(y) and np.isfinite(p).all(),'Predição inválida')
    err=p-y; ae=np.abs(err); sq=err**2; sst=np.sum((y-y.mean())**2)
    def mean(x): return float(x.mean()) if len(x) else None
    ex=e>0; pos=y>0; sste=np.sum((y[ex]-y[ex].mean())**2) if ex.any() else 0
    return {'n':len(y),'mae':float(ae.mean()),'rmse':float(np.sqrt(sq.mean())),'r2':float(1-sq.sum()/sst) if sst>0 else None,'vies':float(err.mean()),'media_observada':float(y.mean()),'media_prevista':float(p.mean()),'mae_expostos':mean(ae[ex]),'rmse_expostos':float(np.sqrt(sq[ex].mean())) if ex.any() else None,'r2_expostos':float(1-sq[ex].sum()/sste) if sste>0 else None,'mae_positivos':mean(ae[pos]),'mae_zero_estrutural':mean(ae[~ex]),'mae_zero_exposto':mean(ae[ex&(y==0)]),'mae_ponderado_ingressantes':float(np.average(ae,weights=n)),'pct_pred_zero':float(100*np.mean(p==0)),'pct_pred_negativa':float(100*np.mean(p<0)),'pct_pred_acima_exposicao':float(100*np.mean(p>e+.01))}


def config_of(r):
    return {'algoritmo':str(r['algoritmo']),'alpha':float(r['alpha']),'l1_ratio':None if pd.isna(r['l1_ratio']) else float(r['l1_ratio'])}


def ranking(rows):
    df=pd.DataFrame(rows); df=df[df['status']=='ok'].copy()
    exigir(not df.empty,'Nenhum ajuste válido')
    groups=['especificacao','estrategia','algoritmo','alpha','l1_ratio','variante']
    r=df.groupby(groups,dropna=False).agg(n_validacoes=('ano_alvo','nunique'),mae_medio=('mae','mean'),mae_dp=('mae','std'),rmse_medio=('rmse','mean'),r2_medio=('r2','mean'),vies_medio=('vies','mean'),mae_expostos_medio=('mae_expostos','mean'),mae_positivos_medio=('mae_positivos','mean'),mae_ponderado_medio=('mae_ponderado_ingressantes','mean'),tempo_medio=('tempo_treino','mean'),elegivel_todos=('elegivel','all'),convergiu_todos=('convergiu','all'),n_limite=('atingiu_limite','sum'),n_features_transformadas=('n_features_transformadas','max')).reset_index()
    r=r[r.n_validacoes==len(ANOS)].copy()
    for k in ['mae_medio','rmse_medio','r2_medio']: r['delta_'+k+'_vs_23']=r[k]-BENCH[k]
    return r.sort_values(['mae_medio','rmse_medio','especificacao','estrategia','algoritmo','alpha','variante'],kind='stable')


def export_rows(cp,stage):
    name={'tuning':'01_tuning_parcial.csv','ablacao':'04_ablacao_parcial.csv','retrospectivo':'06_metricas_retrospectivas.csv'}[stage]
    atomic_csv(pd.DataFrame(cp.rows(stage)),OUT_DIR/name)


def progress(cp,stage,total,started):
    count=cp.count(stage); elapsed=time.perf_counter()-started[0]; new=count-started[1]
    eta=(elapsed/new*(total-count)/60) if new>0 else None
    log.info('%s: %s/%s (%.1f%%) | ETA aproximada: %s',stage,count,total,100*count/max(1,total),f'{eta:.1f} min' if eta is not None else 'indisponível')


def run_stage(b,c,s,cp,stage,groups,final_selection=None):
    total=sum(len(configs) for _,_,_,configs in groups); started=(time.perf_counter(),cp.count(stage))
    progress(cp,stage,total,started)
    for ano,spec,strategy,configs in groups:
        pending=[cfg for cfg in configs if not cp.has(task_id(stage,ano,spec,strategy,cfg))]
        if not pending: continue
        features=s[spec]; tr,te=fold(b,c,features,ano,strategy,stage)
        log.info('%s | %s | %s | %s | treino=%s avaliação=%s',stage,ano,spec,strategy,tr.height,te.height)
        yt=arr(tr,Y); y=arr(te,Y); e=arr(te,'exposicao'); n=arr(te,'QT_INGRESSANTE')
        nt=tr.height; xt,xv,pre,mib=matrices(tr,te,features,c); del tr; gc.collect()
        for cfg in pending:
            k=task_id(stage,ano,spec,strategy,cfg)
            meta={'etapa':stage,'ano_alvo':ano,'especificacao':spec,'estrategia':strategy,**cfg,'n_treino':nt,'n_features_originais':len(features),'n_features_transformadas':xt.shape[1],'memoria_csr_mib':mib}
            log.info('Ajustando %s',k); t=time.perf_counter()
            try:
                model,diag=fit(cfg,xt,yt,stage)
                pred=model.predict(xv).astype(np.float64)
                exigir(np.isfinite(pred).all(),'Predições não finitas')
                vs=variants(pred,e)
                rows=[{**meta,**diag,'status':'ok','variante':v,**metrics(y,p,e,n)} for v,p in vs.items()]
            except (ValueError,FloatingPointError,OverflowError) as err:
                # Falha numérica registrada; não conta como configuração elegível.
                # Erros de I/O e memória não são engolidos, permitindo retomada.
                log.exception('Ajuste falhou: %s',k)
                rows=[{**meta,'status':'falha','erro':str(err),'tempo_treino':time.perf_counter()-t}]
                cp.save(k,stage,rows); export_rows(cp,stage); progress(cp,stage,total,started); continue
            if stage=='retrospectivo':
                label=hashlib.sha256(k.encode()).hexdigest()[:12]; name=f"{spec}_{strategy}_{cfg['algoritmo']}_{label}"
                selections=final_selection[k]
                for r in rows:
                    r['modelo_id']=name; r['selecionada_validacao']=r['variante'] in selections
                    r['criterios_validacao']=';'.join(selections.get(r['variante'],[]))
                    for m in ['mae','rmse','r2']: r['delta_'+m+'_vs_23']=r[m]-BENCH[m+'_2024'] if r[m] is not None else None
                # Commit somente DEPOIS de persistir artefatos; uma interrupção
                # aqui refaz apenas este ajuste, sobrescrevendo seus próprios arquivos.
                if SALVAR_MODELOS:
                    atomic_joblib({'preprocessador':pre,'modelo':model,'features':features,'categoricas':[v for v in features if v in c['categoricas']],'numericas':[v for v in features if v in c['numericas']],'configuracao':cfg,'selecoes_validacao':selections,'ano_maximo_alvo_treino':ano-1,'target':Y,'clip_num':CLIP_NUM},OUT_DIR/'modelos'/f'{name}.joblib')
                if SALVAR_PREVISOES:
                    out=te.select(uniq(KEYS+[A,Y]+CONTROLES)).with_columns([pl.Series('pred_'+v,p) for v,p in vs.items()])
                    atomic_parquet(out,OUT_DIR/'previsoes'/f'{name}.parquet'); del out
            cp.save(k,stage,rows); export_rows(cp,stage); progress(cp,stage,total,started)
            del model,pred,vs,rows; gc.collect()
        del xt,xv,pre,te,yt,y,e,n; gc.collect()
    export_rows(cp,stage)


def choose_tuning(r):
    # Um alpha por algoritmo, escolhido em 4 folds, depois mantido fixo na ablação.
    sel={}
    for alg in ['ridge','lasso_sgd','elasticnet_sgd']:
        q=r[(r.algoritmo==alg)&r.elegivel_todos]
        if len(q): sel[alg]=config_of(q.iloc[0])
        else: log.warning('Sem configuração elegível: %s',alg)
    exigir(bool(sel),'Nenhum algoritmo elegível no tuning')
    return sel


def finalists(r):
    d={}; groups=[]
    for alg in sorted(r.algoritmo.unique()):
        q=r[(r.algoritmo==alg)&r.elegivel_todos]
        if not len(q): continue
        for metric,asc in [('mae_medio',True),('rmse_medio',True),('r2_medio',False)]:
            a=q.sort_values(metric,ascending=asc,kind='stable').iloc[0]; cfg=config_of(a)
            k=task_id('retrospectivo',RETRO,a.especificacao,a.estrategia,cfg)
            if k not in d:
                d[k]={}; groups.append((RETRO,a.especificacao,a.estrategia,[cfg]))
            d[k].setdefault(a.variante,[]).append(metric)
    exigir(bool(d),'Nenhum finalista elegível')
    return groups,d


def compare_baselines(rows):
    path=BASELINES_DIR/'01_metricas_completas.csv'
    if not path.is_file(): log.warning('CSV do 23 ausente; referência fixa disponível'); return
    b=pd.read_csv(path); b=b[b.modelo.isin(REFS)][['ano_alvo','modelo','n','mae','rmse','r2']].rename(columns={'modelo':'baseline','n':'n_baseline','mae':'mae_baseline','rmse':'rmse_baseline','r2':'r2_baseline'})
    exigir(not b.duplicated(['ano_alvo','baseline']).any() and len(b)==10,'Baselines 23 incompletos')
    d=pd.DataFrame(rows); d=d[d.status=='ok'].merge(b,on='ano_alvo',how='left',validate='many_to_many')
    exigir((d.n==d.n_baseline).all(),'Amostras de avaliação diferentes do 23')
    for m in ['mae','rmse','r2']: d['delta_'+m]=d[m]-d[m+'_baseline']
    atomic_csv(d,OUT_DIR/'08_comparacao_baselines_por_ano.csv')


def synthetic():
    import tempfile
    c={'numericas':['x'],'categoricas':['cat']}; f=['x','cat']
    tr=pd.DataFrame({'x':np.array([1,2,3,4,5,6],np.float32),'cat':['a','b','a','b','a','b']}); te=pd.DataFrame({'x':[100.,np.nan],'cat':['futuro','a']})
    pre=preprocess(f,c); x=csr(pre.fit_transform(tr)); z=csr(pre.transform(te))
    exigir('futuro' not in pre.named_transformers_['cat'].categories_[0],'Categoria futura aprendida')
    exigir(pre.named_transformers_['num'].named_steps['imputer'].statistics_[0]==3.5,'Imputação contaminada')
    exigir(np.allclose(pre.named_transformers_['num'].named_steps['scaler'].mean_[0],3.5),'Centralização incorreta')
    for cfg in [grade()[0],{'algoritmo':'lasso_sgd','alpha':.01,'l1_ratio':None},{'algoritmo':'elasticnet_sgd','alpha':.01,'l1_ratio':.5}]:
        m,_=fit(cfg,x,np.arange(6,dtype=float),'tuning'); p=m.predict(z); exigir(np.isfinite(p).all(),'Teste sintético de ajuste falhou')
    exigir(np.allclose(variants(np.array([8.,-2.,15.]),np.array([0.,5.,10.]))['limitada_exposicao'],[0,0,10]),'Restrição falhou')
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'test.sqlite'; q=Checkpoint(p,'a'); q.save('id','tuning',[{'valor':1}]); q.close(); q=Checkpoint(p,'a'); exigir(q.has('id') and q.rows('tuning')==[{'valor':1}],'Retomada falhou'); q.close()
    log.info('Testes sintéticos aprovados')


def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    for d in ['modelos','previsoes']: (OUT_DIR/d).mkdir(exist_ok=True)
    synthetic(); p=localizar(); c=contrato(p); s=specs(c)
    fp,config=fingerprint(p,c,s); cp=Checkpoint(OUT_DIR/'24_checkpoint.sqlite',fp)
    try:
        atomic_json(OUT_DIR/'24_resumo_modelos.json',{'status':'EM_PROCESSAMENTO','fingerprint':fp})
        atomic_json(OUT_DIR/'00_especificacoes_features.json',s); atomic_json(OUT_DIR/'00_configuracao_execucao.json',config)
        b=abrir_base(p,c)
        log.info('Saída: %s | checkpoint: %s',OUT_DIR.resolve(),fp[:12])
        # Ridge primeiro dentro de cada grupo. Matrizes compartilhadas por grade.
        groups=[(a,m,e,grade()) for m in TUNING_SPECS for e in ESTRATEGIAS for a in ANOS]
        run_stage(b,c,s,cp,'tuning',groups)
        rows=cp.rows('tuning'); atomic_csv(pd.DataFrame(rows),OUT_DIR/'01_tuning_metricas.csv')
        r=ranking(rows); atomic_csv(r,OUT_DIR/'02_tuning_ranking.csv'); print(r.head(15).to_string(index=False))
        sel=choose_tuning(r); atomic_json(OUT_DIR/'03_hiperparametros_selecionados.json',sel)
        # Um ajuste por algoritmo/especificação/estratégia/ano, na base completa.
        groups=[(a,m,e,list(sel.values())) for m in ABLACAO_SPECS for e in ESTRATEGIAS for a in ANOS]
        run_stage(b,c,s,cp,'ablacao',groups)
        rows=cp.rows('ablacao'); atomic_csv(pd.DataFrame(rows),OUT_DIR/'04_ablacao_metricas.csv')
        r=ranking(rows); atomic_csv(r,OUT_DIR/'05_ablacao_ranking.csv'); print(r.head(20).to_string(index=False))
        groups,choices=finalists(r); atomic_json(OUT_DIR/'09_finalistas_congelados.json',choices)
        run_stage(b,c,s,cp,'retrospectivo',groups,choices)
        retro=cp.rows('retrospectivo'); df=pd.DataFrame(retro); df= df[df.status=='ok']
        exigir(not df.empty,'Nenhum finalista retrospectivo válido')
        selected=df[df.selecionada_validacao==True].sort_values('mae')
        atomic_csv(selected,OUT_DIR/'07_finalistas_selecionados_validacao.csv'); compare_baselines(rows+retro)
        print(selected[['modelo_id','variante','mae','rmse','r2','vies','atingiu_limite']].to_string(index=False))
        # Pandas converte NaN para null no JSON.
        counts={stage:cp.count(stage) for stage in ['tuning','ablacao','retrospectivo']}
        summary={'status':'CONCLUIDO','fingerprint':fp,'n_features_autorizadas':len(c['features_autorizadas']),'n_ajustes':counts,'hiperparametros':sel,'benchmark_23':BENCH,'melhores_validacao':json.loads(r.head(20).to_json(orient='records')),'resultados_retrospectivos':json.loads(selected.to_json(orient='records')),'n_falhas':{stage:sum(v.get('status')=='falha' for v in cp.rows(stage)) for stage in counts},'nota':'Tuning em subamostra por ano, ablação e retrospectiva em treino completo. SGD aproximado, teto de épocas explicitado. 2024 previamente explorado. Seleção apenas em 2020–2023.'}
        atomic_json(OUT_DIR/'24_resumo_modelos.json',summary); log.info('Concluído: %s',OUT_DIR.resolve())
    except KeyboardInterrupt:
        log.warning('Interrompido. Ajustes confirmados no SQLite serão retomados automaticamente.')
        atomic_json(OUT_DIR/'24_resumo_modelos.json',{'status':'INTERROMPIDO','fingerprint':fp,'n_ajustes':{s:cp.count(s) for s in ['tuning','ablacao','retrospectivo']}})
        raise
    finally: cp.close()


if __name__=='__main__': main()
