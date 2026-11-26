"""Auditoria longitudinal de TADA. Não modifica a Silver nem ajusta modelos.

Uso: python code/08_auditar_dinamica_temporal_tada.py
      ou execute o arquivo no console do Positron.
Dependências: polars, numpy (pip install polars numpy).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# CONFIGURAÇÕES
# Caminhos relativos ao diretório de trabalho, não ao arquivo Python.
SILVER_DIR = None  # None: procura a Silver no cwd e no sys.path.
OUT_DIR = Path('reports') / 'audit' / 'dinamica_temporal_tada'

KEYS = ['CO_IES', 'CO_CURSO', 'NU_ANO_INGRESSO']
YEAR = 'NU_ANO_REFERENCIA'
TARGET = 'TADA'
FOLDER = 'fluxo_educacao_superior'
CATS = ['CO_REGIAO','CO_UF','CO_MUNICIPIO','CO_CINE_AREA_GERAL','CO_CINE_ROTULO','TP_CATEGORIA_ADMINISTRATIVA','TP_ORGANIZACAO_ACADEMICA','TP_GRAU_ACADEMICO','TP_MODALIDADE_ENSINO']
COUNTS = ['QT_INGRESSANTE','QT_PERMANENCIA','QT_DESISTENCIA','QT_CONCLUINTE']
RATES = ['TADA','TDA','TAP','TCA']
OPTIONAL = ['QT_INGRESSANTE', 'QT_PERMANENCIA', 'QT_DESISTENCIA', 'QT_CONCLUINTE', 'TAP', 'TCA', 'TDA', 'TCCA', 'TAPA', 'CO_UF', 'TP_MODALIDADE_ENSINO']
OPTIONAL = list(dict.fromkeys(OPTIONAL + CATS))
LOG = logging.getLogger('auditoria_tada')
EPS = 1e-8


def resolve_silver(explicit):
    def candidates(p):
        return [p / 'data' / 'silver' / FOLDER, p / 'silver' / FOLDER, p / FOLDER, p]
    if explicit:
        roots = [Path(explicit).expanduser()]
    elif os.environ.get('AM_SUP_SILVER_DIR'):
        roots = [Path(os.environ['AM_SUP_SILVER_DIR']).expanduser()]
    else:
        roots = [Path.cwd()] + [Path(p) for p in sys.path if p]
    seen = set()
    for root in roots:
        for p in candidates(root):
            p = p.resolve()
            if p in seen:
                continue
            seen.add(p)
            if p.is_dir() and any(f for f in p.glob('*.parquet') if not f.name.endswith('.tmp.parquet')):
                if explicit or os.environ.get('AM_SUP_SILVER_DIR') or p.name == FOLDER:
                    return p
    raise FileNotFoundError('Silver não localizada. Configure SILVER_DIR com o caminho da pasta silver ou fluxo_educacao_superior.')


def num(c):
    return pl.col(c).cast(pl.Float64, strict=False).fill_nan(None)


def pct(expr):
    return expr.cast(pl.Float64).mean() * 100


def corr(a, b):
    return pl.corr(a, b).fill_nan(None)


def distribution(c):
    x = pl.col(c)
    return [pl.len().alias('n'), x.count().alias('n_validos'), x.mean().alias('media'), x.std().alias('dp'), x.min().alias('min'), x.quantile(.25).alias('p25'), x.median().alias('mediana'), x.quantile(.75).alias('p75'), x.quantile(.90).alias('p90'), x.quantile(.95).alias('p95'), x.quantile(.99).alias('p99'), x.max().alias('max'), pct(x == 0).alias('pct_zero'), pct(x > 10).alias('pct_acima_10'), pct(x > 25).alias('pct_acima_25')]


def transitions():
    a, b, d = pl.col('TADA'), pl.col('TADA_FUTURA'), pl.col('delta')
    return [pl.len().alias('n'), a.mean().alias('media_origem'), b.mean().alias('media_alvo'), d.mean().alias('delta_medio'), d.abs().mean().alias('mae_persistencia'), (d.pow(2).mean().sqrt()).alias('rmse_persistencia'), pct(d > EPS).alias('pct_aumento'), pct(d < -EPS).alias('pct_reducao'), pct(d.abs() <= EPS).alias('pct_estavel'), pct((a == 0) & (b == 0)).alias('pct_zero_zero'), pct((a == 0) & (b > 0)).alias('pct_zero_positivo'), pct((a > 0) & (b == 0)).alias('pct_positivo_zero'), pct((a > 0) & (b > 0)).alias('pct_positivo_positivo'), pl.when((a == 0).sum() > 0).then(((a == 0) & (b > 0)).sum() / (a == 0).sum() * 100).otherwise(None).alias('pct_positivo_dado_zero'), pl.when((a > 0).sum() > 0).then(((a > 0) & (b == 0)).sum() / (a > 0).sum() * 100).otherwise(None).alias('pct_zero_dado_positivo'), corr('TADA', 'TADA_FUTURA').alias('pearson'), pl.corr('TADA', 'TADA_FUTURA', method='spearman').fill_nan(None).alias('spearman')]


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    silver = resolve_silver(SILVER_DIR)
    out = Path(OUT_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    LOG.info('Silver: %s | Saída: %s', silver, out)

    def save(name, df):
        df.write_csv(out / (name + '.csv'))
        LOG.info('%s: %s linhas', name, df.height)

    files = sorted(p for p in silver.glob('*.parquet') if not p.name.endswith('.tmp.parquet'))
    parts, inventory = [], []
    required = KEYS + [YEAR, TARGET]
    present_optional = set()
    for f in files:
        scan = pl.scan_parquet(f)
        names = scan.collect_schema().names()
        missing = set(required) - set(names)
        if missing:
            raise ValueError(f'{f.name}: colunas obrigatórias ausentes: {sorted(missing)}')
        present_optional.update(set(OPTIONAL) & set(names))
        inventory.append({'arquivo': f.name, 'colunas': '|'.join(names), 'n_colunas': len(names), 'n_linhas': scan.select(pl.len()).collect().item(), 'opcionais_presentes': '|'.join(c for c in OPTIONAL if c in names)})
        parts.append(scan.select([pl.col(c).cast(pl.String, strict=False).str.strip_chars().alias(c) for c in KEYS[:2]] + [num(c).alias(c) for c in [KEYS[2], YEAR, TARGET]] + [((pl.col(c).cast(pl.String, strict=False) if c in CATS else num(c)) if c in names else pl.lit(None, dtype=pl.String if c in CATS else pl.Float64)).alias(c) for c in OPTIONAL]))
    save('01_inventario', pl.DataFrame(inventory))
    base = pl.concat(parts).collect()
    n_raw = base.height
    valid = pl.all_horizontal([pl.col(c).is_not_null() for c in required]) & pl.all_horizontal([pl.col(c).str.len_chars() > 0 for c in KEYS[:2]]) & pl.col(TARGET).is_between(0, 100) & pl.col(YEAR).is_finite() & pl.col(KEYS[2]).is_finite() & (pl.col(YEAR) == pl.col(YEAR).floor()) & (pl.col(KEYS[2]) == pl.col(KEYS[2]).floor()) & (pl.col(YEAR) >= pl.col(KEYS[2]))
    bad = base.filter(~valid.fill_null(False))
    save('02_invalidos_resumo', base.select([pl.len().alias('n_bruto'), (~valid.fill_null(False)).sum().alias('n_invalidos'), *[pl.col(c).null_count().alias('n_nulos_' + c) for c in required], (~pl.col(TARGET).is_between(0,100)).sum().alias('n_tada_fora_intervalo')]))
    base = base.filter(valid).with_columns(pl.col(YEAR, KEYS[2]).cast(pl.Int32)).with_columns((pl.col(YEAR)-pl.col(KEYS[2])).alias('idade_coorte')).sort(KEYS + [YEAR])
    dup = base.group_by(KEYS + [YEAR]).agg(pl.len().alias('n'), pl.col(TARGET).n_unique().alias('n_tada_distintas')).filter(pl.col('n') > 1)
    save('03_duplicatas', dup)
    if dup.height:
        raise ValueError('Chaves duplicadas encontradas. Consulte 03_duplicatas.csv antes de prosseguir; não haverá descarte arbitrário.')
    save('04_distribuicao_anual', base.group_by(YEAR).agg(distribution(TARGET)).sort(YEAR))
    save('05_distribuicao_idade', base.group_by('idade_coorte').agg(distribution(TARGET)).sort('idade_coorte'))
    save('06_distribuicao_ano_idade', base.group_by([YEAR, 'idade_coorte']).agg(distribution(TARGET)).sort([YEAR,'idade_coorte']))
    save('07_cobertura_coorte_ano', base.group_by([KEYS[2], YEAR]).agg(pl.len().alias('n'), pl.col('CO_IES').n_unique().alias('n_ies'), pl.col('CO_CURSO').n_unique().alias('n_cursos')).sort([KEYS[2],YEAR]))
    coverage = base.group_by(KEYS).agg(pl.len().alias('n_anos'), pl.col(YEAR).min().alias('primeiro_ano'), pl.col(YEAR).max().alias('ultimo_ano')).with_columns((pl.col('ultimo_ano')-pl.col('primeiro_ano')+1-pl.col('n_anos')).alias('n_lacunas_internas'))
    save('08_cobertura_trajetorias', coverage.group_by(KEYS[2]).agg(pl.len().alias('n_trajetorias'), (pl.col('n_lacunas_internas')>0).sum().alias('n_com_lacunas'), pl.col('n_lacunas_internas').sum().alias('total_lacunas'), pl.col('n_anos').mean().alias('media_anos')).sort(KEYS[2]))
    # Join exato: não conecta anos separados por lacunas.
    dest = base.select(KEYS + [(pl.col(YEAR)-1).alias(YEAR), pl.col(TARGET).alias('TADA_FUTURA')] + [pl.col(c).alias(c+'_FUTURO') for c in COUNTS+RATES if c != TARGET])
    pairs = base.join(dest, on=KEYS+[YEAR], how='inner', validate='1:1').with_columns((pl.col(YEAR)+1).alias('ANO_ALVO'), (pl.col('TADA_FUTURA')-pl.col(TARGET)).alias('delta'))
    if pairs.is_empty():
        raise ValueError('Nenhum par temporal válido encontrado.')
    # Cobertura do pareamento: último ano não é interpretado como abandono.
    save('09_cobertura_pares', base.group_by([YEAR,'idade_coorte']).len(name='n_origem').join(pairs.group_by([YEAR,'idade_coorte']).len(name='n_com_proximo_ano'), on=[YEAR,'idade_coorte'], how='left').with_columns(pl.col('n_com_proximo_ano').fill_null(0)).with_columns((100*pl.col('n_com_proximo_ano')/pl.col('n_origem')).alias('pct_pareado')).sort([YEAR,'idade_coorte']))
    save('10_transicoes_ano', pairs.group_by(YEAR).agg(transitions()).sort(YEAR))
    save('11_transicoes_idade', pairs.group_by('idade_coorte').agg(transitions()).sort('idade_coorte'))
    save('12_transicoes_ano_idade', pairs.group_by([YEAR,'idade_coorte']).agg(transitions()).sort([YEAR,'idade_coorte']))
    # Defasagens de calendário, sem preencher anos ausentes com zero.
    for k in (1,2,3):
        lag = base.select(KEYS + [(pl.col(YEAR)+k).alias(YEAR), pl.col(TARGET).alias(f'tada_lag{k}')])
        pairs = pairs.join(lag,on=KEYS+[YEAR],how='left',validate='1:1')
    lagrows=[]
    for year in sorted(pairs[YEAR].unique().to_list()):
        subset=pairs.filter(pl.col(YEAR)==year)
        for c in [TARGET,'tada_lag1','tada_lag2','tada_lag3']:
            s=subset.filter(pl.col(c).is_not_null())
            lagrows.append({'ano_origem':year,'variavel':c,'n':s.height,'pearson':s.select(corr(c,'TADA_FUTURA')).item() if s.height>1 else None,'spearman':s.select(pl.corr(c,'TADA_FUTURA',method='spearman').fill_nan(None)).item() if s.height>1 else None})
    save('13_correlacoes_defasagens',pl.DataFrame(lagrows))
    if 'QT_INGRESSANTE' in present_optional:
        q=pl.col('QT_INGRESSANTE')
        pairs=pairs.with_columns(pl.when(q.is_null() | ~q.is_finite() | (q<=0)).then(pl.lit('00_invalido_ausente')).when(q<=10).then(pl.lit('01_1a10')).when(q<=25).then(pl.lit('02_11a25')).when(q<=50).then(pl.lit('03_26a50')).when(q<=100).then(pl.lit('04_51a100')).when(q<=250).then(pl.lit('05_101a250')).otherwise(pl.lit('06_acima250')).alias('faixa_ingressantes'))
        save('14_transicoes_tamanho_coorte',pairs.group_by([YEAR,'faixa_ingressantes']).agg(transitions()).sort([YEAR,'faixa_ingressantes']))
    optional_rows=[]
    for c in sorted(present_optional):
        if c in CATS:
            continue
        optional_rows.append(pairs.group_by(YEAR).agg(pl.col(c).count().alias('n_validos'), pl.col(c).min().alias('min'), pl.col(c).max().alias('max'), pl.col(c).mean().alias('media'), pct(pl.col(c)==0).alias('pct_zero'), corr(c,'TADA_FUTURA').alias('pearson_com_alvo')).with_columns(pl.lit(c).alias('variavel')))
    if optional_rows:
        save('15_variaveis_auxiliares',pl.concat(optional_rows).sort([YEAR,'variavel']))
    # Baselines estimados exclusivamente em pares com alvo disponível até a origem.
    rows=[]
    for year in sorted(pairs[YEAR].unique().to_list()):
        train=pairs.filter(pl.col(YEAR)<year)
        test=pairs.filter(pl.col(YEAR)==year)
        if train.is_empty():
            continue
        mean,median=train.select(pl.col('TADA_FUTURA').mean().alias('media'),pl.col('TADA_FUTURA').median().alias('mediana')).row(0)
        hist=train.group_by('idade_coorte').agg(pl.col('TADA_FUTURA').mean().alias('media_idade'),pl.col('TADA_FUTURA').median().alias('mediana_idade'))
        test=test.join(hist,on='idade_coorte',how='left')
        y=test['TADA_FUTURA'].to_numpy()
        q=test['QT_INGRESSANTE'].to_numpy()
        w=np.isfinite(q)&(q>0)
        ss=float(np.sum((y-y.mean())**2))
        predictions={'zero':np.zeros(len(y)), 'media_treino':np.full(len(y),mean), 'mediana_treino':np.full(len(y),median), 'persistencia':test[TARGET].to_numpy(), 'media_historica_idade':test['media_idade'].fill_null(mean).to_numpy(), 'mediana_historica_idade':test['mediana_idade'].fill_null(median).to_numpy()}
        for name,pred in predictions.items():
            err=pred-y
            pos=y>0
            rows.append({'ano_origem':year,'ano_alvo':year+1,'modelo':name,'n_treino_disponivel':train.height,'n_teste':len(y),'mae':float(np.mean(abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'r2':1-float(np.sum(err**2))/ss if ss>0 else None,'vies':float(err.mean()),'mae_positivos':float(np.mean(abs(err[pos]))) if pos.any() else None,'mae_ponderado_ingressantes':float(np.average(abs(err[w]),weights=q[w])) if w.any() else None,'n_peso_valido':int(w.sum())})
    if rows:
        save('16_baselines_temporais',pl.DataFrame(rows).sort(['ano_origem','mae']))
    extra = auditoria_exposicao(base, pairs, save)
    test=pairs.filter(pl.col(YEAR)==2023)
    summary={'silver':str(silver),'n_arquivos':len(files),'n_bruto':n_raw,'n_invalidos':bad.height,'n_validos':base.height,'n_trajetorias':coverage.height,'n_pares':pairs.height,'n_treino_ate_2022':pairs.filter(pl.col(YEAR)<=2022).height,'n_teste_2023_2024':test.height,'opcionais_presentes':sorted(present_optional),'opcionais_ausentes':sorted(set(OPTIONAL)-present_optional),'nota':'TADA tratada como indicador numérico de 0 a 100. A interpretação anual/acumulada e a população exposta exigem confirmação no dicionário oficial. Reduções não são classificadas automaticamente como erros.'}
    summary.update(extra)
    (out/'00_resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if test.height:
        print('\nTRANSIÇÕES 2023 → 2024:')
        print(test.select(transitions()).to_dicts()[0])
    if rows:
        print('\nBASELINES 2023 → 2024:')
        for row in rows:
            if row['ano_origem']==2023:
                print(f"{row['modelo']:26s} MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R²={row['r2']}")
    LOG.info('Auditoria concluída: %s',out)



def auditoria_exposicao(base, pairs, save):
    """Diagnósticos complementares consolidados no mesmo pipeline de auditoria."""
    N, P, D, C = COUNTS
    tol = .01
    def err(x, name):
        return [x.count().alias(name+'_n'), x.abs().mean().alias(name+'_mae'), x.abs().max().alias(name+'_max'), pct(x.abs()<=tol).alias(name+'_pct_igual')]
    quality=[]
    for c in COUNTS+RATES:
        x=base[c]
        quality.append(dict(variavel=c,n=len(x),ausentes=x.null_count(),nao_finitos=int((~x.is_finite()).fill_null(False).sum()),negativos=int((x<0).sum()),nao_inteiros=int(((x-x.round(0)).abs()>EPS).sum()) if c in COUNTS else None,acima_100=int((x>100).sum()) if c in RATES else None))
    save('17_qualidade_auxiliares',pl.DataFrame(quality))
    if any(r['ausentes'] or r['nao_finitos'] or r['negativos'] or r['nao_inteiros'] or r['acima_100'] for r in quality) or base.filter(pl.col(N)<=0).height:
        raise ValueError('Taxas ou contagens inválidas; consulte 17_qualidade_auxiliares.')
    cats=KEYS[:2]+CATS+[KEYS[2],'idade_coorte']
    card=[]; test=pairs.filter(pl.col('ANO_ALVO')==2024); train=pairs.filter(pl.col('ANO_ALVO')<2024)
    for c in cats:
        f=base.filter(pl.col(c).is_not_null()).group_by(c).len()
        a=train.select(c).drop_nulls().unique(); b=test.select(c).drop_nulls().unique(); new=b.join(a,on=c,how='anti')
        card.append(dict(variavel=c,categorias=f.height,ausentes=base[c].null_count(),categorias_n_menor_20=f.filter(pl.col('len')<20).height,freq_mediana=f['len'].median(),categorias_treino=a.height,categorias_teste=b.height,categorias_ineditas=new.height,registros_ineditos=test.join(new,on=c,how='semi').height))
    save('18_cardinalidade',pl.DataFrame(card))
    save('19_distribuicao_coorte_idade',base.group_by([KEYS[2],'idade_coorte']).agg(distribution(TARGET)).sort([KEYS[2],'idade_coorte']))
    save('20_denominador',base.group_by(KEYS).agg(pl.col(N).n_unique().alias('n_valores')).group_by(KEYS[2]).agg(pl.len().alias('n_trajetorias'),(pl.col('n_valores')!=1).sum().alias('n_variavel')).sort(KEYS[2]))
    save('21_identidades_taxas',base.group_by(YEAR).agg(pl.len().alias('n'),*err(pl.col('TAP')+pl.col('TDA')+pl.col('TCA')-100,'soma'),*err(pl.col(TARGET)-100*pl.col(D)/pl.col(N),'tada'),*err(pl.col('TAP')-100*pl.col(P)/pl.col(N),'tap')).sort(YEAR))
    p=pairs.with_columns((100*pl.col(P)/pl.col(N)).alias('exposicao'),(pl.col(P)-pl.col(P+'_FUTURO')-pl.col(D+'_FUTURO')-pl.col(C+'_FUTURO')).alias('residuo_fluxo'),pl.when(pl.col(P)>0).then(pl.col(D+'_FUTURO')/pl.col(P)).otherwise(None).alias('risco'),(100*pl.col(D+'_FUTURO')/pl.col(N)).alias('tada_reconstruida'))
    a=pl.col('ANO_ALVO'); e=pl.col('exposicao'); y=pl.col('TADA_FUTURA'); r=pl.col('risco'); z=pl.col(P)==0
    save('22_consistencia_temporal',p.group_by('ANO_ALVO').agg(pl.len().alias('n'),(pl.col(N)!=pl.col(N+'_FUTURO')).sum().alias('n_denominador_mudou'),(pl.col('TDA_FUTURO')<pl.col('TDA')-tol).sum().alias('n_tda_reduziu'),(pl.col('TCA_FUTURO')<pl.col('TCA')-tol).sum().alias('n_tca_reduziu'),*err(pl.col('TDA_FUTURO')-pl.col('TDA')-y,'delta_tda'),(pl.col('residuo_fluxo')==0).sum().alias('n_fluxo_fecha'),(pl.col('residuo_fluxo')<0).sum().alias('n_fluxo_negativo'),(pl.col('residuo_fluxo')>0).sum().alias('n_fluxo_positivo'),pl.col('residuo_fluxo').abs().max().alias('residuo_max'),(pl.col(D+'_FUTURO')>pl.col(P)).sum().alias('n_desistentes_excedem'),(pl.col(D+'_FUTURO')+pl.col(C+'_FUTURO')>pl.col(P)).sum().alias('n_saidas_excedem'),(y>e+tol).sum().alias('n_tada_excede_exposicao'),*err(y-pl.col('tada_reconstruida'),'reconstrucao')).sort('ANO_ALVO'))
    save('23_zeros_estruturais',p.group_by('ANO_ALVO').agg(pl.len().alias('n'),z.sum().alias('n_sem_expostos'),pct(z).alias('pct_sem_expostos'),pct(y==0).alias('pct_zero_total'),(z & (y>0)).sum().alias('n_violacoes_tada'),(z & (pl.col(D+'_FUTURO')>0)).sum().alias('n_violacoes_contagem'),(z & (pl.col(P+'_FUTURO')>0)).sum().alias('n_reaparecimento')).sort('ANO_ALVO'))
    active=p.filter(pl.col(P)>0); valid=active.filter(r.is_between(0,1))
    valid=valid.with_columns(pl.when(pl.col(P)==1).then(pl.lit('01_1')).when(pl.col(P)<=5).then(pl.lit('02_2a5')).when(pl.col(P)<=10).then(pl.lit('03_6a10')).when(pl.col(P)<=20).then(pl.lit('04_11a20')).when(pl.col(P)<=50).then(pl.lit('05_21a50')).when(pl.col(P)<=100).then(pl.lit('06_51a100')).otherwise(pl.lit('07_acima100')).alias('faixa_expostos'))
    def riskstats():
        return distribution('risco_pct')+[(100*pl.col(D+'_FUTURO').sum()/pl.col(P).sum()).alias('risco_ponderado'),pct(r==1).alias('pct_cem')]
    valid=valid.with_columns((100*r).alias('risco_pct'))
    for name,groups in [('24_risco_ano',['ANO_ALVO']),('25_risco_idade',['idade_coorte']),('26_risco_coorte_ano',[KEYS[2],'ANO_ALVO']),('27_risco_tamanho',['ANO_ALVO','faixa_expostos'])]:
        save(name,valid.group_by(groups).agg(riskstats()).sort(groups))
    cr=[]
    for year in sorted(p['ANO_ALVO'].unique().to_list()):
        for name,df,target in [('tada',p,'TADA_FUTURA'),('risco',valid,'risco')]:
            t=df.filter(a==year)
            for c in COUNTS+RATES+['idade_coorte','exposicao','tada_lag1','tada_lag2','tada_lag3']:
                t1=t.filter(pl.col(c).is_not_null()); cr.append(dict(ano_alvo=year,alvo=name,variavel=c,n=t1.height,pearson=t1.select(corr(c,target)).item() if t1.height>1 else None,spearman=t1.select(pl.corr(c,target,method='spearman').fill_nan(None)).item() if t1.height>1 else None))
    save('28_correlacoes_alvos',pl.DataFrame(cr))
    # Históricos somente de alvos anteriores ao período avaliado.
    rows=[]; het=[]
    for year in (2022,2023,2024):
        tr=p.filter(a<year); te=p.filter(a==year); vr=valid.filter(a<year)
        if tr.is_empty() or te.is_empty() or vr.is_empty(): continue
        if te.filter(~e.is_between(0,100)).height: raise ValueError('Exposição fora de [0,100]; investigar antes de comparar.')
        gm=tr['TADA_FUTURA'].mean(); gd=tr['TADA_FUTURA'].median(); gr=vr['risco'].mean(); grd=vr['risco'].median(); gw=vr[D+'_FUTURO'].sum()/vr[P].sum()
        h=tr.group_by('idade_coorte').agg(y.mean().alias('mi'),y.median().alias('md'))
        hr=vr.group_by('idade_coorte').agg(r.mean().alias('ri'),r.median().alias('rd'),(pl.col(D+'_FUTURO').sum()/pl.col(P).sum()).alias('rw'))
        te=te.join(h,on='idade_coorte',how='left',validate='m:1').join(hr,on='idade_coorte',how='left',validate='m:1').with_columns(pl.col('mi').fill_null(gm),pl.col('md').fill_null(gd),pl.col('ri').fill_null(gr),pl.col('rd').fill_null(grd),pl.col('rw').fill_null(gw))
        def evaluate(name, df, pred, coverage=1.0):
            yy=df['TADA_FUTURA'].to_numpy(); ee=df['exposicao'].to_numpy(); pp=np.asarray(pred,dtype=float); error=pp-yy; pos=yy>0; act=ee>0; ss=np.sum((yy-yy.mean())**2)
            rows.append(dict(ano_alvo=year,modelo=name,n=len(yy),mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),r2=float(1-np.sum(error**2)/ss) if ss>0 else None,vies=float(error.mean()),mae_expostos=float(np.abs(error[act]).mean()) if act.any() else None,mae_positivos=float(np.abs(error[pos]).mean()) if pos.any() else None,mae_ponderado_ingressantes=float(np.average(np.abs(error),weights=df[N].to_numpy())),pct_pred_zero=float(100*np.mean(pp==0)),pct_pred_acima_exposicao=float(100*np.mean(pp>ee+tol)),cobertura_historico=float(coverage)))
        ee=te['exposicao'].to_numpy(); pp=te[TARGET].to_numpy(); mi=te['mi'].to_numpy(); md=te['md'].to_numpy()
        preds={'zero':np.zeros(len(ee)),'media_global':np.full(len(ee),gm),'mediana_global':np.full(len(ee),gd),'persistencia':pp,'media_idade':mi,'mediana_idade':md,'risco_global':ee*gr,'risco_global_ponderado':ee*gw,'risco_medio_idade':ee*te['ri'].to_numpy(),'risco_mediano_idade':ee*te['rd'].to_numpy(),'risco_ponderado_idade':ee*te['rw'].to_numpy()}
        for n,v in [('persistencia',pp),('media_idade',mi),('mediana_idade',md)]:
            preds[n+'_zero_estrutural']=np.where(ee==0,0,v); preds[n+'_limitada']=np.minimum(v,ee)
        for n,v in preds.items(): evaluate(n,te,v)
        for groups in [['CO_IES'],['CO_CURSO'],KEYS[:2],['CO_IES','idade_coorte'],['CO_CURSO','idade_coorte'],KEYS[:2]+['idade_coorte']]:
            name='_'.join(groups); hist=tr.group_by(groups).agg(pl.len().alias('nh'),y.mean().alias('mh'),y.median().alias('dh'),y.std().alias('sd'))
            het.append(dict(ano_alvo=year,grupo=name,n_grupos=hist.height,n_grupos_20=hist.filter(pl.col('nh')>=20).height,dp_entre_medias=hist['mh'].std(),dp_interno_medio=hist['sd'].mean()))
            q=te.join(hist,on=groups,how='left',validate='m:1'); nn=q['nh'].fill_null(0).to_numpy(); m=q['mi'].to_numpy(); exp=q['exposicao'].to_numpy(); cov=np.mean(nn>=20)
            smooth=(nn*q['mh'].fill_null(0).to_numpy()+20*m)/(nn+20)
            for label,v in [('media',np.where(nn>=20,q['mh'].fill_null(gm).to_numpy(),m)),('mediana',np.where(nn>=20,q['dh'].fill_null(gd).to_numpy(),q['md'].to_numpy())),('suavizada',smooth)]:
                evaluate(label+'_'+name,q,v,cov); evaluate(label+'_'+name+'_limitada',q,np.minimum(v,exp),cov)
            rh=vr.group_by(groups).agg(pl.col(P).sum().alias('np_hist'),pl.col(D+'_FUTURO').sum().alias('nd_hist'))
            q=te.join(rh,on=groups,how='left',validate='m:1'); np_hist=q['np_hist'].fill_null(0).to_numpy(); nd_hist=q['nd_hist'].fill_null(0).to_numpy(); rr=(nd_hist+20*q['rw'].to_numpy())/(np_hist+20)
            evaluate('risco_suavizado_'+name,q,q['exposicao'].to_numpy()*rr,np.mean(np_hist>0))
    if rows:
        result=pl.DataFrame(rows); save('29_baselines_ampliados',result.sort(['ano_alvo','mae'])); save('30_ranking_validacoes',result.filter(pl.col('ano_alvo').is_in([2022,2023])).group_by('modelo').agg(pl.len().alias('n_validacoes'),*[pl.col(c).mean().alias(c) for c in ['mae','rmse','r2','vies','mae_expostos','mae_positivos']]).sort('mae')); print('BASELINES 2024 (10 melhores MAE):'); print(result.filter(pl.col('ano_alvo')==2024).sort('mae').head(10).write_csv(separator='\t'))
    if het: save('31_heterogeneidade',pl.DataFrame(het))
    return dict(n_sem_expostos=p.filter(z).height,n_violacoes_zero=p.filter(z & (y>0)).height,n_risco_valido=valid.height,n_risco_invalido=active.height-valid.height,n_tada_acima_exposicao=p.filter(y>e+tol).height,nota_exposicao='Restrição exata nas contagens deve ser verificada; TADA publicada pode divergir da reconstrução. 2024 é diagnóstico retrospectivo; parâmetros de suavização exploratórios.')


if __name__=='__main__':
    main()
