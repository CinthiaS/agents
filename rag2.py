import pandas as pd
import numpy as np
import pymc as pm

# 1. Simular dados
np.random.seed(42)
n = 300
setores = ['Indústria', 'Serviços', 'Financeiro']
portes = ['Pequeno', 'Médio', 'Grande']
mu_sigma = {
    ('Indústria', 'Pequeno'): (2.5, 0.5),
    ('Indústria', 'Médio'): (2.0, 0.4),
    ('Indústria', 'Grande'): (1.5, 0.3),
    ('Serviços', 'Pequeno'): (1.8, 0.3),
    ('Serviços', 'Médio'): (1.6, 0.3),
    ('Serviços', 'Grande'): (1.2, 0.2),
    ('Financeiro', 'Pequeno'): (8.0, 1.0),
    ('Financeiro', 'Médio'): (7.0, 0.8),
    ('Financeiro', 'Grande'): (6.0, 0.6),
}
df = pd.DataFrame({
    'setor': np.random.choice(setores, n),
    'porte': np.random.choice(portes, n)
})
df['grupo'] = df['setor'] + '-' + df['porte']
df['alavancagem'] = [np.random.normal(*mu_sigma[(s, p)]) for s, p in zip(df['setor'], df['porte'])]

# 2. Função para modelar cada grupo com PyMC
def modelar_grupo(grupo_nome, grupo_df):
    dados = grupo_df['alavancagem'].values
    with pm.Model() as modelo:
        mu = pm.Normal("mu", mu=np.mean(dados), sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=dados)
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, progressbar=False)
        pred = pm.sample_posterior_predictive(trace, var_names=["obs"], random_seed=42, progressbar=False)

    amostras = pred["obs"].flatten()
    q1 = np.percentile(amostras, 25)
    q2 = np.percentile(amostras, 50)
    q3 = np.percentile(amostras, 75)
    return {
        "grupo": grupo_nome,
        "Q1 (Saudável)": round(q1, 2),
        "Q2 (Aceitável)": round(q2, 2),
        "Q3 (Ruim)": round(q3, 2),
    }

# 3. Aplicar modelagem a cada grupo
resultados = []
for grupo_nome, grupo_df in df.groupby('grupo'):
    resultados.append(modelar_grupo(grupo_nome, grupo_df))

resultados_df = pd.DataFrame(resultados)
print(resultados_df)
