

```python
import keras
import pandas
import sklearn
import scipy
import dask.dataframe as dd
import numpy
```

    C:\Users\danil\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
# use this code to transfer between csv to parquet
df.to_parquet("train_data.parquet")
```


```python
training_data = pandas.read_csv("train_data.csv")
```


```python
training_sample = training_data.sample(frac=0.3)
```


```python
training_sample.shape
```


```python
training_sample.to_csv("train_data_sample.csv", index=False)
```

# read sample if already available


```python
training_data = pandas.read_csv("train_data_sample.csv")
```


```python
training_data.columns
```




    Index(['id', 'UF_1', 'UF_2', 'UF_3', 'UF_4', 'UF_5', 'UF_6', 'UF_7', 'IDADE',
           'SEXO_1',
           ...
           'CEP4_7', 'CEP4_8', 'CEP4_9', 'CEP4_10', 'CEP4_11', 'CEP4_12',
           'CEP4_13', 'CEP4_14', 'IND_BOM_1_1', 'IND_BOM_1_2'],
          dtype='object', length=246)




```python
# keep ids and their index on database for further reference
ids = training_data["id"]
```


```python
features = training_data.drop(["IND_BOM_1_1", "IND_BOM_1_2", "id"], axis=1)
```


```python
features.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <th>...</th>
      <th>CEP4_5</th>
      <th>CEP4_6</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217846</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.750400</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.074953</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.355855</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.930834</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.678045</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.485231</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.654419</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.358808</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.132485</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 243 columns</p>
</div>




```python
labels = training_data["IND_BOM_1_1"]
```

Vou tentar reduzir a dimensionalidade do dataframe utilizando LDA para poder analisar 
melhor um subconjunto de variáveis e mensurar a acurácia

## Data Cleaning


```python
training_data.dtypes.value_counts()
```




    float64    144
    int64      102
    dtype: int64



Pandas leu muito dos valores de forma errada, gerando até problemas para o uso de memória, existem formas
melhores de representar estas variáveis.


```python
training_data.columns
```




    Index(['id', 'UF_1', 'UF_2', 'UF_3', 'UF_4', 'UF_5', 'UF_6', 'UF_7', 'IDADE',
           'SEXO_1',
           ...
           'CEP4_7', 'CEP4_8', 'CEP4_9', 'CEP4_10', 'CEP4_11', 'CEP4_12',
           'CEP4_13', 'CEP4_14', 'IND_BOM_1_1', 'IND_BOM_1_2'],
          dtype='object', length=246)




```python
training_data.dtypes
```




    id                                  int64
    UF_1                                int64
    UF_2                                int64
    UF_3                                int64
    UF_4                                int64
    UF_5                                int64
    UF_6                                int64
    UF_7                                int64
    IDADE                             float64
    SEXO_1                              int64
    NIVEL_RELACIONAMENTO_CREDITO01    float64
    NIVEL_RELACIONAMENTO_CREDITO02    float64
    BANCO_REST_IRPF_ULTIMA_1            int64
    BANCO_REST_IRPF_ULTIMA_2            int64
    BANCO_REST_IRPF_ULTIMA_3            int64
    BANCO_REST_IRPF_ULTIMA_4            int64
    BANCO_REST_IRPF_ULTIMA_5            int64
    BANCO_REST_IRPF_ULTIMA_6            int64
    BANCO_REST_IRPF_ULTIMA_7            int64
    ATIVIDADE_EMAIL                   float64
    EXPOSICAO_ENDERECO                float64
    EXPOSICAO_EMAIL                   float64
    EXPOSICAO_TELEFONE                float64
    ATIVIDADE_ENDERECO                float64
    ATUALIZACAO_ENDERECO              float64
    ATUALIZACAO_EMAIL                 float64
    EXPOSICAO_CONSUMIDOR_EMAILS       float64
    EXPOSICAO_CONSUMIDOR_TELEFONES    float64
    ATIVIDADE_TELEFONE                float64
    VALOR_PARCELA_BOLSA_FAMILIA       float64
                                       ...   
    CEP2_8                              int64
    CEP2_9                              int64
    CEP3_1                              int64
    CEP3_2                              int64
    CEP3_3                              int64
    CEP3_4                              int64
    CEP3_5                              int64
    CEP3_6                              int64
    CEP3_7                              int64
    CEP3_8                              int64
    CEP3_9                              int64
    CEP3_10                             int64
    CEP3_11                             int64
    CEP3_12                             int64
    CEP4_1                              int64
    CEP4_2                              int64
    CEP4_3                              int64
    CEP4_4                              int64
    CEP4_5                              int64
    CEP4_6                              int64
    CEP4_7                              int64
    CEP4_8                              int64
    CEP4_9                              int64
    CEP4_10                             int64
    CEP4_11                             int64
    CEP4_12                             int64
    CEP4_13                             int64
    CEP4_14                             int64
    IND_BOM_1_1                         int64
    IND_BOM_1_2                         int64
    Length: 246, dtype: object




```python
category_columns = ["UF_1", "UF_2", "UF_3", "UF_4", "UF_5", "UF_6", "UF_7",
                   "BANCO_REST_IRPF_ULTIMA_1", "BANCO_REST_IRPF_ULTIMA_2", "BANCO_REST_IRPF_ULTIMA_3",
                   "BANCO_REST_IRPF_ULTIMA_4", "BANCO_REST_IRPF_ULTIMA_5", "BANCO_REST_IRPF_ULTIMA_6",
                   "BANCO_REST_IRPF_ULTIMA_7", "FLAG_BOLSA_FAMILIA_1", "SIGLA_PARTIDO_FILIADO_1",
                   "SIGLA_PARTIDO_FILIADO_2", "SIGLA_PARTIDO_FILIADO_3", "SIGLA_PARTIDO_FILIADO_4",
                   "SIGLA_PARTIDO_FILIADO_5", "SIGLA_PARTIDO_FILIADO_6", "SIGLA_PARTIDO_FILIADO_7",
                   "FLAG_FILIADO_PARTIDO_POLITICO_1", "FLAG_PROUNI_1", "RENDA_VIZINHANCA_1", 
                   "RENDA_VIZINHANCA_2", "RENDA_VIZINHANCA_3", "RENDA_VIZINHANCA_4", 
                    "COMPARATIVO_RENDA_CEP_1", "COMPARATIVO_RENDA_CEP_2", "COMPARATIVO_RENDA_CEP_3",
                   "COMPARATIVO_RENDA_CEP_4", "COMPARATIVO_RENDA_CEP_5", "CLASSE_SOCIAL_CONSUMIDOR_1",
                   "CLASSE_SOCIAL_CONSUMIDOR_2", "CLASSE_SOCIAL_CONSUMIDOR_3", "CLASSE_SOCIAL_CONSUMIDOR_4",
                   "FLAG_REDE_SOCIAL_1", "FLAG_REDE_SOCIAL_2", "FLAG_REDE_SOCIAL_3",
                   "CEP1_1", "CEP1_2", "CEP1_3", "CEP1_4", "CEP1_5", "CEP2_1", "CEP2_2", "CEP2_3", "CEP2_4",
                   "CEP2_5", "CEP2_6", "CEP2_7", "CEP2_8", "CEP2_9", "CEP3_1", "CEP3_2", "CEP3_3", "CEP3_4",
                   "CEP3_5", "CEP3_6", "CEP3_7", "CEP3_8", "CEP3_9", "CEP3_10", "CEP3_11", "CEP3_12",
                   "CEP4_1", "CEP4_2", "CEP4_3", "CEP4_4", "CEP4_5", "CEP4_6", "CEP4_7", "CEP4_8", "CEP4_9",
                   "CEP4_10", "CEP4_11", "CEP4_12", "CEP4_13", "CEP4_14"]

ordered_category_columns = ["NIVEL_RELACIONAMENTO_CREDITO02", "EXPOSICAO_CONSUMIDOR_EMAILS", 
                            "EXPOSICAO_CONSUMIDOR_TELEFONES"]
```


```python
for column in training_data.columns:
    print(column)
    values = training_data[column].value_counts()
    if type(values) == list:
        print(values[:10])
    else:
        print(values.head(10))
    print(len(values))
    
```

    id
    4094      1
    270713    1
    210764    1
    206670    1
    112591    1
    259924    1
    261973    1
    56484     1
    336343    1
    108493    1
    Name: id, dtype: int64
    116759
    UF_1
    1    103790
    0     12969
    Name: UF_1, dtype: int64
    2
    UF_2
    1    80712
    0    36047
    Name: UF_2, dtype: int64
    2
    UF_3
    0    61114
    1    55645
    Name: UF_3, dtype: int64
    2
    UF_4
    0    82157
    1    34602
    Name: UF_4, dtype: int64
    2
    UF_5
    0    88506
    1    28253
    Name: UF_5, dtype: int64
    2
    UF_6
    0    91341
    1    25418
    Name: UF_6, dtype: int64
    2
    UF_7
    0    94902
    1    21857
    Name: UF_7, dtype: int64
    2
    IDADE
    5.506237e-16    1273
    1.000000e+00    1158
    3.078642e-01      36
    2.940066e-01      34
    3.008786e-01      32
    3.130324e-01      29
    2.189822e-01      24
    3.908966e-01      21
    3.636924e-01      21
    5.574156e-01      21
    Name: IDADE, dtype: int64
    16918
    SEXO_1
    1    60606
    0    56153
    Name: SEXO_1, dtype: int64
    2
    NIVEL_RELACIONAMENTO_CREDITO01
    0.111111    110419
    1.000000      1533
    0.000000      1324
    0.222222       733
    0.888889       652
    0.777778       612
    0.666667       450
    0.333333       393
    0.444444       341
    0.555556       302
    Name: NIVEL_RELACIONAMENTO_CREDITO01, dtype: int64
    10
    NIVEL_RELACIONAMENTO_CREDITO02
    0.0    116456
    1.0       213
    0.5        90
    Name: NIVEL_RELACIONAMENTO_CREDITO02, dtype: int64
    3
    BANCO_REST_IRPF_ULTIMA_1
    0    107813
    1      8946
    Name: BANCO_REST_IRPF_ULTIMA_1, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_2
    0    112951
    1      3808
    Name: BANCO_REST_IRPF_ULTIMA_2, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_3
    0    114747
    1      2012
    Name: BANCO_REST_IRPF_ULTIMA_3, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_4
    0    114984
    1      1775
    Name: BANCO_REST_IRPF_ULTIMA_4, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_5
    0    115408
    1      1351
    Name: BANCO_REST_IRPF_ULTIMA_5, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_6
    1    108196
    0      8563
    Name: BANCO_REST_IRPF_ULTIMA_6, dtype: int64
    2
    BANCO_REST_IRPF_ULTIMA_7
    1    107430
    0      9329
    Name: BANCO_REST_IRPF_ULTIMA_7, dtype: int64
    2
    ATIVIDADE_EMAIL
    5.881235e-18    83340
    5.694352e-02     2341
    9.129151e-02      844
    9.102106e-02      842
    7.695731e-02      756
    3.720255e-01      409
    1.000000e+00      371
    3.493072e-01      143
    3.717551e-01      128
    3.660755e-01      126
    Name: ATIVIDADE_EMAIL, dtype: int64
    2807
    EXPOSICAO_ENDERECO
    0.000000    31879
    0.013889    13268
    0.041667    12090
    0.027778     8784
    0.055556     7927
    0.111111     5344
    0.069444     5299
    0.083333     3522
    0.097222     3339
    0.125000     2879
    Name: EXPOSICAO_ENDERECO, dtype: int64
    73
    EXPOSICAO_EMAIL
    0.000000    84565
    0.031579     5424
    0.010526     4789
    0.052632     2817
    0.073684     1867
    0.094737     1817
    0.115789     1149
    0.136842     1030
    0.063158      973
    0.042105      875
    Name: EXPOSICAO_EMAIL, dtype: int64
    96
    EXPOSICAO_TELEFONE
    0.000000    69635
    0.027778    11620
    0.055556     8867
    0.083333     5251
    0.111111     4041
    0.138889     2928
    0.166667     2225
    0.194444     1786
    0.222222     1404
    0.250000     1192
    Name: EXPOSICAO_TELEFONE, dtype: int64
    37
    ATIVIDADE_ENDERECO
    1.473561e-01    68534
    1.144370e-01    12141
    1.055877e-01     5651
    2.044015e-16     2024
    1.519577e-01     1933
    1.526657e-01     1873
    1.516038e-01     1489
    1.523117e-01     1311
    1.000000e+00     1231
    1.919563e-01     1172
    Name: ATIVIDADE_ENDERECO, dtype: int64
    1636
    ATUALIZACAO_ENDERECO
    0.398645    26634
    0.979968    24862
    0.815862    22245
    0.002289    11671
    0.850630    10176
    0.003680     5719
    0.593347     2636
    0.084342     1387
    0.796392     1331
    1.000000     1104
    Name: ATUALIZACAO_ENDERECO, dtype: int64
    305
    ATUALIZACAO_EMAIL
    -5.140958e-17    83364
     8.325188e-02     6086
     1.081891e-01     1854
     2.562665e-02     1391
     1.257126e-01     1307
     1.260496e-01     1219
     2.596364e-02      815
     5.070283e-03      728
     1.000000e+00      331
     2.528966e-02      292
    Name: ATUALIZACAO_EMAIL, dtype: int64
    2108
    EXPOSICAO_CONSUMIDOR_EMAILS
    0.0    83063
    0.2    19874
    0.4     7631
    0.6     3225
    1.0     1506
    0.8     1460
    Name: EXPOSICAO_CONSUMIDOR_EMAILS, dtype: int64
    6
    EXPOSICAO_CONSUMIDOR_TELEFONES
    0.1    40317
    0.0    38727
    0.2    13083
    0.3     8604
    0.4     5232
    0.5     3379
    0.6     2295
    1.0     1667
    0.7     1575
    0.8     1072
    Name: EXPOSICAO_CONSUMIDOR_TELEFONES, dtype: int64
    11
    ATIVIDADE_TELEFONE
    -2.188272e-17    39501
     1.388212e-01    27555
     6.681370e-02     5279
     1.698044e-01     1620
     1.692307e-01     1491
     8.574793e-02     1480
     1.689438e-01     1122
     1.695176e-01     1065
     2.016484e-01      838
     1.000000e+00      797
    Name: ATIVIDADE_TELEFONE, dtype: int64
    2407
    VALOR_PARCELA_BOLSA_FAMILIA
    0.000000    92617
    0.098958     2060
    0.182292     1738
    0.117188     1600
    0.208333     1569
    0.265625     1529
    0.299479      926
    0.348958      737
    0.122396      586
    0.138021      530
    Name: VALOR_PARCELA_BOLSA_FAMILIA, dtype: int64
    313
    FLAG_BOLSA_FAMILIA_1
    0    91921
    1    24838
    Name: FLAG_BOLSA_FAMILIA_1, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_1
    1    112666
    0      4093
    Name: SIGLA_PARTIDO_FILIADO_1, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_2
    1    104315
    0     12444
    Name: SIGLA_PARTIDO_FILIADO_2, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_3
    1    99617
    0    17142
    Name: SIGLA_PARTIDO_FILIADO_3, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_4
    0    106740
    1     10019
    Name: SIGLA_PARTIDO_FILIADO_4, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_5
    0    108091
    1      8668
    Name: SIGLA_PARTIDO_FILIADO_5, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_6
    0    108660
    1      8099
    Name: SIGLA_PARTIDO_FILIADO_6, dtype: int64
    2
    SIGLA_PARTIDO_FILIADO_7
    0    109866
    1      6893
    Name: SIGLA_PARTIDO_FILIADO_7, dtype: int64
    2
    FLAG_FILIADO_PARTIDO_POLITICO_1
    0    90104
    1    26655
    Name: FLAG_FILIADO_PARTIDO_POLITICO_1, dtype: int64
    2
    FLAG_PROUNI_1
    1    67550
    0    49209
    Name: FLAG_PROUNI_1, dtype: int64
    2
    RENDA_VIZINHANCA_1
    0    107379
    1      9380
    Name: RENDA_VIZINHANCA_1, dtype: int64
    2
    RENDA_VIZINHANCA_2
    1    112951
    0      3808
    Name: RENDA_VIZINHANCA_2, dtype: int64
    2
    RENDA_VIZINHANCA_3
    0    113221
    1      3538
    Name: RENDA_VIZINHANCA_3, dtype: int64
    2
    RENDA_VIZINHANCA_4
    1    107649
    0      9110
    Name: RENDA_VIZINHANCA_4, dtype: int64
    2
    QUANTIDADE_VIZINHANCA
    1.000000    1694
    0.000000     564
    0.991063     216
    0.901305     183
    0.000304     170
    0.397017     169
    0.696642     168
    0.000393     160
    0.734648     159
    0.000405     157
    Name: QUANTIDADE_VIZINHANCA, dtype: int64
    7416
    COMPARATIVO_RENDA_CEP_1
    1    112355
    0      4404
    Name: COMPARATIVO_RENDA_CEP_1, dtype: int64
    2
    COMPARATIVO_RENDA_CEP_2
    0    58715
    1    58044
    Name: COMPARATIVO_RENDA_CEP_2, dtype: int64
    2
    COMPARATIVO_RENDA_CEP_3
    0    77128
    1    39631
    Name: COMPARATIVO_RENDA_CEP_3, dtype: int64
    2
    COMPARATIVO_RENDA_CEP_4
    0    100398
    1     16361
    Name: COMPARATIVO_RENDA_CEP_4, dtype: int64
    2
    COMPARATIVO_RENDA_CEP_5
    0    109632
    1      7127
    Name: COMPARATIVO_RENDA_CEP_5, dtype: int64
    2
    CLASSE_SOCIAL_CONSUMIDOR_1
    0    91097
    1    25662
    Name: CLASSE_SOCIAL_CONSUMIDOR_1, dtype: int64
    2
    CLASSE_SOCIAL_CONSUMIDOR_2
    1    107154
    0      9605
    Name: CLASSE_SOCIAL_CONSUMIDOR_2, dtype: int64
    2
    CLASSE_SOCIAL_CONSUMIDOR_3
    0    108210
    1      8549
    Name: CLASSE_SOCIAL_CONSUMIDOR_3, dtype: int64
    2
    CLASSE_SOCIAL_CONSUMIDOR_4
    1    92153
    0    24606
    Name: CLASSE_SOCIAL_CONSUMIDOR_4, dtype: int64
    2
    ATIVIDADE_CONSUMIDOR_MERCADO_FINANCEIRO
    0.000000    86939
    0.050505     1052
    0.090909      946
    0.161616      926
    0.040404      923
    0.262626      917
    0.020202      916
    0.010101      903
    0.080808      817
    0.252525      803
    Name: ATIVIDADE_CONSUMIDOR_MERCADO_FINANCEIRO, dtype: int64
    100
    ATUALIZACAO_CONSUMIDOR_MERCADO_FINANCEIRO
    0.000000    90834
    0.013158     1991
    0.039474     1830
    0.078947     1644
    0.026316     1629
    0.065789     1321
    0.052632     1228
    0.131579     1006
    0.118421      971
    0.092105      956
    Name: ATUALIZACAO_CONSUMIDOR_MERCADO_FINANCEIRO, dtype: int64
    77
    FLAG_PROGRAMAS_SOCIAIS_1
    0    90894
    1    25865
    Name: FLAG_PROGRAMAS_SOCIAIS_1, dtype: int64
    2
    CAD_DEMOGRAFICO_VAR_0
    0.000000    14990
    0.048492      282
    0.955691      234
    0.810936      216
    0.253486      206
    0.963828      183
    0.970214      169
    0.847030      168
    0.890235      159
    0.935111      150
    Name: CAD_DEMOGRAFICO_VAR_0, dtype: int64
    30774
    CAD_DEMOGRAFICO_VAR_1
    0.000000    14990
    0.040312      282
    0.396743      234
    0.873684      216
    0.209010      206
    0.818808      183
    0.906474      169
    0.608767      168
    0.701063      159
    0.863097      150
    Name: CAD_DEMOGRAFICO_VAR_1, dtype: int64
    31585
    CAD_DEMOGRAFICO_VAR_2
    0.000000    14990
    0.989493      284
    0.838472      234
    0.885520      216
    0.998899      206
    0.535248      183
    0.779409      169
    0.998439      168
    0.564296      159
    0.359907      150
    Name: CAD_DEMOGRAFICO_VAR_2, dtype: int64
    31447
    CAD_DEMOGRAFICO_VAR_3
    0.000000    14990
    0.987110      282
    0.995386      234
    0.008934      216
    0.972762      207
    0.693235      183
    0.489917      169
    0.052614      168
    0.004468      159
    0.109779      150
    Name: CAD_DEMOGRAFICO_VAR_3, dtype: int64
    31723
    CAD_DEMOGRAFICO_VAR_4
    0.000000    14990
    0.022697      283
    0.045476      234
    0.963439      216
    0.043169      206
    0.968847      183
    0.976295      169
    0.789861      168
    0.998518      159
    0.999176      152
    Name: CAD_DEMOGRAFICO_VAR_4, dtype: int64
    31454
    CAD_DEMOGRAFICO_VAR_5
    0.000000    14990
    0.012063      282
    0.269885      234
    0.943856      216
    0.282468      206
    0.978815      184
    0.987182      169
    0.993987      168
    0.899048      159
    0.945438      150
    Name: CAD_DEMOGRAFICO_VAR_5, dtype: int64
    30668
    CAD_DEMOGRAFICO_VAR_6
    0.000000    14990
    0.958932      282
    0.449129      234
    0.710904      216
    0.862243      207
    0.331384      183
    0.992104      169
    0.779666      168
    0.503754      159
    0.605663      150
    Name: CAD_DEMOGRAFICO_VAR_6, dtype: int64
    31617
    CAD_DEMOGRAFICO_VAR_7
    0.000000    14990
    0.985317      282
    0.882730      234
    0.008898      216
    0.995418      208
    0.065813      183
    0.017847      169
    0.335856      168
    0.031452      160
    0.002856      150
    Name: CAD_DEMOGRAFICO_VAR_7, dtype: int64
    31663
    CAD_DEMOGRAFICO_VAR_8
    0.000000    14990
    0.003971      282
    0.999475      245
    0.922812      216
    0.807542      206
    0.005801      183
    0.826861      169
    0.853727      168
    0.018307      159
    0.028198      150
    Name: CAD_DEMOGRAFICO_VAR_8, dtype: int64
    15841
    CAD_DEMOGRAFICO_VAR_10
    0.000000    14990
    0.993323      282
    0.336118      234
    0.641228      216
    0.033679      206
    0.976283      183
    0.503201      169
    0.320616      168
    0.082427      159
    0.224403      150
    Name: CAD_DEMOGRAFICO_VAR_10, dtype: int64
    31718
    CAD_DEMOGRAFICO_VAR_11
    0.000000    14990
    0.007490      282
    0.039769      234
    0.887487      216
    0.555661      206
    0.096545      183
    0.992439      180
    0.627339      169
    0.489805      159
    0.954182      150
    Name: CAD_DEMOGRAFICO_VAR_11, dtype: int64
    31789
    CAD_DEMOGRAFICO_VAR_13
    0.000000    14990
    0.003422      284
    0.998974      235
    0.010760      217
    0.304410      206
    0.039413      183
    0.016907      169
    0.370168      168
    0.073925      159
    0.043512      150
    Name: CAD_DEMOGRAFICO_VAR_13, dtype: int64
    29538
    CAD_DEMOGRAFICO_VAR_14
    0.000000    14990
    0.002933      282
    0.792059      234
    0.838217      216
    0.057010      206
    0.998249      183
    0.969553      169
    0.964569      168
    0.497979      159
    0.405989      150
    Name: CAD_DEMOGRAFICO_VAR_14, dtype: int64
    31450
    CAD_DEMOGRAFICO_VAR_15
    0.000000    14990
    0.976251      282
    0.959851      234
    0.799402      217
    0.947507      206
    0.944919      183
    0.929839      169
    0.441613      168
    0.792720      159
    0.637405      150
    Name: CAD_DEMOGRAFICO_VAR_15, dtype: int64
    31718
    CAD_DEMOGRAFICO_VAR_16
    0.000000    14990
    0.004232      282
    0.901381      234
    0.045935      216
    0.080766      206
    0.080150      183
    0.134445      169
    0.620366      168
    0.014748      159
    0.507049      150
    Name: CAD_DEMOGRAFICO_VAR_16, dtype: int64
    31728
    CAD_DEMOGRAFICO_VAR_17
    0.000000    14990
    0.998928      302
    0.037255      282
    0.964458      234
    0.000253      217
    0.963185      183
    0.989980      169
    0.967579      168
    0.998417      161
    0.991472      151
    Name: CAD_DEMOGRAFICO_VAR_17, dtype: int64
    25880
    CAD_DEMOGRAFICO_VAR_19
    0.000000    14990
    0.000037      371
    0.999998      348
    0.999987      327
    0.000063      320
    0.999999      319
    0.999996      288
    0.002678      282
    0.999995      278
    0.999985      269
    Name: CAD_DEMOGRAFICO_VAR_19, dtype: int64
    17759
    CAD_DEMOGRAFICO_VAR_21
    0.000000    14990
    0.017351      282
    0.991670      234
    0.634559      216
    0.238135      206
    0.364606      183
    0.170363      169
    0.100960      168
    0.216848      159
    0.505530      150
    Name: CAD_DEMOGRAFICO_VAR_21, dtype: int64
    31671
    CAD_DEMOGRAFICO_VAR_22
    0.000000    14990
    0.031813      282
    0.993852      234
    0.583199      216
    0.982244      207
    0.009870      183
    0.010279      169
    0.008048      168
    0.145849      159
    0.079105      150
    Name: CAD_DEMOGRAFICO_VAR_22, dtype: int64
    31198
    CAD_DEMOGRAFICO_VAR_23
    0.000000    14990
    0.992942      282
    0.136754      234
    0.673335      216
    0.994052      206
    0.024422      183
    0.527489      169
    0.397584      168
    0.067695      159
    0.209691      150
    Name: CAD_DEMOGRAFICO_VAR_23, dtype: int64
    31830
    CAD_DEMOGRAFICO_VAR_24
    0.000000    14990
    0.978205      282
    0.615352      234
    0.722526      216
    0.095653      206
    0.006292      183
    0.120610      169
    0.324291      168
    0.322618      159
    0.960056      150
    Name: CAD_DEMOGRAFICO_VAR_24, dtype: int64
    31717
    CAD_DEMOGRAFICO_VAR_25
    0.000000    14990
    0.791048      282
    0.838127      234
    0.969985      216
    0.772585      206
    0.015573      184
    0.016869      171
    0.761632      169
    0.474924      159
    0.760417      150
    Name: CAD_DEMOGRAFICO_VAR_25, dtype: int64
    29850
    CAD_DEMOGRAFICO_VAR_26
    0.000000    14990
    0.025606      282
    0.994045      234
    0.991948      216
    0.989887      206
    0.992075      183
    0.990596      169
    0.973544      168
    0.987543      159
    0.963862      150
    Name: CAD_DEMOGRAFICO_VAR_26, dtype: int64
    31726
    CAD_DEMOGRAFICO_VAR_27
    0.000000    14990
    0.973584      282
    0.923571      234
    0.671476      216
    0.458836      206
    0.987169      183
    0.760318      169
    0.496887      168
    0.160566      159
    0.709441      150
    Name: CAD_DEMOGRAFICO_VAR_27, dtype: int64
    31500
    CAD_DEMOGRAFICO_VAR_28
    0.000000    14990
    0.025227      282
    0.480200      234
    0.911405      216
    0.045757      206
    0.968164      190
    0.618670      169
    0.069670      168
    0.507606      159
    0.957693      150
    Name: CAD_DEMOGRAFICO_VAR_28, dtype: int64
    31836
    CAD_DEMOGRAFICO_VAR_30
    0.000000    14990
    0.013986      282
    0.424137      234
    0.156681      216
    0.999883      207
    0.315356      183
    0.693926      169
    0.794873      169
    0.982943      159
    0.760080      150
    Name: CAD_DEMOGRAFICO_VAR_30, dtype: int64
    31661
    CAD_DEMOGRAFICO_VAR_31
    0.000000    14990
    0.987138      282
    0.095134      234
    0.036774      216
    0.500496      206
    0.021727      183
    0.019823      169
    0.035938      168
    0.017232      159
    0.165333      150
    Name: CAD_DEMOGRAFICO_VAR_31, dtype: int64
    31467
    CAD_DEMOGRAFICO_VAR_33
    0.000000    14990
    0.039500      282
    0.933549      234
    0.007087      216
    0.939612      206
    0.052203      183
    0.083365      169
    0.202751      168
    0.135881      159
    0.113060      150
    Name: CAD_DEMOGRAFICO_VAR_33, dtype: int64
    31542
    CAD_DEMOGRAFICO_VAR_34
    0.000000    14990
    0.996157      282
    0.958168      234
    0.149179      216
    0.965408      206
    0.001959      183
    0.036582      169
    0.012262      168
    0.003623      159
    0.995214      151
    Name: CAD_DEMOGRAFICO_VAR_34, dtype: int64
    29037
    CAD_DEMOGRAFICO_VAR_35
    0.000000    14990
    0.957224      282
    0.992920      234
    0.170196      216
    0.112137      206
    0.378482      183
    0.047683      169
    0.283356      168
    0.381318      159
    0.056517      150
    Name: CAD_DEMOGRAFICO_VAR_35, dtype: int64
    31620
    CAD_DEMOGRAFICO_VAR_36
    0.000000    14990
    0.990461      283
    0.882631      234
    0.086666      216
    0.840922      206
    0.089409      183
    0.363218      169
    0.030563      168
    0.149357      159
    0.007676      150
    Name: CAD_DEMOGRAFICO_VAR_36, dtype: int64
    30845
    CAD_DEMOGRAFICO_VAR_37
    0.000000    14990
    0.007406      283
    0.489579      234
    0.977308      216
    0.985987      206
    0.643099      183
    0.963047      169
    0.920343      168
    0.974953      159
    0.979374      150
    Name: CAD_DEMOGRAFICO_VAR_37, dtype: int64
    30802
    CAD_DEMOGRAFICO_VAR_38
    0.000000    14990
    0.994157      282
    0.333299      234
    0.716329      216
    0.289495      206
    0.052726      183
    0.970953      169
    0.082572      168
    0.873123      159
    0.042640      150
    Name: CAD_DEMOGRAFICO_VAR_38, dtype: int64
    31392
    CAD_DEMOGRAFICO_VAR_39
    0.000000    14990
    0.919202      282
    0.556019      234
    0.021359      216
    0.324641      206
    0.002606      183
    0.004883      169
    0.040914      168
    0.002779      165
    0.939007      159
    Name: CAD_DEMOGRAFICO_VAR_39, dtype: int64
    31630
    CAD_DEMOGRAFICO_VAR_40
    0.000000    14990
    0.989259      282
    0.181396      234
    0.981907      217
    0.220857      206
    0.995360      183
    0.941941      169
    0.985194      168
    0.962060      159
    0.981160      150
    Name: CAD_DEMOGRAFICO_VAR_40, dtype: int64
    28995
    CAD_DEMOGRAFICO_VAR_41
    0.000000    14990
    0.008433      282
    0.837825      234
    0.094442      216
    0.817987      206
    0.008620      197
    0.028057      183
    0.261219      169
    0.272123      159
    0.005525      150
    Name: CAD_DEMOGRAFICO_VAR_41, dtype: int64
    31529
    CAD_DEMOGRAFICO_VAR_42
    0.000000    14990
    0.985474      282
    0.930034      234
    0.004221      216
    0.002705      206
    0.014950      184
    0.005857      173
    0.357077      168
    0.352651      159
    0.020894      152
    Name: CAD_DEMOGRAFICO_VAR_42, dtype: int64
    31743
    CAD_DEMOGRAFICO_VAR_43
    0.000000    14990
    0.018129      282
    0.962558      234
    0.856517      216
    0.951118      206
    0.751157      183
    0.112395      169
    0.291758      168
    0.930509      159
    0.768503      150
    Name: CAD_DEMOGRAFICO_VAR_43, dtype: int64
    31250
    CAD_DEMOGRAFICO_VAR_44
    0.000000    14990
    0.970443      282
    0.956441      234
    0.496023      216
    0.848736      206
    0.963518      183
    0.910697      169
    0.662755      168
    0.994678      159
    0.670155      150
    Name: CAD_DEMOGRAFICO_VAR_44, dtype: int64
    31769
    CAD_DEMOGRAFICO_VAR_45
    0.000000    14990
    0.989486      282
    0.002111      234
    0.933879      216
    0.916289      206
    0.676835      183
    0.745547      169
    0.962982      169
    0.765626      159
    0.370513      150
    Name: CAD_DEMOGRAFICO_VAR_45, dtype: int64
    31581
    CAD_DEMOGRAFICO_VAR_46
    0.000000    16499
    0.999996     1098
    0.000001     1069
    0.999995      787
    0.999992      787
    0.999997      717
    0.000002      707
    0.999993      696
    0.999989      684
    0.999990      671
    Name: CAD_DEMOGRAFICO_VAR_46, dtype: int64
    9126
    CAD_DEMOGRAFICO_VAR_47
    0.000000    14990
    0.018602      282
    0.869920      234
    0.925734      216
    0.025413      206
    0.848635      183
    0.993679      169
    0.950776      168
    0.950815      159
    0.688638      150
    Name: CAD_DEMOGRAFICO_VAR_47, dtype: int64
    30942
    CAD_DEMOGRAFICO_VAR_50
    0.000000    14990
    0.013875      282
    0.991028      234
    0.076425      216
    0.597071      206
    0.732845      183
    0.716635      169
    0.501067      168
    0.513301      159
    0.717097      150
    Name: CAD_DEMOGRAFICO_VAR_50, dtype: int64
    31775
    CAD_DEMOGRAFICO_VAR_52
    0.000000    14990
    0.975665      282
    0.774659      234
    0.583758      216
    0.501111      206
    0.994365      183
    0.264439      169
    0.607346      168
    0.990618      159
    0.785820      150
    Name: CAD_DEMOGRAFICO_VAR_52, dtype: int64
    31654
    CAD_DEMOGRAFICO_VAR_53
    0.000000    14990
    0.980588      282
    0.609061      234
    0.113343      216
    0.193659      206
    0.201542      183
    0.528319      169
    0.772696      168
    0.743176      159
    0.177074      150
    Name: CAD_DEMOGRAFICO_VAR_53, dtype: int64
    31649
    CAD_DEMOGRAFICO_VAR_54
    0.000000    14990
    0.012885      282
    0.031941      234
    0.924132      217
    0.020873      206
    0.997501      188
    0.991132      169
    0.996789      168
    0.997489      159
    0.991301      151
    Name: CAD_DEMOGRAFICO_VAR_54, dtype: int64
    30246
    CAD_DEMOGRAFICO_VAR_55
    0.000000    14990
    0.992939      283
    0.999103      236
    0.990948      216
    0.041601      206
    0.962303      183
    0.986959      169
    0.999225      168
    0.981492      159
    0.984153      150
    Name: CAD_DEMOGRAFICO_VAR_55, dtype: int64
    31317
    CAD_DEMOGRAFICO_VAR_57
    0.000000    14990
    0.008421      283
    0.867802      234
    0.142247      216
    0.063863      206
    0.967660      183
    0.959987      169
    0.949456      168
    0.439697      159
    0.894450      150
    Name: CAD_DEMOGRAFICO_VAR_57, dtype: int64
    31426
    CAD_DEMOGRAFICO_VAR_58
    0.000000    14990
    0.005879      282
    0.002526      234
    0.006123      216
    0.380956      206
    0.007945      184
    0.002649      169
    0.009686      169
    0.040422      159
    0.002722      150
    Name: CAD_DEMOGRAFICO_VAR_58, dtype: int64
    28294
    CAD_DEMOGRAFICO_VAR_59
    0.000000    14990
    0.011486      282
    0.255540      234
    0.979131      216
    0.450253      206
    0.913440      183
    0.991781      169
    0.992934      168
    0.997089      159
    0.984619      150
    Name: CAD_DEMOGRAFICO_VAR_59, dtype: int64
    31389
    CAD_DEMOGRAFICO_VAR_61
    0.000000    14990
    0.005768      282
    0.000835      259
    0.007768      234
    0.905352      216
    0.964857      183
    0.998470      169
    0.964221      168
    0.787865      159
    0.945109      150
    Name: CAD_DEMOGRAFICO_VAR_61, dtype: int64
    30826
    CAD_DEMOGRAFICO_VAR_62
    0.000000    14990
    0.015690      283
    0.079488      234
    0.020256      216
    0.593470      206
    0.270722      183
    0.158964      169
    0.350418      169
    0.993589      159
    0.562213      150
    Name: CAD_DEMOGRAFICO_VAR_62, dtype: int64
    31751
    MENOR_DIST_ENDERECO_AEROPORTOS
    2.539246e-18    2000
    1.000000e+00    1980
    2.801763e-02     232
    5.108797e-02     216
    4.455518e-02     206
    6.193273e-02     183
    1.517788e-01     169
    1.840941e-01     168
    2.102735e-01     159
    1.442375e-01     150
    Name: MENOR_DIST_ENDERECO_AEROPORTOS, dtype: int64
    52425
    MENOR_DIST_ENDERECO_PARQUES_DIVERSAO
    1.000000e+00    1698
    2.770096e-18    1492
    1.617548e-02     232
    6.777406e-02     216
    3.754384e-02     206
    1.664236e-02     183
    2.750940e-01     169
    5.358099e-01     168
    5.035149e-02     159
    5.404773e-02     150
    Name: MENOR_DIST_ENDERECO_PARQUES_DIVERSAO, dtype: int64
    52350
    MENOR_DIST_ENDERECO_CAIXA_ELETRONICO
     1.000000e+00    2080
    -5.197942e-20    1531
     2.104202e-02     232
     1.005299e-01     216
     6.894711e-02     206
     9.658023e-02     183
     8.705225e-02     169
     3.277823e-01     168
     5.982250e-02     159
     2.207974e-01     150
    Name: MENOR_DIST_ENDERECO_CAIXA_ELETRONICO, dtype: int64
    52919
    MENOR_DIST_BANCO
    1.000000    1963
    0.004037     232
    0.106025     216
    0.078665     206
    0.125061     183
    0.129556     169
    0.372201     168
    0.068398     159
    0.220930     150
    0.014465     132
    Name: MENOR_DIST_BANCO, dtype: int64
    53264
    MENOR_DIST_ENDERECO_BARES
     1.000000e+00    1905
    -6.443398e-20    1521
     3.965335e-02     232
     6.429643e-02     216
     1.046334e-01     206
     1.319723e-01     183
     1.272101e-01     169
     3.858556e-01     168
     6.198480e-02     159
     9.904480e-02     150
    Name: MENOR_DIST_ENDERECO_BARES, dtype: int64
    52972
    MENOR_DIST_ENDERECO_ESTACAO_ONIBUS
    1.000000e+00    2362
    4.756433e-19     930
    3.269956e-03     232
    4.104545e-02     216
    4.017416e-02     206
    5.345394e-02     183
    5.854451e-02     169
    3.996122e-01     168
    6.132947e-02     159
    1.257479e-01     150
    Name: MENOR_DIST_ENDERECO_ESTACAO_ONIBUS, dtype: int64
    52754
    MENOR_DIST_CONCESSIONARIA
    1.000000    2221
    0.020744     232
    0.030935     216
    0.036628     206
    0.064224     183
    0.058108     169
    0.162539     168
    0.032951     159
    0.135526     150
    0.008788     132
    Name: MENOR_DIST_CONCESSIONARIA, dtype: int64
    53146
    MENOR_DIST_ALUGUEL_CARROS
    1.000000e+00    2273
    7.599871e-20    1435
    2.781320e-03     232
    4.249618e-02     216
    2.777880e-02     206
    4.939028e-02     183
    6.607459e-01     169
    1.269857e-01     168
    2.987231e-02     159
    2.622454e-01     150
    Name: MENOR_DIST_ALUGUEL_CARROS, dtype: int64
    52587
    MENOR_DIST_ENDERECO_OFICINAS
    1.000000    1987
    0.014348     232
    0.020555     216
    0.000153     206
    0.058305     183
    0.056720     169
    0.310845     168
    0.043373     159
    0.195154     150
    0.019281     132
    Name: MENOR_DIST_ENDERECO_OFICINAS, dtype: int64
    53260
    MENOR_DIST_ENDERECO_LAVA_RAPIDO
     1.000000e+00    2113
    -4.379933e-19    1165
     1.823984e-02     232
     3.700059e-02     216
     5.140202e-02     206
     6.619197e-02     183
     7.993909e-01     169
     1.973262e-01     168
     3.820978e-02     159
     1.362536e-01     150
    Name: MENOR_DIST_ENDERECO_LAVA_RAPIDO, dtype: int64
    52625
    MENOR_DIST_ENDERECO_CEMITERIO
     1.000000e+00    2020
    -1.371173e-18    1640
     3.017856e-02     232
     5.487558e-02     216
     5.523813e-02     206
     7.888535e-02     183
     5.301598e-02     169
     2.372356e-01     168
     3.655255e-02     159
     1.359753e-01     150
    Name: MENOR_DIST_ENDERECO_CEMITERIO, dtype: int64
    52373
    MENOR_DIST_ENDERECO_IGREJA
     1.000000e+00    1802
    -6.595748e-20    1763
     2.059771e-02     232
     6.125039e-02     216
     7.560011e-02     206
     4.109651e-02     183
     1.043120e-01     169
     3.744941e-01     168
     2.283397e-02     159
     2.039378e-01     150
    Name: MENOR_DIST_ENDERECO_IGREJA, dtype: int64
    53006
    MENOR_DIST_ENDERECO_PREFEITURA
    1.000000    1915
    0.020789     232
    0.058389     216
    0.000141     206
    0.104612     183
    0.089304     169
    0.337165     168
    0.078639     159
    0.248626     150
    0.000184     132
    Name: MENOR_DIST_ENDERECO_PREFEITURA, dtype: int64
    53324
    MENOR_DIST_ENDERECO_BOMBEIRO
     1.000000e+00    2061
    -1.708888e-18    1552
     1.335011e-02     232
     2.200075e-02     216
     2.597031e-02     206
     4.877819e-02     183
     3.260193e-02     169
     1.436401e-01     168
     3.060983e-02     159
     7.800987e-02     150
    Name: MENOR_DIST_ENDERECO_BOMBEIRO, dtype: int64
    52516
    MENOR_DIST_ENDERECO_FAVELA
     1.000000e+00    2964
    -6.974137e-21    1270
     3.539302e-02     232
     1.836316e-01     216
     2.562051e-03     206
     4.694967e-01     183
     8.060681e-01     169
     3.888943e-02     159
     3.518173e-01     150
     2.188055e-03     132
    Name: MENOR_DIST_ENDERECO_FAVELA, dtype: int64
    51940
    MENOR_DIST_ENDERECO_FUNERARIA
    1.000000e+00    2045
    2.116851e-19    1783
    2.341346e-02     232
    3.733457e-02     216
    4.224142e-02     206
    7.444900e-02     183
    7.714812e-02     169
    2.006598e-01     168
    4.316056e-02     159
    1.248303e-01     150
    Name: MENOR_DIST_ENDERECO_FUNERARIA, dtype: int64
    52491
    MENOR_DIST_ENDERECO_POSTO_GASOLINA
    1.000000    2034
    0.057674     232
    0.055989     216
    0.039122     206
    0.106645     183
    0.053665     169
    0.407753     168
    0.055373     159
    0.000043     150
    0.024657     132
    Name: MENOR_DIST_ENDERECO_POSTO_GASOLINA, dtype: int64
    53253
    MENOR_DIST_ENDERECO_SUPERMERCADO
    1.000000    1928
    0.058541     232
    0.015525     216
    0.103830     206
    0.127366     183
    0.000214     169
    0.397360     168
    0.033640     159
    0.212808     150
    0.000215     132
    Name: MENOR_DIST_ENDERECO_SUPERMERCADO, dtype: int64
    53299
    MENOR_DIST_ENDERECO_ACADEMIAS
     1.000000e+00    2088
    -4.548199e-20    1173
     2.147866e-02     232
     6.985113e-02     216
     2.607871e-02     206
     1.099460e-01     183
     8.679613e-02     169
     2.810976e-01     168
     4.403385e-02     159
     1.768822e-01     150
    Name: MENOR_DIST_ENDERECO_ACADEMIAS, dtype: int64
    53060
    MENOR_DIST_ENDERECO_HOSPITAL
     1.000000e+00    2359
    -3.140927e-20    1656
     3.232807e-03     232
     2.482019e-02     216
     3.613646e-02     206
     7.247690e-02     183
     5.346287e-02     169
     1.930042e-01     168
     4.224334e-02     159
     9.586353e-02     150
    Name: MENOR_DIST_ENDERECO_HOSPITAL, dtype: int64
    52844
    MENOR_DIST_CORRETOR_SEGUROS
     1.000000e+00    2816
    -9.457537e-20    1951
     2.808479e-03     232
     1.635467e-01     216
     1.124008e-02     206
     1.822354e-02     183
     1.164961e-01     169
     4.508174e-01     168
     9.318787e-02     159
     2.929726e-02     150
    Name: MENOR_DIST_CORRETOR_SEGUROS, dtype: int64
    52135
    MENOR_DIST_ENDERECO_BEBIDAS
     1.000000e+00    1762
    -8.305931e-20    1367
     6.295477e-03     232
     1.283328e-02     216
     1.154678e-02     206
     1.874329e-02     183
     2.268720e-01     169
     9.102872e-02     168
     1.076419e-02     159
     1.521498e-01     150
    Name: MENOR_DIST_ENDERECO_BEBIDAS, dtype: int64
    52658
    MENOR_DIST_ENDERECO_HOTEL
    1.000000    2071
    0.059226     232
    0.084463     216
    0.133270     206
    0.105030     183
    0.044143     169
    0.334819     168
    0.045229     159
    0.279692     150
    0.028603     132
    Name: MENOR_DIST_ENDERECO_HOTEL, dtype: int64
    53225
    MENOR_DIST_ENDERECO_CINEMAS
     1.000000e+00    2003
    -3.831494e-18    1853
     9.285169e-03     232
     3.364686e-02     216
     3.378340e-02     206
     3.133821e-01     183
     3.993402e-02     169
     6.313191e-01     168
     2.513264e-02     159
     7.765493e-02     150
    Name: MENOR_DIST_ENDERECO_CINEMAS, dtype: int64
    52434
    MENOR_DIST_ENDERECO_CASA_NOTURNA
    1.000000e+00    2046
    7.983483e-19    1393
    2.953183e-02     232
    2.816648e-02     216
    5.519930e-02     206
    8.993286e-02     183
    4.871137e-01     169
    2.467815e-01     168
    4.671487e-02     159
    1.567958e-01     150
    Name: MENOR_DIST_ENDERECO_CASA_NOTURNA, dtype: int64
    52606
    MENOR_DIST_ENDERECO_PARQUE
     1.000000e+00    2068
    -6.554914e-19    1233
     4.694045e-03     232
     4.064626e-02     216
     8.212750e-02     206
     1.008627e-01     183
     5.541076e-02     169
     3.177421e-01     168
     8.412781e-02     159
     2.072835e-01     150
    Name: MENOR_DIST_ENDERECO_PARQUE, dtype: int64
    52611
    MENOR_DIST_ESTACIONAMENTOS
    1.000000e+00    1993
    3.079579e-19    1732
    2.397857e-03     232
    2.881845e-02     216
    1.444275e-02     206
    4.130666e-02     183
    2.865836e-01     169
    9.983412e-02     168
    2.492172e-02     159
    8.886379e-02     150
    Name: MENOR_DIST_ESTACIONAMENTOS, dtype: int64
    52434
    MENOR_DIST_ENDERECO_POLICIA
    3.907195e-20    1694
    1.000000e+00    1377
    5.063025e-03     232
    2.539737e-01     216
    1.449352e-02     206
    5.475057e-01     183
    1.378220e-02     159
    1.733407e-01     150
    1.506759e-02     132
    3.741287e-01     115
    Name: MENOR_DIST_ENDERECO_POLICIA, dtype: int64
    52770
    MENOR_DIST_ENDERECO_CORREIOS
    1.493065e-18    1951
    1.000000e+00    1499
    1.031300e-03     232
    2.365512e-02     216
    9.893134e-03     206
    1.565873e-01     183
    3.231401e-01     169
    6.368483e-01     168
    6.950849e-03     159
    1.455266e-01     150
    Name: MENOR_DIST_ENDERECO_CORREIOS, dtype: int64
    52396
    MENOR_DIST_ENDERECO_ESCOLAS
    1.000000    1785
    0.056414     232
    0.032847     216
    0.098199     206
    0.096906     183
    0.145523     169
    0.190153     168
    0.052551     159
    0.000058     150
    0.028379     132
    Name: MENOR_DIST_ENDERECO_ESCOLAS, dtype: int64
    53325
    MENOR_DIST_ENDERECO_SHOPPING
    1.000000e+00    2127
    1.186602e-19    1805
    2.008229e-02     232
    3.498853e-02     216
    4.777771e-02     206
    8.726322e-02     183
    4.929834e-02     169
    1.991860e-01     168
    3.704925e-02     159
    1.438467e-01     150
    Name: MENOR_DIST_ENDERECO_SHOPPING, dtype: int64
    52379
    MENOR_DIST_ENDERECO_METRO
    1.000000e+00    27809
    8.473269e-19     2114
    4.446790e-03      232
    2.991055e-01      216
    2.678521e-02      206
    2.129273e-01      159
    2.562833e-01      132
    3.517178e-01      115
    2.628248e-01      109
    1.103585e-02      100
    Name: MENOR_DIST_ENDERECO_METRO, dtype: int64
    42349
    MENOR_DIST_PONTO_TAXI
    1.000000e+00    1862
    8.534396e-20    1520
    1.175790e-02     232
    1.899704e-02     216
    2.895084e-02     206
    3.435091e-02     183
    3.735749e-01     169
    7.522638e-02     168
    1.729237e-02     159
    4.046433e-01     150
    Name: MENOR_DIST_PONTO_TAXI, dtype: int64
    52344
    MENOR_DIST_ENDERECO_TREM
     1.000000e+00    9294
    -1.245274e-18    1330
     8.683823e-01     232
     2.244713e-02     216
     4.693475e-02     206
     4.717132e-01     183
     6.682426e-01     169
     8.030674e-01     168
     5.832251e-02     159
     9.339127e-01     150
    Name: MENOR_DIST_ENDERECO_TREM, dtype: int64
    49534
    MENOR_DIST_UNIVERSIDADE
     1.000000e+00    2064
    -3.924331e-19    1806
     2.025158e-02     232
     5.279241e-02     216
     1.782199e-02     206
     9.525572e-02     183
     3.943389e-02     169
     1.967114e-01     168
     2.614324e-02     159
     1.708475e-01     150
    Name: MENOR_DIST_UNIVERSIDADE, dtype: int64
    52474
    MENOR_DIST_ENDERECO_FRONTEIRA_ESTADUAL
    8.425253e-19    1386
    1.000000e+00    1378
    1.477244e-01     232
    3.687333e-01     216
    2.178323e-02     206
    4.277294e-01     183
    2.491539e-01     169
    7.180600e-02     168
    3.551824e-01     159
    6.602487e-02     150
    Name: MENOR_DIST_ENDERECO_FRONTEIRA_ESTADUAL, dtype: int64
    52439
    MENOR_DIST_ENDERECO_FRONTEIRA_MARITIMA
     1.000000e+00    35449
    -1.109648e-17     1383
     3.028137e-01      216
     7.181341e-02      206
     2.892596e-01      159
     3.214180e-01      132
     3.329858e-01      115
     1.325828e-01      114
     2.652864e-01      109
     1.361561e-01      100
    Name: MENOR_DIST_ENDERECO_FRONTEIRA_MARITIMA, dtype: int64
    38653
    MENOR_DIST_ENDERECO_FRONTEIRA_INTERNACIONAL
    1.000000e+00    102751
    1.312635e-16      1242
    3.702257e-01       183
    1.826407e-01       169
    9.079855e-02       168
    4.929721e-01        97
    4.683321e-01        95
    6.578699e-01        90
    5.286700e-02        84
    9.092993e-01        83
    Name: MENOR_DIST_ENDERECO_FRONTEIRA_INTERNACIONAL, dtype: int64
    3947
    EXPOSICAO_ENDERECO_AEROPORTOS
    0.000000    64802
    0.015385    17030
    0.030769     9992
    0.046154     4237
    0.076923     3159
    0.061538     3036
    0.107692     2418
    1.000000     2222
    0.092308     2110
    0.123077     1532
    Name: EXPOSICAO_ENDERECO_AEROPORTOS, dtype: int64
    66
    EXPOSICAO_ENDERECO_PARQUES_DIVERSAO
    0.0    62158
    0.1    17437
    0.2    10453
    0.3     8420
    0.4     5514
    0.6     3024
    0.5     2744
    1.0     2603
    0.7     2139
    0.8     1194
    Name: EXPOSICAO_ENDERECO_PARQUES_DIVERSAO, dtype: int64
    11
    EXPOSICAO_ENDERECO_AREA_RISCO
    0.000000    47103
    0.013825     5371
    0.023041     3796
    0.018433     3570
    0.027650     2606
    0.032258     2288
    0.046083     2116
    0.036866     1850
    0.050691     1610
    0.041475     1602
    Name: EXPOSICAO_ENDERECO_AREA_RISCO, dtype: int64
    218
    EXPOSICAO_ENDERECO_CAIXA_ELETRONICO
    0.000000    27318
    0.003704     4713
    0.007407     2724
    0.011111     2407
    0.014815     2028
    0.033333     1974
    0.018519     1960
    0.029630     1798
    1.000000     1733
    0.022222     1729
    Name: EXPOSICAO_ENDERECO_CAIXA_ELETRONICO, dtype: int64
    271
    EXPOSICAO_ENDERECO_BANCOS
    0.000000    24976
    0.004425     4710
    0.008850     2987
    0.013274     2138
    0.017699     1958
    0.035398     1843
    0.030973     1704
    0.022124     1698
    0.026549     1629
    1.000000     1613
    Name: EXPOSICAO_ENDERECO_BANCOS, dtype: int64
    227
    EXPOSICAO_ENDERECO_BARES
    0.000000    20927
    0.002041     5926
    0.004082     3408
    0.006122     2705
    0.008163     2058
    0.010204     2005
    0.018367     1894
    0.012245     1590
    1.000000     1575
    0.014286     1555
    Name: EXPOSICAO_ENDERECO_BARES, dtype: int64
    491
    EXPOSICAO_ENDERECO_ESTACAO_ONIBUS
    0.000000    39112
    0.000585    12896
    0.001170     6866
    0.001754     4185
    0.002339     2154
    1.000000     1532
    0.003509     1413
    0.002924     1200
    0.004094      741
    0.004678      712
    Name: EXPOSICAO_ENDERECO_ESTACAO_ONIBUS, dtype: int64
    1569
    EXPOSICAO_ENDERECO_CONCESSIONARIA
    0.000000    32075
    0.004219     4186
    0.008439     2218
    0.012658     2176
    1.000000     1517
    0.016878     1445
    0.025316     1147
    0.054852     1146
    0.113924     1120
    0.109705     1091
    Name: EXPOSICAO_ENDERECO_CONCESSIONARIA, dtype: int64
    238
    EXPOSICAO_ENDERECO_ALUGUEL_CARROS
    0.000000    41697
    0.007092     5466
    0.014184     4078
    0.021277     3647
    0.028369     3180
    0.035461     2564
    0.049645     2463
    0.042553     2152
    1.000000     1824
    0.070922     1711
    Name: EXPOSICAO_ENDERECO_ALUGUEL_CARROS, dtype: int64
    142
    EXPOSICAO_ENDERECO_OFICINAS
    0.000000    21444
    0.001730     4456
    0.003460     2448
    0.005190     2248
    0.006920     1872
    0.008651     1529
    0.010381     1300
    1.000000     1279
    0.019031     1274
    0.013841     1072
    Name: EXPOSICAO_ENDERECO_OFICINAS, dtype: int64
    579
    EXPOSICAO_ENDERECO_LAVA_RAPIDO
    0.000000    34801
    0.004762     5618
    0.009524     3347
    0.014286     2066
    0.028571     1977
    0.019048     1961
    0.023810     1821
    1.000000     1740
    0.038095     1363
    0.057143     1358
    Name: EXPOSICAO_ENDERECO_LAVA_RAPIDO, dtype: int64
    211
    EXPOSICAO_ENDERECO_CEMITERIO
    0.000000    33860
    0.030303    14835
    0.060606    12515
    0.090909     9139
    0.121212     7404
    0.151515     5448
    0.181818     4331
    0.212121     3439
    0.272727     2816
    0.242424     2706
    Name: EXPOSICAO_ENDERECO_CEMITERIO, dtype: int64
    34
    EXPOSICAO_ENDERECO_IGREJA
    0.000000    18441
    0.003367     5036
    0.006734     2817
    0.010101     2675
    0.013468     2238
    0.020202     1638
    0.016835     1567
    0.087542     1526
    0.023569     1343
    0.090909     1325
    Name: EXPOSICAO_ENDERECO_IGREJA, dtype: int64
    298
    EXPOSICAO_ENDERECO_PREFEITURA
    0.000000    25432
    0.014493     9008
    0.028986     7848
    0.043478     6379
    0.057971     5267
    0.086957     4369
    0.072464     4190
    0.115942     3656
    0.101449     3225
    0.130435     3179
    Name: EXPOSICAO_ENDERECO_PREFEITURA, dtype: int64
    70
    EXPOSICAO_ENDERECO_BOMBEIRO
    0.00    43479
    0.04    14289
    0.08    11241
    0.12     8457
    0.16     6209
    0.20     5318
    0.24     5171
    0.28     3557
    0.32     2890
    0.36     2345
    Name: EXPOSICAO_ENDERECO_BOMBEIRO, dtype: int64
    26
    EXPOSICAO_ENDERECO_FAVELAS
    0.000000    58670
    1.000000      931
    0.003369      225
    0.032330      223
    0.001720      208
    0.002867      205
    0.001792      195
    0.003297      180
    0.002652      177
    0.000932      170
    Name: EXPOSICAO_ENDERECO_FAVELAS, dtype: int64
    7109
    EXPOSICAO_ENDERECO_FUNERARIA
    0.000000    35381
    0.015873     8332
    0.031746     7063
    0.047619     5541
    0.063492     5024
    0.079365     4764
    0.095238     3992
    0.126984     3127
    0.111111     3070
    0.142857     2933
    Name: EXPOSICAO_ENDERECO_FUNERARIA, dtype: int64
    64
    EXPOSICAO_ENDERECO_POSTO_GASOLINA
    0.000000    19772
    0.004237     4921
    0.008475     3561
    0.012712     2508
    0.016949     1891
    0.021186     1712
    0.063559     1620
    0.025424     1573
    0.033898     1510
    0.029661     1470
    Name: EXPOSICAO_ENDERECO_POSTO_GASOLINA, dtype: int64
    237
    EXPOSICAO_ENDERECO_SUPERMERCADO
    0.000000    19916
    0.003155     4491
    0.006309     2959
    0.009464     1896
    0.028391     1837
    0.015773     1793
    0.012618     1651
    0.044164     1441
    0.018927     1407
    0.066246     1296
    Name: EXPOSICAO_ENDERECO_SUPERMERCADO, dtype: int64
    318
    EXPOSICAO_ENDERECO_ACADEMIAS
    0.000000    28074
    0.003571     3917
    0.007143     3265
    0.010714     2235
    0.014286     1602
    1.000000     1520
    0.017857     1342
    0.046429     1268
    0.039286     1220
    0.021429     1137
    Name: EXPOSICAO_ENDERECO_ACADEMIAS, dtype: int64
    281
    EXPOSICAO_ENDERECO_HOSPITAL
    0.000000    28345
    0.003413     5902
    0.006826     3877
    0.010239     2875
    0.013652     2181
    1.000000     1890
    0.017065     1795
    0.020478     1525
    0.040956     1433
    0.030717     1393
    Name: EXPOSICAO_ENDERECO_HOSPITAL, dtype: int64
    294
    EXPOSICAO_ENDERECO_CORRETOR_SEGUROS
    0.000000    46911
    0.004673     3507
    0.009346     2778
    0.014019     2008
    0.023364     1928
    0.060748     1723
    0.018692     1690
    0.070093     1537
    0.065421     1453
    1.000000     1443
    Name: EXPOSICAO_ENDERECO_CORRETOR_SEGUROS, dtype: int64
    215
    EXPOSICAO_ENDERECO_BEBIDAS
    0.000000    44224
    0.010638     5745
    0.021277     3705
    0.053191     3136
    0.031915     3019
    0.042553     2741
    0.085106     2272
    0.074468     2272
    0.095745     2020
    1.000000     1922
    Name: EXPOSICAO_ENDERECO_BEBIDAS, dtype: int64
    95
    EXPOSICAO_ENDERECO_HOTEL
    0.000000    21871
    0.004016     6178
    0.008032     4860
    0.012048     3875
    0.016064     3342
    0.020080     2420
    0.024096     2177
    0.028112     2085
    0.036145     1887
    0.032129     1887
    Name: EXPOSICAO_ENDERECO_HOTEL, dtype: int64
    250
    EXPOSICAO_ENDERECO_CINEMAS
    0.000000    46874
    0.008621    11532
    0.017241     8768
    0.025862     7058
    0.034483     5337
    0.043103     4215
    0.051724     3707
    0.060345     2904
    0.077586     2372
    0.068966     2188
    Name: EXPOSICAO_ENDERECO_CINEMAS, dtype: int64
    117
    EXPOSICAO_ENDERECO_CASA_NOTURNA
    0.000000    30373
    0.007042     5672
    0.014085     3717
    0.035211     2578
    0.021127     2455
    0.028169     2406
    1.000000     1925
    0.056338     1808
    0.098592     1719
    0.070423     1683
    Name: EXPOSICAO_ENDERECO_CASA_NOTURNA, dtype: int64
    143
    EXPOSICAO_ENDERECO_PARQUE
    0.000000    27386
    0.010870     6076
    0.021739     4138
    0.032609     3609
    0.076087     3505
    0.054348     3091
    0.086957     3040
    0.108696     2862
    0.065217     2835
    0.043478     2767
    Name: EXPOSICAO_ENDERECO_PARQUE, dtype: int64
    93
    EXPOSICAO_ENDERECO_ESTACIONAMENTOS
    0.000000    41417
    0.003460     5626
    0.006920     3385
    0.013841     2750
    0.010381     2700
    0.017301     2289
    0.020761     2105
    0.027682     1804
    1.000000     1610
    0.024221     1606
    Name: EXPOSICAO_ENDERECO_ESTACIONAMENTOS, dtype: int64
    290
    EXPOSICAO_ENDERECO_POLICIA
    0.000000    43102
    0.012658     4689
    0.025316     4152
    0.037975     3157
    0.050633     2939
    0.088608     2783
    0.075949     2262
    0.063291     2038
    0.101266     1983
    0.202532     1931
    Name: EXPOSICAO_ENDERECO_POLICIA, dtype: int64
    80
    EXPOSICAO_ENDERECO_CORREIOS
    0.000000    47161
    0.015385     6655
    0.030769     4702
    0.061538     3693
    0.046154     3568
    0.092308     2557
    0.153846     2444
    0.107692     2402
    0.123077     2381
    0.076923     2320
    Name: EXPOSICAO_ENDERECO_CORREIOS, dtype: int64
    66
    EXPOSICAO_ENDERECO_ESCOLAS
    0.000000    13092
    0.002899     6806
    0.005797     3398
    0.008696     2568
    0.011594     1705
    0.014493     1522
    0.017391     1415
    0.086957     1376
    1.000000     1351
    0.063768     1290
    Name: EXPOSICAO_ENDERECO_ESCOLAS, dtype: int64
    346
    EXPOSICAO_ENDERECO_SHOPPING
    0.000000    34447
    0.004310     5643
    0.008621     3203
    0.017241     3087
    0.012931     2964
    0.021552     2595
    0.030172     2182
    0.025862     2107
    0.060345     1866
    0.043103     1652
    Name: EXPOSICAO_ENDERECO_SHOPPING, dtype: int64
    233
    EXPOSICAO_ENDERECO_METRO
    0.000000    93836
    0.045455     2735
    0.318182     2149
    1.000000     2060
    0.227273     1998
    0.272727     1976
    0.181818     1818
    0.136364     1804
    0.090909     1615
    0.363636     1459
    Name: EXPOSICAO_ENDERECO_METRO, dtype: int64
    23
    EXPOSICAO_ENDERECO_PONTO_TAXI
    0.000000    51173
    0.008929    10706
    0.017857     6652
    0.035714     4036
    0.026786     4013
    0.044643     3844
    0.062500     3314
    0.053571     3209
    0.071429     2762
    0.080357     2012
    Name: EXPOSICAO_ENDERECO_PONTO_TAXI, dtype: int64
    113
    EXPOSICAO_ENDERECO_TREM
    0.000000    88216
    0.076923     8961
    0.153846     3327
    0.230769     3062
    0.307692     2615
    0.384615     1940
    0.461538     1662
    0.692308     1475
    0.538462     1436
    1.000000     1405
    Name: EXPOSICAO_ENDERECO_TREM, dtype: int64
    14
    EXPOSICAO_ENDERECO_UNIVERSIDADE
    0.000000    37464
    0.004587     4566
    0.009174     3010
    0.013761     2892
    0.022936     2618
    0.018349     2404
    0.050459     2086
    0.027523     2040
    0.055046     1902
    0.032110     1871
    Name: EXPOSICAO_ENDERECO_UNIVERSIDADE, dtype: int64
    219
    FLAG_REDE_SOCIAL_1
    1    94507
    0    22252
    Name: FLAG_REDE_SOCIAL_1, dtype: int64
    2
    FLAG_REDE_SOCIAL_2
    0    105388
    1     11371
    Name: FLAG_REDE_SOCIAL_2, dtype: int64
    2
    FLAG_REDE_SOCIAL_3
    0    105878
    1     10881
    Name: FLAG_REDE_SOCIAL_3, dtype: int64
    2
    FLAG_WEB_ARTES_1
    0    78957
    1    37802
    Name: FLAG_WEB_ARTES_1, dtype: int64
    2
    FLAG_WEB_MUSICA_1
    0    102993
    1     13766
    Name: FLAG_WEB_MUSICA_1, dtype: int64
    2
    FLAG_WEB_TV_1
    0    90768
    1    25991
    Name: FLAG_WEB_TV_1, dtype: int64
    2
    FLAG_WEB_LIVROS_1
    0    105546
    1     11213
    Name: FLAG_WEB_LIVROS_1, dtype: int64
    2
    FLAG_WEB_NEGOCIOS_1
    0    88047
    1    28712
    Name: FLAG_WEB_NEGOCIOS_1, dtype: int64
    2
    FLAG_WEB_NEGOCIOS_SERVICOS_1
    0    107946
    1      8813
    Name: FLAG_WEB_NEGOCIOS_SERVICOS_1, dtype: int64
    2
    FLAG_WEB_NEGOCIOS_MARKETING_1
    0    105804
    1     10955
    Name: FLAG_WEB_NEGOCIOS_MARKETING_1, dtype: int64
    2
    FLAG_WEB_NEGOCIOS_SERVICOS_UNIVERSIDADES_1
    0    101690
    1     15069
    Name: FLAG_WEB_NEGOCIOS_SERVICOS_UNIVERSIDADES_1, dtype: int64
    2
    FLAG_WEB_NEGOCIOS_SERVICOS_COMPUTACAO_1
    0    105116
    1     11643
    Name: FLAG_WEB_NEGOCIOS_SERVICOS_COMPUTACAO_1, dtype: int64
    2
    FLAG_WEB_SAUDE_1
    0    104150
    1     12609
    Name: FLAG_WEB_SAUDE_1, dtype: int64
    2
    FLAG_WEB_NOTICIAS_1
    0    92427
    1    24332
    Name: FLAG_WEB_NOTICIAS_1, dtype: int64
    2
    FLAG_WEB_SOCIEDADE_1
    0    98716
    1    18043
    Name: FLAG_WEB_SOCIEDADE_1, dtype: int64
    2
    FLAG_WEB_SOCIEDADE_GENEALOGIA_1
    0    103628
    1     13131
    Name: FLAG_WEB_SOCIEDADE_GENEALOGIA_1, dtype: int64
    2
    EXPOSICAO_WEB
    1.0000    23434
    0.0000    11382
    0.0003     3982
    0.0004     3950
    0.0001     3933
    0.0002     3835
    0.0005     3583
    0.0006     3434
    0.0007     3404
    0.0008     3281
    Name: EXPOSICAO_WEB, dtype: int64
    1900
    FLAG_WEB_CIENCIA_1
    0    105551
    1     11208
    Name: FLAG_WEB_CIENCIA_1, dtype: int64
    2
    FLAG_WEB_COMPRAS_1
    0    99589
    1    17170
    Name: FLAG_WEB_COMPRAS_1, dtype: int64
    2
    FLAG_WEB_ESPORTES_FUTEBOL_1
    0    106750
    1     10009
    Name: FLAG_WEB_ESPORTES_FUTEBOL_1, dtype: int64
    2
    FLAG_WEB_1
    0    109161
    1      7598
    Name: FLAG_WEB_1, dtype: int64
    2
    CEP1_1
    0    62306
    1    54453
    Name: CEP1_1, dtype: int64
    2
    CEP1_2
    0    67426
    1    49333
    Name: CEP1_2, dtype: int64
    2
    CEP1_3
    0    71300
    1    45459
    Name: CEP1_3, dtype: int64
    2
    CEP1_4
    0    73619
    1    43140
    Name: CEP1_4, dtype: int64
    2
    CEP1_5
    0    75626
    1    41133
    Name: CEP1_5, dtype: int64
    2
    CEP2_1
    1    92290
    0    24469
    Name: CEP2_1, dtype: int64
    2
    CEP2_2
    1    68773
    0    47986
    Name: CEP2_2, dtype: int64
    2
    CEP2_3
    0    62039
    1    54720
    Name: CEP2_3, dtype: int64
    2
    CEP2_4
    0    68207
    1    48552
    Name: CEP2_4, dtype: int64
    2
    CEP2_5
    0    73441
    1    43318
    Name: CEP2_5, dtype: int64
    2
    CEP2_6
    0    75089
    1    41670
    Name: CEP2_6, dtype: int64
    2
    CEP2_7
    0    76798
    1    39961
    Name: CEP2_7, dtype: int64
    2
    CEP2_8
    0    77259
    1    39500
    Name: CEP2_8, dtype: int64
    2
    CEP2_9
    0    78507
    1    38252
    Name: CEP2_9, dtype: int64
    2
    CEP3_1
    1    87393
    0    29366
    Name: CEP3_1, dtype: int64
    2
    CEP3_2
    1    70758
    0    46001
    Name: CEP3_2, dtype: int64
    2
    CEP3_3
    0    64494
    1    52265
    Name: CEP3_3, dtype: int64
    2
    CEP3_4
    0    70271
    1    46488
    Name: CEP3_4, dtype: int64
    2
    CEP3_5
    0    72258
    1    44501
    Name: CEP3_5, dtype: int64
    2
    CEP3_6
    0    77876
    1    38883
    Name: CEP3_6, dtype: int64
    2
    CEP3_7
    0    78308
    1    38451
    Name: CEP3_7, dtype: int64
    2
    CEP3_8
    0    79536
    1    37223
    Name: CEP3_8, dtype: int64
    2
    CEP3_9
    0    76266
    1    40493
    Name: CEP3_9, dtype: int64
    2
    CEP3_10
    0    77822
    1    38937
    Name: CEP3_10, dtype: int64
    2
    CEP3_11
    0    72816
    1    43943
    Name: CEP3_11, dtype: int64
    2
    CEP3_12
    0    72299
    1    44460
    Name: CEP3_12, dtype: int64
    2
    CEP4_1
    1    94075
    0    22684
    Name: CEP4_1, dtype: int64
    2
    CEP4_2
    1    80682
    0    36077
    Name: CEP4_2, dtype: int64
    2
    CEP4_3
    1    69938
    0    46821
    Name: CEP4_3, dtype: int64
    2
    CEP4_4
    1    59947
    0    56812
    Name: CEP4_4, dtype: int64
    2
    CEP4_5
    0    61252
    1    55507
    Name: CEP4_5, dtype: int64
    2
    CEP4_6
    0    65669
    1    51090
    Name: CEP4_6, dtype: int64
    2
    CEP4_7
    0    67258
    1    49501
    Name: CEP4_7, dtype: int64
    2
    CEP4_8
    0    67961
    1    48798
    Name: CEP4_8, dtype: int64
    2
    CEP4_9
    0    67262
    1    49497
    Name: CEP4_9, dtype: int64
    2
    CEP4_10
    0    63034
    1    53725
    Name: CEP4_10, dtype: int64
    2
    CEP4_11
    0    65196
    1    51563
    Name: CEP4_11, dtype: int64
    2
    CEP4_12
    0    65628
    1    51131
    Name: CEP4_12, dtype: int64
    2
    CEP4_13
    0    66314
    1    50445
    Name: CEP4_13, dtype: int64
    2
    CEP4_14
    0    65345
    1    51414
    Name: CEP4_14, dtype: int64
    2
    IND_BOM_1_1
    1    76592
    0    40167
    Name: IND_BOM_1_1, dtype: int64
    2
    IND_BOM_1_2
    0    76592
    1    40167
    Name: IND_BOM_1_2, dtype: int64
    2
    


```python
for column in category_columns:
    training_data[column] = training_data[column].astype('category')
```


```python
training_data.dtypes.value_counts()
```




    float64     144
    category     80
    int64        22
    dtype: int64




```python
confunsion_matrix = pandas.crosstab(training_data["SEXO_1"], training_data["IND_BOM_1_1"])
confunsion_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>IND_BOM_1_1</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>SEXO_1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19330</td>
      <td>36823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20837</td>
      <td>39769</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy.stats import chisquare
chisquare(confunsion_matrix)
```




    Power_divergenceResult(statistic=array([ 56.54016979, 113.31360978]), pvalue=array([5.50619646e-14, 1.84208521e-26]))



# Feature Selection


```python
features.corr().abs()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <th>...</th>
      <th>CEP4_5</th>
      <th>CEP4_6</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>UF_1</th>
      <td>1.000000</td>
      <td>0.209093</td>
      <td>0.048255</td>
      <td>0.146628</td>
      <td>0.118488</td>
      <td>0.100570</td>
      <td>0.087990</td>
      <td>0.019386</td>
      <td>0.007811</td>
      <td>0.006435</td>
      <td>...</td>
      <td>0.025403</td>
      <td>0.017901</td>
      <td>0.002279</td>
      <td>0.000400</td>
      <td>0.041249</td>
      <td>0.004074</td>
      <td>0.004246</td>
      <td>0.009868</td>
      <td>0.022239</td>
      <td>0.035315</td>
    </tr>
    <tr>
      <th>UF_2</th>
      <td>0.209093</td>
      <td>1.000000</td>
      <td>0.123264</td>
      <td>0.247423</td>
      <td>0.187327</td>
      <td>0.171868</td>
      <td>0.180898</td>
      <td>0.012217</td>
      <td>0.013103</td>
      <td>0.014506</td>
      <td>...</td>
      <td>0.040997</td>
      <td>0.006801</td>
      <td>0.002159</td>
      <td>0.035769</td>
      <td>0.002821</td>
      <td>0.023900</td>
      <td>0.012583</td>
      <td>0.021424</td>
      <td>0.007369</td>
      <td>0.029016</td>
    </tr>
    <tr>
      <th>UF_3</th>
      <td>0.048255</td>
      <td>0.123264</td>
      <td>1.000000</td>
      <td>0.275167</td>
      <td>0.269899</td>
      <td>0.246315</td>
      <td>0.216494</td>
      <td>0.031256</td>
      <td>0.003892</td>
      <td>0.013546</td>
      <td>...</td>
      <td>0.011452</td>
      <td>0.008139</td>
      <td>0.009605</td>
      <td>0.062716</td>
      <td>0.011476</td>
      <td>0.038354</td>
      <td>0.017366</td>
      <td>0.002350</td>
      <td>0.018033</td>
      <td>0.011395</td>
    </tr>
    <tr>
      <th>UF_4</th>
      <td>0.146628</td>
      <td>0.247423</td>
      <td>0.275167</td>
      <td>1.000000</td>
      <td>0.126074</td>
      <td>0.126333</td>
      <td>0.135131</td>
      <td>0.020921</td>
      <td>0.004924</td>
      <td>0.041647</td>
      <td>...</td>
      <td>0.001024</td>
      <td>0.005757</td>
      <td>0.009747</td>
      <td>0.021085</td>
      <td>0.012738</td>
      <td>0.019959</td>
      <td>0.028406</td>
      <td>0.010240</td>
      <td>0.023383</td>
      <td>0.047764</td>
    </tr>
    <tr>
      <th>UF_5</th>
      <td>0.118488</td>
      <td>0.187327</td>
      <td>0.269899</td>
      <td>0.126074</td>
      <td>1.000000</td>
      <td>0.156790</td>
      <td>0.121600</td>
      <td>0.003239</td>
      <td>0.006073</td>
      <td>0.018759</td>
      <td>...</td>
      <td>0.000297</td>
      <td>0.002524</td>
      <td>0.060533</td>
      <td>0.009569</td>
      <td>0.000885</td>
      <td>0.000713</td>
      <td>0.001331</td>
      <td>0.023599</td>
      <td>0.026219</td>
      <td>0.026264</td>
    </tr>
    <tr>
      <th>UF_6</th>
      <td>0.100570</td>
      <td>0.171868</td>
      <td>0.246315</td>
      <td>0.126333</td>
      <td>0.156790</td>
      <td>1.000000</td>
      <td>0.137971</td>
      <td>0.026368</td>
      <td>0.017433</td>
      <td>0.028672</td>
      <td>...</td>
      <td>0.002480</td>
      <td>0.023013</td>
      <td>0.067999</td>
      <td>0.022139</td>
      <td>0.029033</td>
      <td>0.065947</td>
      <td>0.025372</td>
      <td>0.000670</td>
      <td>0.015740</td>
      <td>0.003706</td>
    </tr>
    <tr>
      <th>UF_7</th>
      <td>0.087990</td>
      <td>0.180898</td>
      <td>0.216494</td>
      <td>0.135131</td>
      <td>0.121600</td>
      <td>0.137971</td>
      <td>1.000000</td>
      <td>0.002974</td>
      <td>0.022839</td>
      <td>0.000691</td>
      <td>...</td>
      <td>0.044504</td>
      <td>0.024435</td>
      <td>0.005311</td>
      <td>0.000662</td>
      <td>0.006608</td>
      <td>0.035075</td>
      <td>0.001038</td>
      <td>0.017101</td>
      <td>0.034493</td>
      <td>0.003431</td>
    </tr>
    <tr>
      <th>IDADE</th>
      <td>0.019386</td>
      <td>0.012217</td>
      <td>0.031256</td>
      <td>0.020921</td>
      <td>0.003239</td>
      <td>0.026368</td>
      <td>0.002974</td>
      <td>1.000000</td>
      <td>0.017405</td>
      <td>0.029988</td>
      <td>...</td>
      <td>0.001120</td>
      <td>0.004402</td>
      <td>0.013037</td>
      <td>0.001559</td>
      <td>0.008037</td>
      <td>0.004025</td>
      <td>0.000544</td>
      <td>0.002066</td>
      <td>0.000616</td>
      <td>0.006353</td>
    </tr>
    <tr>
      <th>SEXO_1</th>
      <td>0.007811</td>
      <td>0.013103</td>
      <td>0.003892</td>
      <td>0.004924</td>
      <td>0.006073</td>
      <td>0.017433</td>
      <td>0.022839</td>
      <td>0.017405</td>
      <td>1.000000</td>
      <td>0.020646</td>
      <td>...</td>
      <td>0.000789</td>
      <td>0.008388</td>
      <td>0.000917</td>
      <td>0.002623</td>
      <td>0.002129</td>
      <td>0.002683</td>
      <td>0.001580</td>
      <td>0.004715</td>
      <td>0.002303</td>
      <td>0.003675</td>
    </tr>
    <tr>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <td>0.006435</td>
      <td>0.014506</td>
      <td>0.013546</td>
      <td>0.041647</td>
      <td>0.018759</td>
      <td>0.028672</td>
      <td>0.000691</td>
      <td>0.029988</td>
      <td>0.020646</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.002201</td>
      <td>0.004603</td>
      <td>0.001246</td>
      <td>0.002234</td>
      <td>0.003535</td>
      <td>0.001846</td>
      <td>0.000003</td>
      <td>0.000372</td>
      <td>0.004002</td>
      <td>0.005085</td>
    </tr>
    <tr>
      <th>NIVEL_RELACIONAMENTO_CREDITO02</th>
      <td>0.001615</td>
      <td>0.000890</td>
      <td>0.010338</td>
      <td>0.009182</td>
      <td>0.007102</td>
      <td>0.002931</td>
      <td>0.000145</td>
      <td>0.009357</td>
      <td>0.004807</td>
      <td>0.239705</td>
      <td>...</td>
      <td>0.001662</td>
      <td>0.000234</td>
      <td>0.000432</td>
      <td>0.002970</td>
      <td>0.000242</td>
      <td>0.003913</td>
      <td>0.002860</td>
      <td>0.001918</td>
      <td>0.000591</td>
      <td>0.001580</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_1</th>
      <td>0.007550</td>
      <td>0.048365</td>
      <td>0.042261</td>
      <td>0.036186</td>
      <td>0.006520</td>
      <td>0.036946</td>
      <td>0.028865</td>
      <td>0.011831</td>
      <td>0.041594</td>
      <td>0.014328</td>
      <td>...</td>
      <td>0.012245</td>
      <td>0.016455</td>
      <td>0.005782</td>
      <td>0.004235</td>
      <td>0.002374</td>
      <td>0.004159</td>
      <td>0.004780</td>
      <td>0.007634</td>
      <td>0.000321</td>
      <td>0.011306</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_2</th>
      <td>0.027318</td>
      <td>0.013636</td>
      <td>0.009286</td>
      <td>0.000790</td>
      <td>0.015587</td>
      <td>0.012387</td>
      <td>0.002948</td>
      <td>0.035525</td>
      <td>0.010267</td>
      <td>0.012884</td>
      <td>...</td>
      <td>0.004858</td>
      <td>0.009597</td>
      <td>0.004823</td>
      <td>0.009435</td>
      <td>0.003250</td>
      <td>0.007759</td>
      <td>0.002070</td>
      <td>0.006667</td>
      <td>0.002163</td>
      <td>0.012610</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_3</th>
      <td>0.031309</td>
      <td>0.049316</td>
      <td>0.007395</td>
      <td>0.035204</td>
      <td>0.002591</td>
      <td>0.009251</td>
      <td>0.039252</td>
      <td>0.002203</td>
      <td>0.022476</td>
      <td>0.003451</td>
      <td>...</td>
      <td>0.011136</td>
      <td>0.002867</td>
      <td>0.002796</td>
      <td>0.000919</td>
      <td>0.006268</td>
      <td>0.004781</td>
      <td>0.009614</td>
      <td>0.000145</td>
      <td>0.004880</td>
      <td>0.004504</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_4</th>
      <td>0.009614</td>
      <td>0.011060</td>
      <td>0.029721</td>
      <td>0.024070</td>
      <td>0.004330</td>
      <td>0.012961</td>
      <td>0.021763</td>
      <td>0.010099</td>
      <td>0.031053</td>
      <td>0.008848</td>
      <td>...</td>
      <td>0.006003</td>
      <td>0.005124</td>
      <td>0.008856</td>
      <td>0.002532</td>
      <td>0.001209</td>
      <td>0.002564</td>
      <td>0.003083</td>
      <td>0.001932</td>
      <td>0.001854</td>
      <td>0.000901</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_5</th>
      <td>0.000016</td>
      <td>0.003451</td>
      <td>0.007674</td>
      <td>0.002916</td>
      <td>0.009894</td>
      <td>0.020938</td>
      <td>0.000802</td>
      <td>0.013540</td>
      <td>0.019192</td>
      <td>0.009553</td>
      <td>...</td>
      <td>0.003891</td>
      <td>0.008531</td>
      <td>0.007255</td>
      <td>0.003513</td>
      <td>0.001576</td>
      <td>0.003156</td>
      <td>0.000101</td>
      <td>0.004460</td>
      <td>0.000758</td>
      <td>0.006276</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_6</th>
      <td>0.004719</td>
      <td>0.036600</td>
      <td>0.024210</td>
      <td>0.031489</td>
      <td>0.008058</td>
      <td>0.008847</td>
      <td>0.023075</td>
      <td>0.017595</td>
      <td>0.041111</td>
      <td>0.015834</td>
      <td>...</td>
      <td>0.012357</td>
      <td>0.015106</td>
      <td>0.006938</td>
      <td>0.006781</td>
      <td>0.000470</td>
      <td>0.004426</td>
      <td>0.005728</td>
      <td>0.006814</td>
      <td>0.000359</td>
      <td>0.012221</td>
    </tr>
    <tr>
      <th>BANCO_REST_IRPF_ULTIMA_7</th>
      <td>0.004345</td>
      <td>0.048489</td>
      <td>0.038199</td>
      <td>0.031314</td>
      <td>0.012052</td>
      <td>0.021653</td>
      <td>0.037037</td>
      <td>0.017687</td>
      <td>0.040374</td>
      <td>0.016688</td>
      <td>...</td>
      <td>0.012903</td>
      <td>0.014959</td>
      <td>0.007677</td>
      <td>0.006785</td>
      <td>0.001073</td>
      <td>0.002700</td>
      <td>0.003872</td>
      <td>0.006262</td>
      <td>0.001432</td>
      <td>0.011833</td>
    </tr>
    <tr>
      <th>ATIVIDADE_EMAIL</th>
      <td>0.029827</td>
      <td>0.102135</td>
      <td>0.053917</td>
      <td>0.079640</td>
      <td>0.008905</td>
      <td>0.067921</td>
      <td>0.058710</td>
      <td>0.022313</td>
      <td>0.010091</td>
      <td>0.015398</td>
      <td>...</td>
      <td>0.009134</td>
      <td>0.017565</td>
      <td>0.001078</td>
      <td>0.000889</td>
      <td>0.003669</td>
      <td>0.001380</td>
      <td>0.002623</td>
      <td>0.008246</td>
      <td>0.001737</td>
      <td>0.015123</td>
    </tr>
    <tr>
      <th>EXPOSICAO_ENDERECO</th>
      <td>0.050144</td>
      <td>0.068285</td>
      <td>0.010087</td>
      <td>0.071787</td>
      <td>0.007488</td>
      <td>0.025830</td>
      <td>0.014593</td>
      <td>0.037784</td>
      <td>0.029683</td>
      <td>0.043574</td>
      <td>...</td>
      <td>0.000716</td>
      <td>0.000250</td>
      <td>0.002238</td>
      <td>0.001343</td>
      <td>0.000541</td>
      <td>0.005199</td>
      <td>0.002688</td>
      <td>0.000534</td>
      <td>0.006409</td>
      <td>0.019879</td>
    </tr>
    <tr>
      <th>EXPOSICAO_EMAIL</th>
      <td>0.023400</td>
      <td>0.076136</td>
      <td>0.038393</td>
      <td>0.065260</td>
      <td>0.000574</td>
      <td>0.041351</td>
      <td>0.038661</td>
      <td>0.036607</td>
      <td>0.010047</td>
      <td>0.013042</td>
      <td>...</td>
      <td>0.008211</td>
      <td>0.016051</td>
      <td>0.001216</td>
      <td>0.002955</td>
      <td>0.000915</td>
      <td>0.000205</td>
      <td>0.001108</td>
      <td>0.006326</td>
      <td>0.002026</td>
      <td>0.007416</td>
    </tr>
    <tr>
      <th>EXPOSICAO_TELEFONE</th>
      <td>0.036865</td>
      <td>0.089696</td>
      <td>0.040969</td>
      <td>0.067006</td>
      <td>0.002890</td>
      <td>0.053896</td>
      <td>0.049747</td>
      <td>0.012228</td>
      <td>0.008596</td>
      <td>0.029972</td>
      <td>...</td>
      <td>0.008441</td>
      <td>0.012044</td>
      <td>0.003002</td>
      <td>0.002395</td>
      <td>0.002630</td>
      <td>0.000210</td>
      <td>0.003113</td>
      <td>0.006901</td>
      <td>0.001755</td>
      <td>0.008774</td>
    </tr>
    <tr>
      <th>ATIVIDADE_ENDERECO</th>
      <td>0.022894</td>
      <td>0.053707</td>
      <td>0.036434</td>
      <td>0.033969</td>
      <td>0.043457</td>
      <td>0.020355</td>
      <td>0.021810</td>
      <td>0.053765</td>
      <td>0.004899</td>
      <td>0.010995</td>
      <td>...</td>
      <td>0.005599</td>
      <td>0.001461</td>
      <td>0.004068</td>
      <td>0.001806</td>
      <td>0.006177</td>
      <td>0.005720</td>
      <td>0.003607</td>
      <td>0.004260</td>
      <td>0.004610</td>
      <td>0.017025</td>
    </tr>
    <tr>
      <th>ATUALIZACAO_ENDERECO</th>
      <td>0.079764</td>
      <td>0.133604</td>
      <td>0.085711</td>
      <td>0.128637</td>
      <td>0.028621</td>
      <td>0.081709</td>
      <td>0.063768</td>
      <td>0.028367</td>
      <td>0.009754</td>
      <td>0.055744</td>
      <td>...</td>
      <td>0.003096</td>
      <td>0.004754</td>
      <td>0.000765</td>
      <td>0.005503</td>
      <td>0.004270</td>
      <td>0.003634</td>
      <td>0.011011</td>
      <td>0.006841</td>
      <td>0.005685</td>
      <td>0.021115</td>
    </tr>
    <tr>
      <th>ATUALIZACAO_EMAIL</th>
      <td>0.022480</td>
      <td>0.064821</td>
      <td>0.029610</td>
      <td>0.043340</td>
      <td>0.007848</td>
      <td>0.048778</td>
      <td>0.039065</td>
      <td>0.007181</td>
      <td>0.026544</td>
      <td>0.007267</td>
      <td>...</td>
      <td>0.003380</td>
      <td>0.009938</td>
      <td>0.002884</td>
      <td>0.000042</td>
      <td>0.001869</td>
      <td>0.001017</td>
      <td>0.000632</td>
      <td>0.005740</td>
      <td>0.000358</td>
      <td>0.011939</td>
    </tr>
    <tr>
      <th>EXPOSICAO_CONSUMIDOR_EMAILS</th>
      <td>0.028191</td>
      <td>0.100637</td>
      <td>0.066333</td>
      <td>0.092479</td>
      <td>0.003326</td>
      <td>0.056396</td>
      <td>0.055250</td>
      <td>0.029127</td>
      <td>0.002422</td>
      <td>0.026866</td>
      <td>...</td>
      <td>0.011044</td>
      <td>0.017121</td>
      <td>0.000029</td>
      <td>0.002683</td>
      <td>0.002318</td>
      <td>0.001352</td>
      <td>0.004446</td>
      <td>0.007219</td>
      <td>0.001482</td>
      <td>0.014085</td>
    </tr>
    <tr>
      <th>EXPOSICAO_CONSUMIDOR_TELEFONES</th>
      <td>0.043818</td>
      <td>0.123445</td>
      <td>0.046967</td>
      <td>0.100710</td>
      <td>0.005675</td>
      <td>0.038440</td>
      <td>0.076835</td>
      <td>0.015251</td>
      <td>0.009026</td>
      <td>0.042013</td>
      <td>...</td>
      <td>0.016380</td>
      <td>0.012466</td>
      <td>0.002132</td>
      <td>0.005159</td>
      <td>0.000541</td>
      <td>0.001936</td>
      <td>0.007509</td>
      <td>0.007721</td>
      <td>0.002992</td>
      <td>0.015405</td>
    </tr>
    <tr>
      <th>ATIVIDADE_TELEFONE</th>
      <td>0.048842</td>
      <td>0.123535</td>
      <td>0.054476</td>
      <td>0.097383</td>
      <td>0.004529</td>
      <td>0.060096</td>
      <td>0.082791</td>
      <td>0.013744</td>
      <td>0.000816</td>
      <td>0.029384</td>
      <td>...</td>
      <td>0.012557</td>
      <td>0.015946</td>
      <td>0.000594</td>
      <td>0.002983</td>
      <td>0.002939</td>
      <td>0.001914</td>
      <td>0.004522</td>
      <td>0.012180</td>
      <td>0.000529</td>
      <td>0.018880</td>
    </tr>
    <tr>
      <th>VALOR_PARCELA_BOLSA_FAMILIA</th>
      <td>0.029549</td>
      <td>0.078861</td>
      <td>0.030546</td>
      <td>0.075914</td>
      <td>0.007226</td>
      <td>0.034562</td>
      <td>0.033717</td>
      <td>0.019411</td>
      <td>0.224650</td>
      <td>0.015068</td>
      <td>...</td>
      <td>0.006317</td>
      <td>0.000778</td>
      <td>0.014631</td>
      <td>0.009022</td>
      <td>0.007882</td>
      <td>0.008043</td>
      <td>0.008097</td>
      <td>0.000593</td>
      <td>0.008237</td>
      <td>0.016280</td>
    </tr>
    <tr>
      <th>FLAG_BOLSA_FAMILIA_1</th>
      <td>0.017125</td>
      <td>0.076686</td>
      <td>0.048595</td>
      <td>0.073702</td>
      <td>0.011201</td>
      <td>0.057516</td>
      <td>0.029261</td>
      <td>0.056231</td>
      <td>0.246700</td>
      <td>0.018757</td>
      <td>...</td>
      <td>0.006456</td>
      <td>0.000662</td>
      <td>0.018833</td>
      <td>0.008540</td>
      <td>0.011924</td>
      <td>0.008404</td>
      <td>0.004892</td>
      <td>0.000170</td>
      <td>0.007389</td>
      <td>0.018539</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>CEP2_6</th>
      <td>0.010724</td>
      <td>0.049254</td>
      <td>0.227824</td>
      <td>0.151508</td>
      <td>0.067630</td>
      <td>0.118541</td>
      <td>0.018354</td>
      <td>0.017671</td>
      <td>0.002026</td>
      <td>0.009349</td>
      <td>...</td>
      <td>0.028036</td>
      <td>0.012740</td>
      <td>0.003134</td>
      <td>0.033257</td>
      <td>0.041206</td>
      <td>0.002075</td>
      <td>0.022095</td>
      <td>0.000501</td>
      <td>0.033753</td>
      <td>0.008207</td>
    </tr>
    <tr>
      <th>CEP2_7</th>
      <td>0.035253</td>
      <td>0.036345</td>
      <td>0.092329</td>
      <td>0.042431</td>
      <td>0.019542</td>
      <td>0.107576</td>
      <td>0.132370</td>
      <td>0.002385</td>
      <td>0.015988</td>
      <td>0.013299</td>
      <td>...</td>
      <td>0.003456</td>
      <td>0.021336</td>
      <td>0.014143</td>
      <td>0.027258</td>
      <td>0.002792</td>
      <td>0.003459</td>
      <td>0.005254</td>
      <td>0.025808</td>
      <td>0.002482</td>
      <td>0.016561</td>
    </tr>
    <tr>
      <th>CEP2_8</th>
      <td>0.127231</td>
      <td>0.064413</td>
      <td>0.153344</td>
      <td>0.004123</td>
      <td>0.231533</td>
      <td>0.110188</td>
      <td>0.080085</td>
      <td>0.011028</td>
      <td>0.013940</td>
      <td>0.000648</td>
      <td>...</td>
      <td>0.026070</td>
      <td>0.000659</td>
      <td>0.023503</td>
      <td>0.024130</td>
      <td>0.000183</td>
      <td>0.015350</td>
      <td>0.001205</td>
      <td>0.018287</td>
      <td>0.027417</td>
      <td>0.024925</td>
    </tr>
    <tr>
      <th>CEP2_9</th>
      <td>0.035706</td>
      <td>0.160233</td>
      <td>0.040816</td>
      <td>0.157641</td>
      <td>0.044606</td>
      <td>0.021549</td>
      <td>0.002541</td>
      <td>0.006815</td>
      <td>0.002322</td>
      <td>0.007211</td>
      <td>...</td>
      <td>0.049948</td>
      <td>0.003169</td>
      <td>0.013246</td>
      <td>0.003809</td>
      <td>0.026293</td>
      <td>0.039538</td>
      <td>0.031095</td>
      <td>0.007514</td>
      <td>0.006430</td>
      <td>0.058079</td>
    </tr>
    <tr>
      <th>CEP3_1</th>
      <td>0.079478</td>
      <td>0.013082</td>
      <td>0.024672</td>
      <td>0.115169</td>
      <td>0.019630</td>
      <td>0.076478</td>
      <td>0.047759</td>
      <td>0.030490</td>
      <td>0.003950</td>
      <td>0.005457</td>
      <td>...</td>
      <td>0.011758</td>
      <td>0.026038</td>
      <td>0.048094</td>
      <td>0.033452</td>
      <td>0.058999</td>
      <td>0.035789</td>
      <td>0.047201</td>
      <td>0.033384</td>
      <td>0.017597</td>
      <td>0.019042</td>
    </tr>
    <tr>
      <th>CEP3_2</th>
      <td>0.095808</td>
      <td>0.012289</td>
      <td>0.032843</td>
      <td>0.019285</td>
      <td>0.013063</td>
      <td>0.086146</td>
      <td>0.050879</td>
      <td>0.022153</td>
      <td>0.006760</td>
      <td>0.010664</td>
      <td>...</td>
      <td>0.027264</td>
      <td>0.054886</td>
      <td>0.053113</td>
      <td>0.054921</td>
      <td>0.079345</td>
      <td>0.059479</td>
      <td>0.034065</td>
      <td>0.006475</td>
      <td>0.003271</td>
      <td>0.059274</td>
    </tr>
    <tr>
      <th>CEP3_3</th>
      <td>0.005244</td>
      <td>0.044583</td>
      <td>0.021466</td>
      <td>0.047603</td>
      <td>0.044638</td>
      <td>0.005848</td>
      <td>0.033995</td>
      <td>0.009803</td>
      <td>0.009796</td>
      <td>0.004269</td>
      <td>...</td>
      <td>0.011289</td>
      <td>0.012343</td>
      <td>0.054169</td>
      <td>0.021250</td>
      <td>0.004511</td>
      <td>0.018212</td>
      <td>0.011057</td>
      <td>0.061374</td>
      <td>0.019429</td>
      <td>0.040716</td>
    </tr>
    <tr>
      <th>CEP3_4</th>
      <td>0.036935</td>
      <td>0.068332</td>
      <td>0.002233</td>
      <td>0.006970</td>
      <td>0.018711</td>
      <td>0.067506</td>
      <td>0.046090</td>
      <td>0.000177</td>
      <td>0.004429</td>
      <td>0.005953</td>
      <td>...</td>
      <td>0.015391</td>
      <td>0.025864</td>
      <td>0.056434</td>
      <td>0.021252</td>
      <td>0.014778</td>
      <td>0.034971</td>
      <td>0.021035</td>
      <td>0.015093</td>
      <td>0.002690</td>
      <td>0.003043</td>
    </tr>
    <tr>
      <th>CEP3_5</th>
      <td>0.018686</td>
      <td>0.007016</td>
      <td>0.044004</td>
      <td>0.021937</td>
      <td>0.035649</td>
      <td>0.034387</td>
      <td>0.055902</td>
      <td>0.020135</td>
      <td>0.005227</td>
      <td>0.000593</td>
      <td>...</td>
      <td>0.026954</td>
      <td>0.014938</td>
      <td>0.021426</td>
      <td>0.007138</td>
      <td>0.051028</td>
      <td>0.003520</td>
      <td>0.016496</td>
      <td>0.019804</td>
      <td>0.003867</td>
      <td>0.011449</td>
    </tr>
    <tr>
      <th>CEP3_6</th>
      <td>0.030020</td>
      <td>0.033136</td>
      <td>0.100501</td>
      <td>0.026748</td>
      <td>0.014460</td>
      <td>0.010951</td>
      <td>0.084968</td>
      <td>0.008376</td>
      <td>0.004291</td>
      <td>0.000195</td>
      <td>...</td>
      <td>0.013941</td>
      <td>0.033553</td>
      <td>0.008870</td>
      <td>0.013424</td>
      <td>0.008919</td>
      <td>0.068129</td>
      <td>0.010120</td>
      <td>0.009546</td>
      <td>0.002526</td>
      <td>0.058430</td>
    </tr>
    <tr>
      <th>CEP3_7</th>
      <td>0.005571</td>
      <td>0.070140</td>
      <td>0.066772</td>
      <td>0.044413</td>
      <td>0.111133</td>
      <td>0.078501</td>
      <td>0.082090</td>
      <td>0.006493</td>
      <td>0.004277</td>
      <td>0.004795</td>
      <td>...</td>
      <td>0.037716</td>
      <td>0.032734</td>
      <td>0.042090</td>
      <td>0.028335</td>
      <td>0.022631</td>
      <td>0.009884</td>
      <td>0.006030</td>
      <td>0.043659</td>
      <td>0.022275</td>
      <td>0.006181</td>
    </tr>
    <tr>
      <th>CEP3_8</th>
      <td>0.007747</td>
      <td>0.038914</td>
      <td>0.014766</td>
      <td>0.155868</td>
      <td>0.078621</td>
      <td>0.040540</td>
      <td>0.019838</td>
      <td>0.002797</td>
      <td>0.009405</td>
      <td>0.009370</td>
      <td>...</td>
      <td>0.013239</td>
      <td>0.020391</td>
      <td>0.001042</td>
      <td>0.021430</td>
      <td>0.030134</td>
      <td>0.009896</td>
      <td>0.036293</td>
      <td>0.005137</td>
      <td>0.000335</td>
      <td>0.002855</td>
    </tr>
    <tr>
      <th>CEP3_9</th>
      <td>0.007316</td>
      <td>0.051206</td>
      <td>0.018560</td>
      <td>0.050332</td>
      <td>0.065487</td>
      <td>0.088553</td>
      <td>0.028149</td>
      <td>0.007804</td>
      <td>0.002677</td>
      <td>0.000952</td>
      <td>...</td>
      <td>0.003629</td>
      <td>0.058565</td>
      <td>0.002754</td>
      <td>0.048102</td>
      <td>0.039036</td>
      <td>0.006741</td>
      <td>0.009801</td>
      <td>0.007158</td>
      <td>0.023076</td>
      <td>0.001276</td>
    </tr>
    <tr>
      <th>CEP3_10</th>
      <td>0.037401</td>
      <td>0.122573</td>
      <td>0.032248</td>
      <td>0.009712</td>
      <td>0.040417</td>
      <td>0.054517</td>
      <td>0.020581</td>
      <td>0.007211</td>
      <td>0.003127</td>
      <td>0.003563</td>
      <td>...</td>
      <td>0.012316</td>
      <td>0.011996</td>
      <td>0.001078</td>
      <td>0.021077</td>
      <td>0.042262</td>
      <td>0.038610</td>
      <td>0.005878</td>
      <td>0.034739</td>
      <td>0.034160</td>
      <td>0.015346</td>
    </tr>
    <tr>
      <th>CEP3_11</th>
      <td>0.038761</td>
      <td>0.003692</td>
      <td>0.032516</td>
      <td>0.032584</td>
      <td>0.059035</td>
      <td>0.029763</td>
      <td>0.002990</td>
      <td>0.019295</td>
      <td>0.004793</td>
      <td>0.004151</td>
      <td>...</td>
      <td>0.001041</td>
      <td>0.024088</td>
      <td>0.016885</td>
      <td>0.016830</td>
      <td>0.016259</td>
      <td>0.008468</td>
      <td>0.004203</td>
      <td>0.011100</td>
      <td>0.007912</td>
      <td>0.012464</td>
    </tr>
    <tr>
      <th>CEP3_12</th>
      <td>0.009115</td>
      <td>0.089565</td>
      <td>0.044717</td>
      <td>0.027196</td>
      <td>0.014113</td>
      <td>0.003386</td>
      <td>0.009442</td>
      <td>0.001182</td>
      <td>0.005584</td>
      <td>0.000917</td>
      <td>...</td>
      <td>0.005185</td>
      <td>0.008472</td>
      <td>0.038325</td>
      <td>0.030986</td>
      <td>0.021511</td>
      <td>0.011453</td>
      <td>0.014967</td>
      <td>0.014722</td>
      <td>0.004071</td>
      <td>0.009819</td>
    </tr>
    <tr>
      <th>CEP4_1</th>
      <td>0.003057</td>
      <td>0.023137</td>
      <td>0.010847</td>
      <td>0.042641</td>
      <td>0.007986</td>
      <td>0.053411</td>
      <td>0.013797</td>
      <td>0.026147</td>
      <td>0.002971</td>
      <td>0.012223</td>
      <td>...</td>
      <td>0.050156</td>
      <td>0.066598</td>
      <td>0.062154</td>
      <td>0.069148</td>
      <td>0.057546</td>
      <td>0.034671</td>
      <td>0.079745</td>
      <td>0.080468</td>
      <td>0.042412</td>
      <td>0.056045</td>
    </tr>
    <tr>
      <th>CEP4_2</th>
      <td>0.041778</td>
      <td>0.003489</td>
      <td>0.045959</td>
      <td>0.016462</td>
      <td>0.072975</td>
      <td>0.115646</td>
      <td>0.026872</td>
      <td>0.014857</td>
      <td>0.000759</td>
      <td>0.009663</td>
      <td>...</td>
      <td>0.077276</td>
      <td>0.079951</td>
      <td>0.108763</td>
      <td>0.099927</td>
      <td>0.064027</td>
      <td>0.066742</td>
      <td>0.060307</td>
      <td>0.064826</td>
      <td>0.068966</td>
      <td>0.079220</td>
    </tr>
    <tr>
      <th>CEP4_3</th>
      <td>0.023338</td>
      <td>0.015772</td>
      <td>0.025754</td>
      <td>0.002694</td>
      <td>0.050612</td>
      <td>0.092219</td>
      <td>0.005744</td>
      <td>0.009344</td>
      <td>0.008977</td>
      <td>0.000904</td>
      <td>...</td>
      <td>0.055127</td>
      <td>0.100593</td>
      <td>0.098300</td>
      <td>0.068514</td>
      <td>0.091356</td>
      <td>0.126188</td>
      <td>0.057642</td>
      <td>0.055201</td>
      <td>0.096777</td>
      <td>0.092043</td>
    </tr>
    <tr>
      <th>CEP4_4</th>
      <td>0.012945</td>
      <td>0.051797</td>
      <td>0.016761</td>
      <td>0.004748</td>
      <td>0.018837</td>
      <td>0.014398</td>
      <td>0.018447</td>
      <td>0.003289</td>
      <td>0.005784</td>
      <td>0.001465</td>
      <td>...</td>
      <td>0.110578</td>
      <td>0.082239</td>
      <td>0.059399</td>
      <td>0.102702</td>
      <td>0.086201</td>
      <td>0.092063</td>
      <td>0.089606</td>
      <td>0.091386</td>
      <td>0.093666</td>
      <td>0.076223</td>
    </tr>
    <tr>
      <th>CEP4_5</th>
      <td>0.025403</td>
      <td>0.040997</td>
      <td>0.011452</td>
      <td>0.001024</td>
      <td>0.000297</td>
      <td>0.002480</td>
      <td>0.044504</td>
      <td>0.001120</td>
      <td>0.000789</td>
      <td>0.002201</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.081452</td>
      <td>0.084146</td>
      <td>0.068273</td>
      <td>0.074884</td>
      <td>0.114886</td>
      <td>0.076358</td>
      <td>0.081288</td>
      <td>0.077429</td>
      <td>0.070549</td>
    </tr>
    <tr>
      <th>CEP4_6</th>
      <td>0.017901</td>
      <td>0.006801</td>
      <td>0.008139</td>
      <td>0.005757</td>
      <td>0.002524</td>
      <td>0.023013</td>
      <td>0.024435</td>
      <td>0.004402</td>
      <td>0.008388</td>
      <td>0.004603</td>
      <td>...</td>
      <td>0.081452</td>
      <td>1.000000</td>
      <td>0.061243</td>
      <td>0.055286</td>
      <td>0.057445</td>
      <td>0.067454</td>
      <td>0.093325</td>
      <td>0.109380</td>
      <td>0.059392</td>
      <td>0.105031</td>
    </tr>
    <tr>
      <th>CEP4_7</th>
      <td>0.002279</td>
      <td>0.002159</td>
      <td>0.009605</td>
      <td>0.009747</td>
      <td>0.060533</td>
      <td>0.067999</td>
      <td>0.005311</td>
      <td>0.013037</td>
      <td>0.000917</td>
      <td>0.001246</td>
      <td>...</td>
      <td>0.084146</td>
      <td>0.061243</td>
      <td>1.000000</td>
      <td>0.076438</td>
      <td>0.065009</td>
      <td>0.079427</td>
      <td>0.053802</td>
      <td>0.095205</td>
      <td>0.093120</td>
      <td>0.079717</td>
    </tr>
    <tr>
      <th>CEP4_8</th>
      <td>0.000400</td>
      <td>0.035769</td>
      <td>0.062716</td>
      <td>0.021085</td>
      <td>0.009569</td>
      <td>0.022139</td>
      <td>0.000662</td>
      <td>0.001559</td>
      <td>0.002623</td>
      <td>0.002234</td>
      <td>...</td>
      <td>0.068273</td>
      <td>0.055286</td>
      <td>0.076438</td>
      <td>1.000000</td>
      <td>0.070476</td>
      <td>0.081482</td>
      <td>0.094420</td>
      <td>0.037541</td>
      <td>0.100988</td>
      <td>0.089647</td>
    </tr>
    <tr>
      <th>CEP4_9</th>
      <td>0.041249</td>
      <td>0.002821</td>
      <td>0.011476</td>
      <td>0.012738</td>
      <td>0.000885</td>
      <td>0.029033</td>
      <td>0.006608</td>
      <td>0.008037</td>
      <td>0.002129</td>
      <td>0.003535</td>
      <td>...</td>
      <td>0.074884</td>
      <td>0.057445</td>
      <td>0.065009</td>
      <td>0.070476</td>
      <td>1.000000</td>
      <td>0.056205</td>
      <td>0.107209</td>
      <td>0.104018</td>
      <td>0.096174</td>
      <td>0.081960</td>
    </tr>
    <tr>
      <th>CEP4_10</th>
      <td>0.004074</td>
      <td>0.023900</td>
      <td>0.038354</td>
      <td>0.019959</td>
      <td>0.000713</td>
      <td>0.065947</td>
      <td>0.035075</td>
      <td>0.004025</td>
      <td>0.002683</td>
      <td>0.001846</td>
      <td>...</td>
      <td>0.114886</td>
      <td>0.067454</td>
      <td>0.079427</td>
      <td>0.081482</td>
      <td>0.056205</td>
      <td>1.000000</td>
      <td>0.077964</td>
      <td>0.093005</td>
      <td>0.059894</td>
      <td>0.067307</td>
    </tr>
    <tr>
      <th>CEP4_11</th>
      <td>0.004246</td>
      <td>0.012583</td>
      <td>0.017366</td>
      <td>0.028406</td>
      <td>0.001331</td>
      <td>0.025372</td>
      <td>0.001038</td>
      <td>0.000544</td>
      <td>0.001580</td>
      <td>0.000003</td>
      <td>...</td>
      <td>0.076358</td>
      <td>0.093325</td>
      <td>0.053802</td>
      <td>0.094420</td>
      <td>0.107209</td>
      <td>0.077964</td>
      <td>1.000000</td>
      <td>0.065892</td>
      <td>0.077590</td>
      <td>0.087774</td>
    </tr>
    <tr>
      <th>CEP4_12</th>
      <td>0.009868</td>
      <td>0.021424</td>
      <td>0.002350</td>
      <td>0.010240</td>
      <td>0.023599</td>
      <td>0.000670</td>
      <td>0.017101</td>
      <td>0.002066</td>
      <td>0.004715</td>
      <td>0.000372</td>
      <td>...</td>
      <td>0.081288</td>
      <td>0.109380</td>
      <td>0.095205</td>
      <td>0.037541</td>
      <td>0.104018</td>
      <td>0.093005</td>
      <td>0.065892</td>
      <td>1.000000</td>
      <td>0.078021</td>
      <td>0.064582</td>
    </tr>
    <tr>
      <th>CEP4_13</th>
      <td>0.022239</td>
      <td>0.007369</td>
      <td>0.018033</td>
      <td>0.023383</td>
      <td>0.026219</td>
      <td>0.015740</td>
      <td>0.034493</td>
      <td>0.000616</td>
      <td>0.002303</td>
      <td>0.004002</td>
      <td>...</td>
      <td>0.077429</td>
      <td>0.059392</td>
      <td>0.093120</td>
      <td>0.100988</td>
      <td>0.096174</td>
      <td>0.059894</td>
      <td>0.077590</td>
      <td>0.078021</td>
      <td>1.000000</td>
      <td>0.068267</td>
    </tr>
    <tr>
      <th>CEP4_14</th>
      <td>0.035315</td>
      <td>0.029016</td>
      <td>0.011395</td>
      <td>0.047764</td>
      <td>0.026264</td>
      <td>0.003706</td>
      <td>0.003431</td>
      <td>0.006353</td>
      <td>0.003675</td>
      <td>0.005085</td>
      <td>...</td>
      <td>0.070549</td>
      <td>0.105031</td>
      <td>0.079717</td>
      <td>0.089647</td>
      <td>0.081960</td>
      <td>0.067307</td>
      <td>0.087774</td>
      <td>0.064582</td>
      <td>0.068267</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>243 rows × 243 columns</p>
</div>




```python
training_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>...</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
      <th>IND_BOM_1_1</th>
      <th>IND_BOM_1_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33220</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217846</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>164123</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.750400</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>340086</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.074953</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>237182</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.355855</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>335250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.930834</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>149584</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.678045</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>71560</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.485231</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>118664</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.654419</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19053</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.358808</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>368546</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.132485</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 246 columns</p>
</div>




```python
corr_matrix = features.corr().abs()
```


```python
values = corr_matrix[corr_matrix > 0.95].count()
```


```python
values[values > 1]
```




    FLAG_BOLSA_FAMILIA_1        2
    RENDA_VIZINHANCA_1          2
    RENDA_VIZINHANCA_4          2
    FLAG_PROGRAMAS_SOCIAIS_1    2
    dtype: int64




```python
corr_matrix[corr_matrix["RENDA_VIZINHANCA_1"] > 0.90]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <th>...</th>
      <th>CEP4_5</th>
      <th>CEP4_6</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RENDA_VIZINHANCA_1</th>
      <td>0.039497</td>
      <td>0.126106</td>
      <td>0.095179</td>
      <td>0.120255</td>
      <td>0.040225</td>
      <td>0.021913</td>
      <td>0.094908</td>
      <td>0.056833</td>
      <td>0.011343</td>
      <td>0.004249</td>
      <td>...</td>
      <td>0.014022</td>
      <td>0.030272</td>
      <td>0.002249</td>
      <td>0.002028</td>
      <td>0.003034</td>
      <td>0.005432</td>
      <td>0.003514</td>
      <td>0.008236</td>
      <td>0.007406</td>
      <td>0.022686</td>
    </tr>
    <tr>
      <th>RENDA_VIZINHANCA_4</th>
      <td>0.042773</td>
      <td>0.123006</td>
      <td>0.096118</td>
      <td>0.121031</td>
      <td>0.041934</td>
      <td>0.020677</td>
      <td>0.093601</td>
      <td>0.053994</td>
      <td>0.009825</td>
      <td>0.003914</td>
      <td>...</td>
      <td>0.014699</td>
      <td>0.027470</td>
      <td>0.003278</td>
      <td>0.000880</td>
      <td>0.003880</td>
      <td>0.005072</td>
      <td>0.002067</td>
      <td>0.006980</td>
      <td>0.006323</td>
      <td>0.020939</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 243 columns</p>
</div>



RENDA_VIZINHANCA_1 e  RENDA_VIZINHANCA_4 possuem alta correlação e FLAG_BOLSA_FAMILIA_1 e FLAG_PROGRAMAS_SOCIAIS_1
também. Logo, vou ficar com somente duas das 4.
Contudo, este teste é para apenas para variáveis correlacionadas linearmente, existem testes melhores para as variáveis categóricas.


```python
features = features.drop(["RENDA_VIZINHANCA_1", "FLAG_BOLSA_FAMILIA_1"], axis="columns")
```


```python
features.shape
```




    (116759, 241)




```python
features.drop_duplicates(inplace=True)
features.shape
```




    (116732, 241)




```python
training_data_model = pandas.concat([features, labels], axis=1)
```


```python
training_data_model.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <th>...</th>
      <th>CEP4_6</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
      <th>IND_BOM_1_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.217846</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.750400</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.074953</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.355855</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.930834</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.678045</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.485231</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.654419</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.358808</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.132485</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 242 columns</p>
</div>




```python
labels.value_counts()
```




    1    76592
    0    40167
    Name: IND_BOM_1_1, dtype: int64



# Feature Engineering

In this module we're trying to build the feature that we see will be more useful in order to learn about
the class we need to predict.

# Model Tranining


```python
features = training_data.drop(["IND_BOM_1_1", "IND_BOM_1_2", "id"], axis=1)
```


```python
features.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UF_1</th>
      <th>UF_2</th>
      <th>UF_3</th>
      <th>UF_4</th>
      <th>UF_5</th>
      <th>UF_6</th>
      <th>UF_7</th>
      <th>IDADE</th>
      <th>SEXO_1</th>
      <th>NIVEL_RELACIONAMENTO_CREDITO01</th>
      <th>...</th>
      <th>CEP4_5</th>
      <th>CEP4_6</th>
      <th>CEP4_7</th>
      <th>CEP4_8</th>
      <th>CEP4_9</th>
      <th>CEP4_10</th>
      <th>CEP4_11</th>
      <th>CEP4_12</th>
      <th>CEP4_13</th>
      <th>CEP4_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217846</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.750400</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.074953</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.355855</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.930834</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.678045</td>
      <td>0</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.485231</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.654419</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.358808</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.132485</td>
      <td>1</td>
      <td>0.111111</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 243 columns</p>
</div>




```python
labels = training_data["IND_BOM_1_1"]
```


```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
```


```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1/4, 
                                                    random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)
```


```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```


```python
input_dimension = X_train.shape[1]
input_dimension
```




    243




```python
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, mode='auto')
```


```python
classifier_1 = Sequential()
classifier_1.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_1.add(Dense(16, activation='relu', input_dim=input_dimension))
classifier_1.add(Dense(1, activation='sigmoid'))

classifier_1.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
```


```python
classifier_2 = Sequential()
classifier_2.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_2.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_2.add(Dense(8, activation='relu', input_dim=input_dimension/2))
classifier_2.add(Dense(1, activation='sigmoid'))

classifier_2.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
```


```python
classifier_3 = Sequential()
classifier_3.add(Dense(16, activation='relu', input_dim=input_dimension))
classifier_3.add(Dense(8, activation='relu', input_dim=input_dimension))
classifier_3.add(Dense(1, activation='sigmoid'))

classifier_3.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
```

I use as_matrix because Keras expects a Numpy array instead of a dataframe.


```python
classifier_1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 16)                3904      
    _________________________________________________________________
    dense_2 (Dense)              (None, 16)                272       
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 4,193
    Trainable params: 4,193
    Non-trainable params: 0
    _________________________________________________________________
    


```python
classifier_2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 16)                3904      
    _________________________________________________________________
    dense_5 (Dense)              (None, 16)                272       
    _________________________________________________________________
    dense_6 (Dense)              (None, 8)                 136       
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 9         
    =================================================================
    Total params: 4,321
    Trainable params: 4,321
    Non-trainable params: 0
    _________________________________________________________________
    


```python
classifier_3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_8 (Dense)              (None, 16)                3904      
    _________________________________________________________________
    dense_9 (Dense)              (None, 8)                 136       
    _________________________________________________________________
    dense_10 (Dense)             (None, 1)                 9         
    =================================================================
    Total params: 4,049
    Trainable params: 4,049
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model = classifier_1.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    

    Train on 49622 samples, validate on 8757 samples
    Epoch 1/500
    49622/49622 [==============================] - 6s 124us/step - loss: 0.2145 - acc: 0.6632 - val_loss: 0.2072 - val_acc: 0.6735
    Epoch 2/500
    49622/49622 [==============================] - 7s 134us/step - loss: 0.2083 - acc: 0.6707 - val_loss: 0.2057 - val_acc: 0.6763
    Epoch 3/500
    49622/49622 [==============================] - 6s 124us/step - loss: 0.2066 - acc: 0.6737 - val_loss: 0.2047 - val_acc: 0.6788
    Epoch 4/500
    49622/49622 [==============================] - 5s 110us/step - loss: 0.2050 - acc: 0.6768 - val_loss: 0.2047 - val_acc: 0.6752
    Epoch 5/500
    49622/49622 [==============================] - 5s 105us/step - loss: 0.2039 - acc: 0.6798 - val_loss: 0.2041 - val_acc: 0.6780
    Epoch 6/500
    49622/49622 [==============================] - 8s 160us/step - loss: 0.2030 - acc: 0.6832 - val_loss: 0.2039 - val_acc: 0.6753
    Epoch 7/500
    49622/49622 [==============================] - 7s 144us/step - loss: 0.2021 - acc: 0.6834 - val_loss: 0.2041 - val_acc: 0.6788
    Epoch 8/500
    49622/49622 [==============================] - 9s 188us/step - loss: 0.2016 - acc: 0.6842 - val_loss: 0.2038 - val_acc: 0.6779
    Epoch 9/500
    49622/49622 [==============================] - 8s 156us/step - loss: 0.2010 - acc: 0.6860 - val_loss: 0.2041 - val_acc: 0.6782
    Epoch 10/500
    49622/49622 [==============================] - 6s 127us/step - loss: 0.1999 - acc: 0.6872 - val_loss: 0.2037 - val_acc: 0.6766
    Epoch 11/500
    49622/49622 [==============================] - 6s 124us/step - loss: 0.1994 - acc: 0.6888 - val_loss: 0.2052 - val_acc: 0.6800
    Epoch 12/500
    49622/49622 [==============================] - 7s 143us/step - loss: 0.1988 - acc: 0.6892 - val_loss: 0.2045 - val_acc: 0.6749
    Epoch 13/500
    49622/49622 [==============================] - 6s 120us/step - loss: 0.1981 - acc: 0.6911 - val_loss: 0.2051 - val_acc: 0.6732
    Epoch 14/500
    49622/49622 [==============================] - 7s 150us/step - loss: 0.1976 - acc: 0.6929 - val_loss: 0.2063 - val_acc: 0.6815
    Epoch 15/500
    49622/49622 [==============================] - 7s 148us/step - loss: 0.1965 - acc: 0.6957 - val_loss: 0.2054 - val_acc: 0.6741
    Epoch 16/500
    49622/49622 [==============================] - 9s 188us/step - loss: 0.1962 - acc: 0.6963 - val_loss: 0.2067 - val_acc: 0.6726
    Epoch 17/500
    49622/49622 [==============================] - 7s 140us/step - loss: 0.1956 - acc: 0.6960 - val_loss: 0.2058 - val_acc: 0.6765
    Epoch 18/500
    49622/49622 [==============================] - 6s 126us/step - loss: 0.1949 - acc: 0.6996 - val_loss: 0.2062 - val_acc: 0.6736
    Epoch 19/500
    49622/49622 [==============================] - 6s 126us/step - loss: 0.1944 - acc: 0.7000 - val_loss: 0.2083 - val_acc: 0.6690
    Epoch 20/500
    49622/49622 [==============================] - 7s 140us/step - loss: 0.1941 - acc: 0.7030 - val_loss: 0.2086 - val_acc: 0.6646
    Epoch 21/500
    49622/49622 [==============================] - 9s 187us/step - loss: 0.1933 - acc: 0.7033 - val_loss: 0.2081 - val_acc: 0.6662
    Epoch 22/500
    49622/49622 [==============================] - 7s 139us/step - loss: 0.1927 - acc: 0.7048 - val_loss: 0.2083 - val_acc: 0.6648
    Epoch 23/500
    49622/49622 [==============================] - 7s 141us/step - loss: 0.1926 - acc: 0.7058 - val_loss: 0.2094 - val_acc: 0.6663
    Epoch 24/500
    49622/49622 [==============================] - 8s 168us/step - loss: 0.1919 - acc: 0.7067 - val_loss: 0.2087 - val_acc: 0.6679
    Epoch 25/500
    49622/49622 [==============================] - 7s 147us/step - loss: 0.1914 - acc: 0.7085 - val_loss: 0.2105 - val_acc: 0.6610
    Epoch 26/500
    49622/49622 [==============================] - 6s 128us/step - loss: 0.1908 - acc: 0.7109 - val_loss: 0.2115 - val_acc: 0.6724
    Epoch 27/500
    49622/49622 [==============================] - 7s 135us/step - loss: 0.1906 - acc: 0.7087 - val_loss: 0.2111 - val_acc: 0.6598
    Epoch 28/500
    49622/49622 [==============================] - 7s 136us/step - loss: 0.1903 - acc: 0.7120 - val_loss: 0.2104 - val_acc: 0.6676
    Epoch 29/500
    49622/49622 [==============================] - 9s 187us/step - loss: 0.1898 - acc: 0.7120 - val_loss: 0.2111 - val_acc: 0.6713
    Epoch 30/500
    49622/49622 [==============================] - 6s 119us/step - loss: 0.1893 - acc: 0.7103 - val_loss: 0.2105 - val_acc: 0.6674
    


```python
model = classifier_3.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    

    Train on 49622 samples, validate on 8757 samples
    Epoch 1/500
    49622/49622 [==============================] - 8s 164us/step - loss: 0.2154 - acc: 0.6602 - val_loss: 0.2080 - val_acc: 0.6726
    Epoch 2/500
    49622/49622 [==============================] - 7s 136us/step - loss: 0.2080 - acc: 0.6707 - val_loss: 0.2056 - val_acc: 0.6783
    Epoch 3/500
    49622/49622 [==============================] - 8s 167us/step - loss: 0.2059 - acc: 0.6748 - val_loss: 0.2071 - val_acc: 0.6758
    Epoch 4/500
    49622/49622 [==============================] - 6s 122us/step - loss: 0.2049 - acc: 0.6765 - val_loss: 0.2038 - val_acc: 0.6763
    Epoch 5/500
    49622/49622 [==============================] - 7s 139us/step - loss: 0.2039 - acc: 0.6809 - val_loss: 0.2052 - val_acc: 0.6750
    Epoch 6/500
    49622/49622 [==============================] - 7s 132us/step - loss: 0.2030 - acc: 0.6818 - val_loss: 0.2060 - val_acc: 0.6684
    Epoch 7/500
    49622/49622 [==============================] - 6s 130us/step - loss: 0.2026 - acc: 0.6838 - val_loss: 0.2033 - val_acc: 0.6775
    Epoch 8/500
    49622/49622 [==============================] - 6s 124us/step - loss: 0.2019 - acc: 0.6840 - val_loss: 0.2051 - val_acc: 0.6725
    Epoch 9/500
    49622/49622 [==============================] - 6s 115us/step - loss: 0.2015 - acc: 0.6844 - val_loss: 0.2047 - val_acc: 0.6784
    Epoch 10/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.2010 - acc: 0.6859 - val_loss: 0.2064 - val_acc: 0.6696
    Epoch 11/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.2005 - acc: 0.6884 - val_loss: 0.2052 - val_acc: 0.6800
    Epoch 12/500
    49622/49622 [==============================] - 6s 113us/step - loss: 0.2001 - acc: 0.6882 - val_loss: 0.2040 - val_acc: 0.6775
    Epoch 13/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.1995 - acc: 0.6899 - val_loss: 0.2064 - val_acc: 0.6699
    Epoch 14/500
    49622/49622 [==============================] - 6s 120us/step - loss: 0.1993 - acc: 0.6901 - val_loss: 0.2064 - val_acc: 0.6685
    Epoch 15/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.1987 - acc: 0.6906 - val_loss: 0.2065 - val_acc: 0.6779
    Epoch 16/500
    49622/49622 [==============================] - 6s 111us/step - loss: 0.1981 - acc: 0.6927 - val_loss: 0.2055 - val_acc: 0.6789
    Epoch 17/500
    49622/49622 [==============================] - 6s 118us/step - loss: 0.1976 - acc: 0.6945 - val_loss: 0.2083 - val_acc: 0.6655
    Epoch 18/500
    49622/49622 [==============================] - 6s 117us/step - loss: 0.1974 - acc: 0.6952 - val_loss: 0.2075 - val_acc: 0.6660
    Epoch 19/500
    49622/49622 [==============================] - 6s 117us/step - loss: 0.1970 - acc: 0.6960 - val_loss: 0.2061 - val_acc: 0.6775
    Epoch 20/500
    49622/49622 [==============================] - 6s 113us/step - loss: 0.1966 - acc: 0.6972 - val_loss: 0.2068 - val_acc: 0.6796
    Epoch 21/500
    49622/49622 [==============================] - 6s 115us/step - loss: 0.1964 - acc: 0.6976 - val_loss: 0.2069 - val_acc: 0.6675
    Epoch 22/500
    49622/49622 [==============================] - 6s 113us/step - loss: 0.1958 - acc: 0.6978 - val_loss: 0.2067 - val_acc: 0.6765
    Epoch 23/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.1958 - acc: 0.6972 - val_loss: 0.2087 - val_acc: 0.6639
    Epoch 24/500
    49622/49622 [==============================] - 6s 118us/step - loss: 0.1954 - acc: 0.7001 - val_loss: 0.2087 - val_acc: 0.6771
    Epoch 25/500
    49622/49622 [==============================] - 6s 113us/step - loss: 0.1951 - acc: 0.7002 - val_loss: 0.2088 - val_acc: 0.6701
    Epoch 26/500
    49622/49622 [==============================] - 6s 112us/step - loss: 0.1948 - acc: 0.7010 - val_loss: 0.2078 - val_acc: 0.6757
    Epoch 27/500
    49622/49622 [==============================] - 6s 111us/step - loss: 0.1946 - acc: 0.7021 - val_loss: 0.2085 - val_acc: 0.6694
    


```python
model = classifier_2.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    

    Train on 49622 samples, validate on 8757 samples
    Epoch 1/500
    49622/49622 [==============================] - 6s 125us/step - loss: 0.2142 - acc: 0.6627 - val_loss: 0.2063 - val_acc: 0.6773
    Epoch 2/500
    49622/49622 [==============================] - 6s 114us/step - loss: 0.2075 - acc: 0.6710 - val_loss: 0.2053 - val_acc: 0.6817
    Epoch 3/500
    49622/49622 [==============================] - 6s 118us/step - loss: 0.2057 - acc: 0.6737 - val_loss: 0.2077 - val_acc: 0.6815
    Epoch 4/500
    49622/49622 [==============================] - 6s 127us/step - loss: 0.2042 - acc: 0.6785 - val_loss: 0.2043 - val_acc: 0.6824
    Epoch 5/500
    49622/49622 [==============================] - 8s 156us/step - loss: 0.2034 - acc: 0.6784 - val_loss: 0.2042 - val_acc: 0.6845
    Epoch 6/500
    49622/49622 [==============================] - 6s 125us/step - loss: 0.2024 - acc: 0.6815 - val_loss: 0.2042 - val_acc: 0.6759
    Epoch 7/500
    49622/49622 [==============================] - 6s 119us/step - loss: 0.2014 - acc: 0.6844 - val_loss: 0.2036 - val_acc: 0.6838
    Epoch 8/500
    49622/49622 [==============================] - 6s 116us/step - loss: 0.2008 - acc: 0.6842 - val_loss: 0.2058 - val_acc: 0.6871
    Epoch 9/500
    49622/49622 [==============================] - 6s 119us/step - loss: 0.1999 - acc: 0.6870 - val_loss: 0.2035 - val_acc: 0.6831
    Epoch 10/500
    49622/49622 [==============================] - 5s 107us/step - loss: 0.1994 - acc: 0.6883 - val_loss: 0.2045 - val_acc: 0.6788
    Epoch 11/500
    49622/49622 [==============================] - 5s 100us/step - loss: 0.1984 - acc: 0.6902 - val_loss: 0.2061 - val_acc: 0.6680
    Epoch 12/500
    49622/49622 [==============================] - 5s 108us/step - loss: 0.1978 - acc: 0.6911 - val_loss: 0.2059 - val_acc: 0.6732
    Epoch 13/500
    49622/49622 [==============================] - 7s 142us/step - loss: 0.1971 - acc: 0.6943 - val_loss: 0.2052 - val_acc: 0.6825
    Epoch 14/500
    49622/49622 [==============================] - 9s 187us/step - loss: 0.1969 - acc: 0.6926 - val_loss: 0.2060 - val_acc: 0.6815
    Epoch 15/500
    49622/49622 [==============================] - 7s 138us/step - loss: 0.1961 - acc: 0.6958 - val_loss: 0.2055 - val_acc: 0.6733
    Epoch 16/500
    49622/49622 [==============================] - 6s 120us/step - loss: 0.1952 - acc: 0.6981 - val_loss: 0.2063 - val_acc: 0.6723
    Epoch 17/500
    49622/49622 [==============================] - 6s 119us/step - loss: 0.1947 - acc: 0.6984 - val_loss: 0.2064 - val_acc: 0.6731
    Epoch 18/500
    49622/49622 [==============================] - 7s 133us/step - loss: 0.1940 - acc: 0.7009 - val_loss: 0.2081 - val_acc: 0.6623
    Epoch 19/500
    49622/49622 [==============================] - 6s 118us/step - loss: 0.1933 - acc: 0.7013 - val_loss: 0.2074 - val_acc: 0.6753
    Epoch 20/500
    49622/49622 [==============================] - 6s 114us/step - loss: 0.1929 - acc: 0.7040 - val_loss: 0.2071 - val_acc: 0.6708
    Epoch 21/500
    49622/49622 [==============================] - 6s 115us/step - loss: 0.1925 - acc: 0.7044 - val_loss: 0.2101 - val_acc: 0.6622
    Epoch 22/500
    49622/49622 [==============================] - 6s 114us/step - loss: 0.1921 - acc: 0.7049 - val_loss: 0.2088 - val_acc: 0.6692
    Epoch 23/500
    49622/49622 [==============================] - 6s 118us/step - loss: 0.1916 - acc: 0.7051 - val_loss: 0.2093 - val_acc: 0.6739
    Epoch 24/500
    49622/49622 [==============================] - 6s 120us/step - loss: 0.1910 - acc: 0.7074 - val_loss: 0.2108 - val_acc: 0.6686
    Epoch 25/500
    49622/49622 [==============================] - 7s 147us/step - loss: 0.1906 - acc: 0.7093 - val_loss: 0.2097 - val_acc: 0.6751
    Epoch 26/500
    49622/49622 [==============================] - 7s 146us/step - loss: 0.1901 - acc: 0.7100 - val_loss: 0.2089 - val_acc: 0.6721
    Epoch 27/500
    49622/49622 [==============================] - 7s 146us/step - loss: 0.1894 - acc: 0.7115 - val_loss: 0.2102 - val_acc: 0.6693
    Epoch 28/500
    49622/49622 [==============================] - 9s 175us/step - loss: 0.1891 - acc: 0.7128 - val_loss: 0.2112 - val_acc: 0.6646
    Epoch 29/500
    49622/49622 [==============================] - 12s 246us/step - loss: 0.1889 - acc: 0.7139 - val_loss: 0.2118 - val_acc: 0.6634
    


```python
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.merge import concatenate
```


```python
inputs = Input(shape=(243,))

x1 = Dense(16, activation="tanh")(inputs)
x1 = Dense(16, activation="relu")(x1)

x2 = Dense(16, activation="tanh")(inputs)
x2 = Dense(8, activation="relu")(x2)

x3 = Dense(16, activation="tanh")(inputs)
x3 = Dense(16, activation="tanh")(x3)
x3 = Dense(8, activation="relu")(x3)

x4 = concatenate([x1,x2,x3])

prediction = Dense(1, activation="sigmoid")(x4)

voting_classifier = Model(inputs=inputs, outputs= prediction)
voting_classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
```


```python
early_stopping_voting = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto')
```


```python
voting_classifier.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping_voting], validation_split=0.15)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    

    Train on 49622 samples, validate on 8757 samples
    Epoch 1/500
    49622/49622 [==============================] - 6s 127us/step - loss: 0.1666 - acc: 0.7577 - val_loss: 0.2292 - val_acc: 0.6459
    Epoch 2/500
    49622/49622 [==============================] - 7s 136us/step - loss: 0.1654 - acc: 0.7612 - val_loss: 0.2301 - val_acc: 0.6517
    Epoch 3/500
    49622/49622 [==============================] - 7s 140us/step - loss: 0.1639 - acc: 0.7650 - val_loss: 0.2323 - val_acc: 0.6433
    Epoch 4/500
    49622/49622 [==============================] - 6s 126us/step - loss: 0.1629 - acc: 0.7669 - val_loss: 0.2308 - val_acc: 0.6599
    Epoch 5/500
    49622/49622 [==============================] - 7s 131us/step - loss: 0.1617 - acc: 0.7688 - val_loss: 0.2331 - val_acc: 0.6493
    Epoch 6/500
    49622/49622 [==============================] - 7s 136us/step - loss: 0.1604 - acc: 0.7738 - val_loss: 0.2350 - val_acc: 0.6433
    Epoch 7/500
    49622/49622 [==============================] - 7s 135us/step - loss: 0.1592 - acc: 0.7733 - val_loss: 0.2365 - val_acc: 0.6426
    Epoch 8/500
    49622/49622 [==============================] - 7s 133us/step - loss: 0.1583 - acc: 0.7753 - val_loss: 0.2370 - val_acc: 0.6437
    Epoch 9/500
    49622/49622 [==============================] - 6s 129us/step - loss: 0.1569 - acc: 0.7787 - val_loss: 0.2341 - val_acc: 0.6558
    Epoch 10/500
    49622/49622 [==============================] - 7s 132us/step - loss: 0.1563 - acc: 0.7801 - val_loss: 0.2376 - val_acc: 0.6571
    Epoch 11/500
    49622/49622 [==============================] - 7s 133us/step - loss: 0.1547 - acc: 0.7836 - val_loss: 0.2374 - val_acc: 0.6534
    




    <keras.callbacks.History at 0x2a449f3c9e8>




```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
```


```python
voting_classifier = VotingClassifier([('classifier_1', classifier_1), ('classifier_2', classifier_2), ('classifier_3',classifier_3)], voting='soft')
```


```python
voting_classifier.fit(X_train, y_train)
```

# Random Forest


```python
rf_clf = RandomForestClassifier()  # Modifique aqui os hyperparâmetros
rf_clf.fit(X_train, y_train)

```


```python
rf_pred_class = rf_clf.predict(X_val)
rf_pred_scores = rf_clf.predict_proba(X_val)[:, 1]
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, rf_pred_class, rf_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```


```python
rf_pred_test_class = rf_clf.predict(X_test)
rf_pred_test_scores = rf_clf.predict_proba(X_test)[:, 1]
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, rf_pred_test_class, rf_pred_test_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

# XGBoost

## Parameter Selection


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
```


```python
# We will use grid search to improve our search for parameters using Cross Validation
number_estimators = [30, 60, 120, 200]
loss_function = ["deviance", "exponential"]
min_samples_leaf = [1, 0.05]
sub_samples = [1.0, 0.8, 0.6]
max_features = ["log2", "sqrt", "auto"]
xgboost_classifier = GridSearchCV(estimator=GradientBoostingClassifier(), 
                          param_grid=dict(
                              n_estimators=number_estimators,
                              max_features=max_features,
                              subsample=sub_samples,
                          min_samples_leaf=min_samples_leaf,
                          loss=loss_function), n_jobs=-1)
```


```python
xgboost_classifier.fit(X_train, y_train)
```


```python
# as I said in the documentation, GridSeach uses a stratified 3-fold cross validation because a Classifier was passed
# instead of a regressor

classifier.best_params_
```


```python
# For in ensemble classifiers
classifier = xgboost_classifier
```

# Ensemble classifiers (Voting)

## Random Forest and XGBoost together


```python
classifier = VotingClassifier([('xgboost', xgboost_classifier), ('randomforest', rf_clf)], voting='soft')
```


```python
classifier.fit(X_train, y_train)
```

# Evaluation


```python
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
```


```python
y_train_pred = classifier.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier.predict(X_test.as_matrix()).ravel()
```

é bom prestar atenção se os valores estão próximo, caso contrário, existe uma boa indicação de que houve
overfitting e o modelo não consegue generalizar tão bem.


```python
print("Mean Square error in train: {:0.1f}".format(mse(y_train, y_train_pred)))
print("Mean Square error in test: {:0.1f}".format(mse(y_test, y_test_pred)))
```


```python
def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics
```


```python
def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
```


```python
# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier.predict_proba(X_test.as_matrix())[:, 0]
```


```python
y_test_pred.shape
```


```python
y_test.shape
```


```python
y_test_pred_prob.shape
```


```python
print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

# Evaluate for Kaggle


```python
kaggle_test_data = pandas.read_csv("real_test_set.csv")
```


```python
kaggle_test_data.shape
features_kaggle = kaggle_test_data.drop(["id"], axis=1)
features_kaggle.head(10)
```


```python
rf_pred_test_class = rf_clf.predict(features_kaggle)
rf_pred_test_scores = rf_clf.predict_proba(features_kaggle)[:, 1]
```


```python
rf_pred_test_class.size
```


```python
rf_pred_test_class
```

Se ligar que na hora que cria o csv, na primeira linha (a linha do header), ele coloca ",0", tem que substituir para "id,IND_BOM_1_1"


```python
df = pandas.DataFrame(data=rf_pred_test_class)
df.to_csv('test.csv', mode='a', index=True)
```


```python
# For in ensemble classifiers
classifier = xgboost_classifier
```

# RESULTS LOG

## XGBoost - 120 estimators


```python

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```


```python

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

## XGBoost - 2nd configuration


```python
# parameters
number_estimators = [30, 60, 120, 200]
loss_function = ["deviance", "exponential"]
min_samples_leaf = [1, 0.05]
sub_samples = [1.0, 0.8, 0.6]
max_features = ["log2", "sqrt", "auto"]
```


```python
classifier.best_params_
```


```python

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

# Ensemble Of XBGoost and RandomForest

# MLPs and Ensemble MLPS


```python
y_train_pred = classifier_1.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_1.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_1.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    

    
    Performance no conjunto de teste:
    
    Accuracy:         0.6664
    Recall:           0.8560
    Precision:        0.7013
    F1:               0.7710
    AUROC:            0.6650
    AUPR:             0.7853
    


```python
y_train_pred = classifier_2.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_2.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_2.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    

    
    Performance no conjunto de teste:
    
    Accuracy:         0.6660
    Recall:           0.8400
    Precision:        0.7064
    F1:               0.7674
    AUROC:            0.6698
    AUPR:             0.7878
    


```python
y_train_pred = classifier_3.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_3.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_3.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    

    
    Performance no conjunto de teste:
    
    Accuracy:         0.6684
    Recall:           0.8315
    Precision:        0.7116
    F1:               0.7669
    AUROC:            0.6739
    AUPR:             0.7916
    


```python
y_train_pred = voting_classifier.predict(X_train.as_matrix()).ravel()
y_test_pred = voting_classifier.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_1.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
```

    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    C:\Users\danil\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    

    
    Performance no conjunto de teste:
    
    Accuracy:         0.6493
    Recall:           0.7997
    Precision:        0.7051
    F1:               0.7495
    AUROC:            0.6650
    AUPR:             0.7853
    
