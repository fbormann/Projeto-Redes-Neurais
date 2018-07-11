
# coding: utf-8

# In[2]:


import keras
import pandas
import sklearn
import scipy
import dask.dataframe as dd


# In[3]:


df = dd.read_csv("train_data.csv")


# In[ ]:


# use this code to transfer between csv to parquet
df.to_parquet("train_data.parquet")


# In[6]:


training_data = pandas.read_csv("train_data.csv")


# In[7]:


training_sample = training_data.sample(frac=0.3)


# In[8]:


training_sample.shape


# In[5]:


training_sample.to_csv("train_data_sample.csv", index=False)


# # read sample if already available

# In[3]:


training_data = pandas.read_csv("train_data_sample.csv")


# In[4]:


training_data.columns


# In[5]:


# keep ids and their index on database for further reference
ids = training_data["id"]


# In[6]:


features = training_data.drop(["IND_BOM_1_1", "IND_BOM_1_2", "id"], axis=1)


# In[7]:


features.head(10)


# In[8]:


labels = training_data["IND_BOM_1_1"]


# Vou tentar reduzir a dimensionalidade do dataframe utilizando LDA para poder analisar 
# melhor um subconjunto de variáveis e mensurar a acurácia

# In[9]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[10]:


lda = LinearDiscriminantAnalysis(n_components=50)


# In[11]:


features_50 = lda.fit(features, labels).transform(features)


# In[12]:


features_50[10]


# ## Data Cleaning

# In[13]:


training_data.dtypes.value_counts()


# Pandas leu muito dos valores de forma errada, gerando até problemas para o uso de memória, existem formas
# melhores de representar estas variáveis.

# In[14]:


training_data.columns


# In[15]:


training_data.dtypes


# In[16]:


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


# In[17]:


for column in training_data.columns:
    print(column)
    values = training_data[column].value_counts()
    if type(values) == list:
        print(values[:10])
    else:
        print(values.head(10))
    print(len(values))
    


# In[18]:


for column in category_columns:
    training_data[column] = training_data[column].astype('category')


# In[19]:


training_data.dtypes.value_counts()


# In[20]:


confunsion_matrix = pandas.crosstab(training_data["SEXO_1"], training_data["IND_BOM_1_1"])
confunsion_matrix


# In[21]:


from scipy.stats import chisquare
chisquare(confunsion_matrix)


# # Feature Selection

# In[22]:


features.corr().abs()


# In[23]:


training_data.head(10)


# In[24]:


corr_matrix = features.corr().abs()


# In[25]:


values = corr_matrix[corr_matrix > 0.95].count()


# In[26]:


values[values > 1]


# In[27]:


corr_matrix[corr_matrix["RENDA_VIZINHANCA_1"] > 0.90]


# RENDA_VIZINHANCA_1 e  RENDA_VIZINHANCA_4 possuem alta correlação e FLAG_BOLSA_FAMILIA_1 e FLAG_PROGRAMAS_SOCIAIS_1
# também. Logo, vou ficar com somente duas das 4.
# Contudo, este teste é para apenas para variáveis correlacionadas linearmente, existem testes melhores para as variáveis categóricas.

# In[28]:


features = features.drop(["RENDA_VIZINHANCA_1", "FLAG_BOLSA_FAMILIA_1"], axis="columns")


# In[29]:


features.shape


# In[30]:


features.drop_duplicates(inplace=True)
features.shape


# In[31]:


training_data_model = pandas.concat([features, labels], axis=1)


# In[32]:


training_data_model.head(10)


# In[77]:


labels.value_counts()


# # Feature Engineering

# In this module we're trying to build the feature that we see will be more useful in order to learn about
# the class we need to predict.

# In[ ]:





# # Model Tranining

# In[37]:


features = training_data.drop(["IND_BOM_1_1", "IND_BOM_1_2", "id"], axis=1)


# In[38]:


features.head(10)


# In[39]:


labels = training_data["IND_BOM_1_1"]


# In[46]:


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1/4, 
                                                    random_state=42, stratify=labels)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
#                                                  random_state=42, stratify=y_train)


# In[52]:


input_dimension = X_train.shape[1]
input_dimension


# In[53]:


classifier = Sequential()
classifier.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='mean_squared_error')


# I use as_matrix because Keras expects a Numpy array instead of a dataframe.

# In[81]:


classifier.summary()


# In[56]:

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0)
]

model = classifier.fit(X_train.as_matrix(), y_train.as_matrix())


# # Evaluation

# In[67]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score


# In[62]:


y_train_pred = classifier.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier.predict(X_test.as_matrix()).ravel()


# é bom prestar atenção se os valores estão próximo, caso contrário, existe uma boa indicação de que houve
# overfitting e o modelo não consegue generalizar tão bem.

# In[65]:


print("Mean Square error in train: {:0.1f}".format(mse(y_train, y_train_pred)))
print("Mean Square error in test: {:0.1f}".format(mse(y_test, y_test_pred)))


# In[69]:


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


# In[79]:


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


# In[73]:


y_test_pred_prob = classifier.predict_proba(X_test.as_matrix())


# In[76]:


y_test_pred


# In[80]:


print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[ ]:




