
# coding: utf-8

# In[1]:


import keras
import pandas
import sklearn
import scipy
import dask.dataframe as dd
import numpy


# In[ ]:


# use this code to transfer between csv to parquet
df.to_parquet("train_data.parquet")


# In[ ]:


training_data = pandas.read_csv("train_data.csv")


# In[ ]:


training_sample = training_data.sample(frac=0.3)


# In[ ]:


training_sample.shape


# In[ ]:


training_sample.to_csv("train_data_sample.csv", index=False)


# # read sample if already available

# In[2]:


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

# ## Data Cleaning

# In[9]:


training_data.dtypes.value_counts()


# Pandas leu muito dos valores de forma errada, gerando até problemas para o uso de memória, existem formas
# melhores de representar estas variáveis.

# In[10]:


training_data.columns


# In[11]:


training_data.dtypes


# In[12]:


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


# In[13]:


for column in training_data.columns:
    print(column)
    values = training_data[column].value_counts()
    if type(values) == list:
        print(values[:10])
    else:
        print(values.head(10))
    print(len(values))
    


# In[14]:


for column in category_columns:
    training_data[column] = training_data[column].astype('category')


# In[15]:


training_data.dtypes.value_counts()


# In[16]:


confunsion_matrix = pandas.crosstab(training_data["SEXO_1"], training_data["IND_BOM_1_1"])
confunsion_matrix


# In[17]:


from scipy.stats import chisquare
chisquare(confunsion_matrix)


# # Feature Selection

# In[18]:


features.corr().abs()


# In[19]:


training_data.head(10)


# In[20]:


corr_matrix = features.corr().abs()


# In[21]:


values = corr_matrix[corr_matrix > 0.95].count()


# In[22]:


values[values > 1]


# In[23]:


corr_matrix[corr_matrix["RENDA_VIZINHANCA_1"] > 0.90]


# RENDA_VIZINHANCA_1 e  RENDA_VIZINHANCA_4 possuem alta correlação e FLAG_BOLSA_FAMILIA_1 e FLAG_PROGRAMAS_SOCIAIS_1
# também. Logo, vou ficar com somente duas das 4.
# Contudo, este teste é para apenas para variáveis correlacionadas linearmente, existem testes melhores para as variáveis categóricas.

# In[24]:


features = features.drop(["RENDA_VIZINHANCA_1", "FLAG_BOLSA_FAMILIA_1"], axis="columns")


# In[25]:


features.shape


# In[26]:


features.drop_duplicates(inplace=True)
features.shape


# In[27]:


training_data_model = pandas.concat([features, labels], axis=1)


# In[28]:


training_data_model.head(10)


# In[29]:


labels.value_counts()


# # Feature Engineering

# In this module we're trying to build the feature that we see will be more useful in order to learn about
# the class we need to predict.

# # Model Tranining

# In[30]:


features = training_data.drop(["IND_BOM_1_1", "IND_BOM_1_2", "id"], axis=1)


# In[31]:


features.head(10)


# In[32]:


labels = training_data["IND_BOM_1_1"]


# In[33]:


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1/4, 
                                                    random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[35]:


input_dimension = X_train.shape[1]
input_dimension


# In[36]:


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, mode='auto')


# In[37]:


classifier_1 = Sequential()
classifier_1.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_1.add(Dense(16, activation='relu', input_dim=input_dimension))
classifier_1.add(Dense(1, activation='sigmoid'))

classifier_1.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])


# In[38]:


classifier_2 = Sequential()
classifier_2.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_2.add(Dense(16, activation='tanh', input_dim=input_dimension))
classifier_2.add(Dense(8, activation='relu', input_dim=input_dimension/2))
classifier_2.add(Dense(1, activation='sigmoid'))

classifier_2.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])


# In[39]:


classifier_3 = Sequential()
classifier_3.add(Dense(16, activation='relu', input_dim=input_dimension))
classifier_3.add(Dense(8, activation='relu', input_dim=input_dimension))
classifier_3.add(Dense(1, activation='sigmoid'))

classifier_3.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])


# I use as_matrix because Keras expects a Numpy array instead of a dataframe.

# In[40]:


classifier_1.summary()


# In[41]:


classifier_2.summary()


# In[42]:


classifier_3.summary()


# In[43]:


model = classifier_1.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)


# In[45]:


model = classifier_3.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)


# In[44]:


model = classifier_2.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping], validation_split=0.15)


# In[46]:


from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.merge import concatenate


# In[47]:


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


# In[53]:


early_stopping_voting = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto')


# In[54]:


voting_classifier.fit(X_train.as_matrix(), y_train.as_matrix(),epochs=500, callbacks=[early_stopping_voting], validation_split=0.15)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier


# In[ ]:


voting_classifier = VotingClassifier([('classifier_1', classifier_1), ('classifier_2', classifier_2), ('classifier_3',classifier_3)], voting='soft')


# In[ ]:


voting_classifier.fit(X_train, y_train)


# # Random Forest

# In[ ]:


rf_clf = RandomForestClassifier()  # Modifique aqui os hyperparâmetros
rf_clf.fit(X_train, y_train)


# In[ ]:


rf_pred_class = rf_clf.predict(X_val)
rf_pred_scores = rf_clf.predict_proba(X_val)[:, 1]
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, rf_pred_class, rf_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[ ]:


rf_pred_test_class = rf_clf.predict(X_test)
rf_pred_test_scores = rf_clf.predict_proba(X_test)[:, 1]
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, rf_pred_test_class, rf_pred_test_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# # XGBoost

# ## Parameter Selection

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


# In[ ]:


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


# In[ ]:


xgboost_classifier.fit(X_train, y_train)


# In[ ]:


# as I said in the documentation, GridSeach uses a stratified 3-fold cross validation because a Classifier was passed
# instead of a regressor

classifier.best_params_


# In[ ]:


# For in ensemble classifiers
classifier = xgboost_classifier


# # Ensemble classifiers (Voting)

# ## Random Forest and XGBoost together

# In[ ]:


classifier = VotingClassifier([('xgboost', xgboost_classifier), ('randomforest', rf_clf)], voting='soft')


# In[ ]:


classifier.fit(X_train, y_train)


# # Evaluation

# In[55]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score


# In[ ]:


y_train_pred = classifier.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier.predict(X_test.as_matrix()).ravel()


# é bom prestar atenção se os valores estão próximo, caso contrário, existe uma boa indicação de que houve
# overfitting e o modelo não consegue generalizar tão bem.

# In[ ]:


print("Mean Square error in train: {:0.1f}".format(mse(y_train, y_train_pred)))
print("Mean Square error in test: {:0.1f}".format(mse(y_test, y_test_pred)))


# In[60]:


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


# In[61]:


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


# In[ ]:


# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier.predict_proba(X_test.as_matrix())[:, 0]


# In[ ]:


y_test_pred.shape


# In[ ]:


y_test.shape


# In[ ]:


y_test_pred_prob.shape


# In[ ]:


print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# # Evaluate for Kaggle

# In[ ]:


kaggle_test_data = pandas.read_csv("real_test_set.csv")


# In[ ]:


kaggle_test_data.shape
features_kaggle = kaggle_test_data.drop(["id"], axis=1)
features_kaggle.head(10)


# In[ ]:


rf_pred_test_class = rf_clf.predict(features_kaggle)
rf_pred_test_scores = rf_clf.predict_proba(features_kaggle)[:, 1]


# In[ ]:


rf_pred_test_class.size


# In[ ]:


rf_pred_test_class


# Se ligar que na hora que cria o csv, na primeira linha (a linha do header), ele coloca ",0", tem que substituir para "id,IND_BOM_1_1"

# In[ ]:


df = pandas.DataFrame(data=rf_pred_test_class)
df.to_csv('test.csv', mode='a', index=True)


# In[ ]:


# For in ensemble classifiers
classifier = xgboost_classifier


# # RESULTS LOG

# ## XGBoost - 120 estimators

# In[ ]:



print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[ ]:



print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# ## XGBoost - 2nd configuration

# In[ ]:


# parameters
number_estimators = [30, 60, 120, 200]
loss_function = ["deviance", "exponential"]
min_samples_leaf = [1, 0.05]
sub_samples = [1.0, 0.8, 0.6]
max_features = ["log2", "sqrt", "auto"]


# In[ ]:


classifier.best_params_


# In[ ]:



print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# # Ensemble Of XBGoost and RandomForest

# # MLPs and Ensemble MLPS

# In[65]:


y_train_pred = classifier_1.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_1.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_1.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[66]:


y_train_pred = classifier_2.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_2.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_2.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[67]:


y_train_pred = classifier_3.predict(X_train.as_matrix()).ravel()
y_test_pred = classifier_3.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_3.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)


# In[68]:


y_train_pred = voting_classifier.predict(X_train.as_matrix()).ravel()
y_test_pred = voting_classifier.predict(X_test.as_matrix()).ravel()

# This returns an array for the probabilities of being each class, so it depends what the focus will be
# in our case, the focus is at class 1
y_test_pred_prob = classifier_1.predict_proba(X_test.as_matrix())[:, 0]

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_test_pred.round(), y_test_pred_prob)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

