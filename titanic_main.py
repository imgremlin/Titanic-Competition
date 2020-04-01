import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')

#CONCAT
X = pd.concat([train.drop("Survived", axis=1), test])
y=train['Survived']

titles_dict = {'Lady':'Rare', 'the':'Rare','Capt':'Rare', 'Col':'Rare',
                'Don':'Rare', 'Dr':'Rare','Major':'Rare', 'Rev':'Rare',
                'Sir':'Rare', 'Jonkheer':'Rare', 'Dona':'Rare',
                'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs','Mr':'Mr','Mrs':'Mrs',
                'Miss':'Miss','Master':'Master'}

X['Title']=X.Name.str.extract(",\s(\w+).")
X['Title'] = X['Title'].map(titles_dict)

X['Sex']=X['Sex'].map({'male':1, 'female':0})

X['Family'] = X['SibSp'] + X['Parch'] + 1
X['Family'] = X['Family'].map(lambda f: 'Solo' if f==1 else 'Small' if 2<=f<=4 else 'Big')

X['FareBins'] = pd.qcut(X['Fare'], 4,
         labels=['LowCost','Economy','Business','RichBitch'])

index_NaN_age = list(X["Age"][X["Age"].isnull()].index)

for ind in index_NaN_age:
    p_class=X.iloc[ind-1]['Pclass']
    sx=X.iloc[ind-1]['Sex']
    mean = X[(X.Pclass==p_class) & (X.Sex==sx)]['Age'].mean()
    X['Age'].iloc[ind-1] = mean
    #std = X[(X.Pclass==p_class) & (X.Sex==sx)]['Age'].std()
    #X['Age'].iloc[ind-1] = np.random.randint(mean - std, mean + std)
    
mode_cols = ['Fare', 'Embarked']
X[mode_cols]=X[mode_cols].fillna(test.mode().iloc[0])

X['AgeBins'] = pd.cut(X['Age'], 5,
     labels=['GEN Z','Zoomer','Millenial','Boomer','Greatest'])

#X['Fare'] = X['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

X['Pclass']=X['Pclass'].apply(str)

X['Cabin'] = X['Cabin'].fillna('NONO')
X['Cabin'] = X['Cabin'].map(lambda c: 1 if c=='NONO' else 0)

X.drop(['Name','Ticket'], axis=1, inplace=True)

#drop_list=['Age','SibSp','Parch', 'Fare']
#X.drop(drop_list, axis=1, inplace=True)

train = X.loc[train.index]
train =train.join(y)
test = X.loc[test.index]
#CONCAT

#log age / bin age

def fam():
    f, axes = plt.subplots(2, figsize=(10, 10))
    fam_size=train.groupby('Fam')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fam_size.keys(), y=fam_size.values,
                ax=axes[0]).set_title('Death ratio')
    fam_count_plot_series=train.groupby('Fam')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fam_count_plot_series.keys(), y=fam_count_plot_series.values,
                ax=axes[1]).set_title('Number of each')

def fare():
    f, axes = plt.subplots(2, figsize=(10, 10))
    fare_ratio=train.groupby('FareBins')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fare_ratio.keys(), y=fare_ratio.values,
                ax=axes[0]).set_title('Death ratio')
    fare_count=train.groupby('FareBins')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fare_count.keys(), y=fare_count.values,
                ax=axes[1]).set_title('Number of each')
    
def age():
    f, axes = plt.subplots(2, figsize=(10, 10))
    fam_size=train.groupby('AgeBins')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fam_size.keys(), y=fam_size.values,
                ax=axes[0]).set_title('Death ratio')
    fam_count_plot_series=train.groupby('AgeBins')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fam_count_plot_series.keys(), y=fam_count_plot_series.values,
                ax=axes[1]).set_title('Number of each')

def pclass():
    pclass_death_ration=train.groupby('Pclass')['Survived'].mean().sort_values(ascending=True)
    plt.figure(figsize=(10,7))
    sns.barplot(x=pclass_death_ration.keys(), y=pclass_death_ration.values)
    plt.show()

def count_names(df):
    f, axes = plt.subplots(2, figsize=(10, 10))
    fare_ratio=train.groupby('Title')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fare_ratio.keys(), y=fare_ratio.values,
                ax=axes[0]).set_title('Death ratio')
    fare_count=train.groupby('Title')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fare_count.keys(), y=fare_count.values,
                ax=axes[1]).set_title('Number of each')      
    
def cabin():
    f, axes = plt.subplots(2, figsize=(10, 10))
    fam_size=train.groupby('Cabin')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fam_size.keys(), y=fam_size.values,
                ax=axes[0]).set_title('Death ratio')
    fam_count_plot_series=train.groupby('Cabin')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fam_count_plot_series.keys(), y=fam_count_plot_series.values,
                ax=axes[1]).set_title('Number of each')

def embarked():
    f, axes = plt.subplots(2, figsize=(10, 10))
    fam_size=train.groupby('Embarked')['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fam_size.keys(), y=fam_size.values,
                ax=axes[0]).set_title('Death ratio')
    fam_count_plot_series=train.groupby('Embarked')['Pclass'].count().sort_values(ascending=True)
    sns.barplot(x=fam_count_plot_series.keys(), y=fam_count_plot_series.values,
                ax=axes[1]).set_title('Number of each')

def corr_matrix(matrix):
    corr = matrix.corr()
    plt.subplots(figsize=(10,8))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(0, 200, n=200)) 
    plt.title('Correlation Heatmap of Numeric Features')


#'''
import category_encoders as ce
cat_cols=list(train.select_dtypes(include=['object','category']).columns)
target_enc = ce.CatBoostEncoder(cols=cat_cols)
target_enc.fit(train[cat_cols], train['Survived'])
train = train.join(target_enc.transform(train[cat_cols]).add_suffix('_cb'))
test = test.join(target_enc.transform(test[cat_cols]).add_suffix('_cb'))
train.drop(cat_cols, axis=1, inplace=True)
test.drop(cat_cols, axis=1, inplace=True)


#'''
#train = pd.get_dummies(train,prefix="Dum_")
#test = pd.get_dummies(test,prefix="Dum_")

#print(train.info())
X_pred = train.drop(labels=['Survived'], axis=1)
y = train['Survived']

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=4, random_state=0)

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from xgboost import XGBClassifier
catb = CatBoostClassifier(verbose=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pred, y,
                                        test_size = 0.25, random_state = 0)


def fit_pred(X, y, test):
    catb.fit(X, y)
    catb_pred = catb.predict(test)
    return catb_pred

results = cross_val_score(catb, X_pred, y, cv=kfold, scoring='accuracy')
print(f"res kfold: {results.mean():.5f}")
    
#predictions=fit_pred(X_pred, y, test)

#output = pd.DataFrame({'PassengerId': test.index,'Survived': predictions})
#output.to_csv('submission_dum.csv', index=False)