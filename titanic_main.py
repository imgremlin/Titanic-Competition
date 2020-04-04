import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')

#concatenating train&test
X = pd.concat([train.drop("Survived", axis=1), test])
y=train['Survived']

#extracting title from Name
X['Title']=X.Name.str.extract(",\s(\w+).")

X['Is_Married'] = 0
X.loc[X.Title == 'Mrs', 'Is_Married'] = 1

#making less features through mapping
titles_dict = {'Lady':'Miss/Mrs/Ms', 'the':'Miss/Mrs/Ms','Capt':'Rare', 'Col':'Rare',
                'Don':'Rare', 'Dr':'Rare','Major':'Rare', 'Rev':'Rare',
                'Sir':'Rare', 'Jonkheer':'Rare', 'Dona':'Miss/Mrs/Ms',
                'Mlle':'Miss/Mrs/Ms', 'Ms':'Miss/Mrs/Ms', 'Mme':'Miss/Mrs/Ms',
                'Mr':'Mr','Mrs':'Miss/Mrs/Ms',
                'Miss':'Miss/Mrs/Ms','Master':'Master'}

X['Title'] = X['Title'].map(titles_dict)

#extract surname from Name
X['Family'] = X.Name.str.extract("([A-Za-z ]+),")

X['Sex']=X['Sex'].map({'male':1, 'female':0})

X['Family_Size'] = X['SibSp'] + X['Parch'] + 1

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
X['Family_Grouped'] = X['Family_Size'].map(family_map)

#dealing with missing data via fill in median age of group by Sex&Pclass
X['Age'] = X.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

X['Ticket_Frequency'] = X.groupby('Ticket')['Ticket'].transform('count')

#fill in the most popular feature
X['Embarked']=X['Embarked'].fillna('S')

#deal with missing data via fill in median age of group by Parch&SibSp&Pclass
na_fare = X.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
X['Fare'] = X['Fare'].fillna(na_fare)

#transform continious features
X['Age'] = pd.qcut(X['Age'], 10)
X['Fare'] = pd.qcut(X['Fare'], 13)

#making Pclass string to one hot encode it 
X['Pclass']=X['Pclass'].apply(str)

#filling Cabin missing data
X['Deck'] = X['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

#'T' and 'A' have got almost the same distribution of passengers
X.loc[X['Deck'] == 'T', 'Deck'] = 'A'

#replacing to make less features
X['Deck'] = X['Deck'].replace(['A', 'B', 'C'], 'ABC')
X['Deck'] = X['Deck'].replace(['D', 'E'], 'DE')
X['Deck'] = X['Deck'].replace(['F', 'G'], 'FG')

X.drop(['Cabin'], inplace=True, axis=1)

train = X.loc[train.index]
train =train.join(y)
test = X.loc[test.index]

#function to calculate survival rate with help of Family&Tickets
non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]
non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]

df_family_survival_rate = train.groupby('Family')['Survived', 'Family','Family_Size'].median()
df_ticket_survival_rate = train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

mean_survival_rate = np.mean(train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(train)):
    if train['Family'].iloc[i] in family_rates:
        train_family_survival_rate.append(family_rates[train['Family'].iloc[i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
train['Family_Survival_Rate'] = train_family_survival_rate
train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
test['Family_Survival_Rate'] = test_family_survival_rate
test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(train)):
    if train['Ticket'].iloc[i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[train['Ticket'].iloc[i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
train['Ticket_Survival_Rate'] = train_ticket_survival_rate
train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
test['Ticket_Survival_Rate'] = test_ticket_survival_rate
test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

for df in [train, test]:
    df['Survival_Rate'] = 0.5*(df['Ticket_Survival_Rate'] + df['Family_Survival_Rate'])
    df['Survival_Rate_NA'] = 0.5*(df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA'])   


#using Label Encoder to transform categorical features to numerical
non_numeric_features = ['Embarked', 'Deck', 'Title', 'Family_Grouped', 'Age', 'Fare']

for df in [train, test]:
    for feature in non_numeric_features:        
        df[feature] = LabelEncoder().fit_transform(df[feature])

cat_features = ['Embarked', 'Deck', 'Title', 'Family_Grouped','Pclass', 'Sex']

#creating one hot encoded features
encoded_features = []

for df in [train, test]:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

#adding encoded features
train = pd.concat([train, *encoded_features[:6]], axis=1)
test = pd.concat([test, *encoded_features[6:]], axis=1)


#dropping unnecessary features
drop_cols = ['Deck', 'Embarked', 'Family',  'Family_Grouped',
             'Name',  'Pclass',  'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate',
            'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA',
            'Sex', 'SibSp','Parch','Family_Size']

train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)


def count_survival_plot(column):
    f, axes = plt.subplots(2, figsize=(10, 10))
    fam_size=train.groupby(column)['Survived'].mean().sort_values(ascending=True)
    sns.barplot(x=fam_size.keys(), y=fam_size.values,
                ax=axes[0]).set_title('Death ratio')
    fam_count_plot_series=train.groupby(column)['Pclass'].count().sort_values(ascending=True)
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

#creating model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1750, max_depth=7, 
                            min_samples_split=6, min_samples_leaf=6,
                            max_features='auto', random_state=42) 

#scaling our df`s
X_train = StandardScaler().fit_transform(train.drop(columns=['Survived']))
y_train = train['Survived'].values
X_test = StandardScaler().fit_transform(test)

def feature_importance_plot():
    rf.fit(X_train,y_train)
    feat_series = pd.Series(rf.feature_importances_, index =train.drop(columns=['Survived']).columns).sort_values(ascending=False)
    plt.figure(figsize=(10,7))
    sns.barplot(y=feat_series.keys(), x=feat_series.values)

#number of folds 
split_num = 4

#making dataframe to check cross validation results
probs = pd.DataFrame(np.zeros((len(X_test), split_num )), columns=['fold_{}'.format(i) for i in range(1, split_num + 1)])

from sklearn.model_selection import KFold

kfold = KFold(n_splits=split_num, random_state=0)

for fold, (trn_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):

    rf.fit(X_train[trn_idx], y_train[trn_idx])
    probs.loc[:, 'fold_{}'.format(fold)] = rf.predict_proba(X_test)[:, 1] 

#calculating accuracy of our model
from sklearn.model_selection import cross_val_score

results = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='accuracy')
print(f"res kfold: {results.mean():.5f}")

#final result
probs['res'] = round(probs.mean(axis=1)).astype(int)

#making submission
output = pd.DataFrame({'PassengerId': test.index,'Survived': probs['res']})
output.to_csv('submission.csv', index=False)
