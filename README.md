# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![321851011-c0be1560-2e7d-46ff-a497-aef29aa8a80e](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/e0d60588-1847-4692-805c-38dc861d47c0)
```
data.isnull().sum()
```
![321851135-b6615fa6-bca0-439e-93d9-e9bac897a698](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/dc59a6c9-b63b-49a7-a96e-063bb0b2bdfc)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![321851227-21a5fac7-9da9-4b74-8d55-f48d3947ef3b](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/a463fa41-817d-4798-bd05-5c74cf2e30f7)
```
data2=data.dropna(axis=0)
data2
```
![321851340-e4be12ce-a8b7-495c-ac98-88439c6b1614](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/4f97955c-1815-4b51-8d43-f486637eac5d)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![321851509-6ed747a8-5ccd-47f4-8677-2e59b8aa450a](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/4ef4e4dc-0a74-4784-9d51-1bce3b68656d)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![321851744-4817bd9c-ebf6-4850-873f-d3b4ec85b57a](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/49433396-b401-4658-9379-23ede96eb4b6)
```
data2
```
![321851849-fbf36cc7-93d2-47de-b665-8ec09b3cd540](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/5fffe3b5-5e11-4754-b68d-bf798cefe58f)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![321851985-70618804-02ea-4244-971e-c0c2d35aba7f](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/a4cce0ec-069a-4473-8774-d9be616740f2)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![321852182-ed5ee91b-8bf1-4e8c-a8cf-3d9d96a5589d](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/ae41f09a-eecc-42df-9970-b501ad8cd3eb)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![321852508-5ddd2e17-6819-4f8b-acba-90cf4897797a](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/ff0035c0-d1ff-441d-ae54-acca6b9d5e62)
```
y=new_data['SalStat'].values
print(y)
```
![321852572-ba4ffd91-efc5-4987-8486-49b620e17f41](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/399ba9c9-03ee-475f-a1c8-82a5cda7a68d)
```
x=new_data[features].values
print(x)
```
![321852650-e0f0561b-97cf-4176-bbb5-19614b52c408](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/f7ce5ecc-aea6-49cf-88dc-373439799263)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![321852745-b9d1634e-87a4-42f6-89f9-2d05628018c5](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/c81472f1-e0a0-4a20-925f-ed638000be89)
```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![321852840-3f22074f-9d4d-4758-962c-57f12b70146b](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/bde67ffb-0b17-40c4-9a79-75523a21bc80)
```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![322336946-e31a4e64-7fca-4531-a188-48a5ff07266e](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/e003c8c0-1a2f-4dcf-8b84-0eff90521367)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![322337033-d7291f25-f68a-4c7b-b781-a745058b2770](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/c796fd06-f195-477a-b386-08455b1b3f7c)
```
data.shape
```
![322337115-bcaaa675-3cb4-477f-83b5-5d4fdea4d996](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/b4dc4a57-2a1b-4182-94b6-dfd0d7143051)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![322337195-9263244e-6532-4827-8413-9a0633efbf7d](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/417a5293-8747-4bef-b625-d5ec2dc8d337)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322337431-f6720e9b-9748-4b5f-a1f8-4a565816f67b](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/1cfced9c-3826-4950-9b7c-b07b707fb132)
```
tips.time.unique()
```
![322337526-f4b72a8c-b35a-40df-8649-9123983f7704](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/e89ee82e-80ef-488b-b91e-c481d4480f05)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![322337626-7cb63b0d-f88d-4397-a6c5-9e77145de74d](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/353e2073-5c04-41fb-a7bd-88937b1f7da8)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![322337688-7a261e68-4878-4416-9e90-0a8ff56c5d74](https://github.com/sujigunasekar/EXNO-4-DS/assets/119559822/ab2c5ba8-1500-4bac-adcb-7c79e437d595)

# RESULT:
Thus, Feature selection and Feature scaling has been used and executed in the given dataset.
