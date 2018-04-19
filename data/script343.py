
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white')


# In[ ]:


train_df = pd.read_csv('../input/train.csv', parse_dates=['Original_Quote_Date'], thousands=',')


# In[ ]:


#train_df.QuoteConversion_Flag.value_counts(normalize=True)


# In[ ]:


def engineer_features(df):
    for column in df.columns[df.isin(['N', 'Y']).any()]:
        unique_values = df[column].unique()
        if len(unique_values) < 4:
            df[column] = df[column].map(lambda x: 1 if x == 'Y' else 0 if x == 'N' else 0.5)

    df.PersonalField84 = df.PersonalField84.fillna(0)
    df.PropertyField29 = df.PropertyField29.fillna(-1)

    df = df.drop(['GeographicField10A', 'PropertyField6'], axis=1)

    df['Original_Quote_Date_Year'] = df.Original_Quote_Date.map(lambda t: t.year)
    df['Original_Quote_Date_Month'] = df.Original_Quote_Date.map(lambda t: t.month)
    df['Original_Quote_Date_Day'] = df.Original_Quote_Date.map(lambda t: t.day)
    df['Original_Quote_Date_Weekday'] = df.Original_Quote_Date.map(lambda t: t.dayofweek)
    df['Original_Quote_Date_Week'] = df.Original_Quote_Date.map(lambda t: t.weekofyear)

    df = df.drop(['Original_Quote_Date', 'QuoteNumber'], axis=1)

    num_train_df = df.select_dtypes(include=['number'], exclude=['datetime64'])
    ord_train_df = df.select_dtypes(include=['object']).apply(lambda s: s.astype('category').cat.codes)
    df = pd.concat([num_train_df, ord_train_df], axis=1)

    for column in df.columns:
        if column != 'QuoteConversion_Flag':
            df[column + '_Count'] = df[[column]].groupby([column])[column].transform(len)

    return df


# In[ ]:


train_df = engineer_features(train_df)


# In[ ]:


#train_df.columns.values


# In[ ]:


variance_s = train_df.apply(lambda c: c.var())
variance_s.sort_values()[:50]


# In[ ]:


variance_s.describe()


# In[ ]:


def plot_corr(df):
    f, ax = plt.subplots(figsize=(65, 18))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidths=1, cbar=False, ax=ax) #cbar_kws={"shrink": 1}


# In[ ]:


corr_date_df = train_df.filter(
    regex='^Original|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_date_df)


# In[ ]:


# train_df['Field8_9'] = train_df.Field8 + train_df.Field9 * 10000
corr_field_df = train_df.filter(
    regex='^Field|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_field_df)


# In[ ]:


# train_df['CoverageField1_2_3_4'] =\
#     train_df.CoverageField1A.map(str) + '|' + train_df.CoverageField1B.map(str) + '|' +\
#     train_df.CoverageField2A.map(str) + '|' + train_df.CoverageField2B.map(str) + '|' +\
#     train_df.CoverageField3A.map(str) + '|' + train_df.CoverageField3B.map(str) + '|' +\
#     train_df.CoverageField4A.map(str) + '|' + train_df.CoverageField4B.map(str)
# train_df['CoverageField1_2_3_4'] = train_df.CoverageField1_2_3_4.astype('category').cat.codes
corr_coverage_df = train_df.filter(
    regex='^Coverage|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_coverage_df)


# In[ ]:


# train_df['SalesField1_2'] =\
#     train_df.SalesField1A.map(str) + '|' + train_df.SalesField1B.map(str) + '|' +\
#     train_df.SalesField2A.map(str) + '|' + train_df.SalesField2B.map(str)
# train_df['SalesField1_2'] = train_df.SalesField1_2.astype('category').cat.codes
# train_df['SalesField11_12'] = train_df.SalesField11.map(str) + '|' + train_df.SalesField12.map(str)
# train_df['SalesField11_12'] = train_df.SalesField11_12.astype('category').cat.codes
# train_df['SalesField14_15'] = train_df.SalesField14.map(str) + '|' + train_df.SalesField15.map(str)
# train_df['SalesField14_15'] = train_df.SalesField14_15.astype('category').cat.codes
corr_sales_df = train_df.filter(
    regex='^Sales|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_sales_df)


# In[ ]:


corr_personal_df = train_df.filter(
    regex='^PersonalField[1-2]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_personal_df)


# In[ ]:


corr_personal_df = train_df.filter(
    regex='^PersonalField[3-4]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_personal_df)


# In[ ]:


corr_personal_df = train_df.filter(
    regex='^PersonalField[5-6]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_personal_df)


# In[ ]:


corr_personal_df = train_df.filter(
    regex='^PersonalField[7-8]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_personal_df)


# In[ ]:


corr_property_df = train_df.filter(
    regex='^Property|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_property_df)


# In[ ]:


corr_geographic_df = train_df.filter(
    regex='^GeographicField[1-2]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_geographic_df)


# In[ ]:


corr_geographic_df = train_df.filter(
    regex='^GeographicField[3-4]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_geographic_df)


# In[ ]:


corr_geographic_df = train_df.filter(
    regex='^GeographicField[5-6]|QuoteConversion_Flag').sort_index(axis=1).corr(method='spearman')
plot_corr(corr_geographic_df)

