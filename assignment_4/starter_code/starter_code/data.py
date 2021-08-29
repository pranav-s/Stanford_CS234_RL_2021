import numpy as np
import pandas as pd

import pdb

LABEL_KEY = 'Therapeutic Dose of Warfarin'


# Modifies given column of df in-place, removing null values
def impute(df, column_name, how, value=None):
    column = df[column_name]
    if how == 'mean':
        imputed_value = column.mean()
    elif how == 'mode':
        imputed_value = column.mode()
    elif how == 'given':
        imputed_value = value
    else:
        raise ValueError(f'Unknown imputation method: {how}')
        
    df[column_name] = column.fillna(imputed_value)

def _transform_age(age):
    if pd.isnull(age):
        return age
        
    for decade in range(1,9):
        if age == f'{decade}0 - {decade}9':
            return decade
    if age == '90+':
        return 9
    
    raise RuntimeError('Unknown age format')


def load_data():
    data = pd.read_csv('data/warfarin.csv')
    
    # Remove individuals for whom we don't know correct dosage
    data = data[pd.notnull(data[LABEL_KEY])]

    data['Male'] = (data['Gender'] == 'male').astype(float)
    data['Female'] = (data['Gender'] == 'female').astype(float)
    data['Asian'] = (data['Race'] == 'Asian').astype(float)
    data['Black'] = (data['Race'] == 'Black or African American').astype(float)
    data['White'] = (data['Race'] == 'White').astype(float)
    data['Unknown race'] = (data['Race'] == 'Unknown').astype(float)
    data['Age in decades'] = data['Age'].transform(_transform_age)
    
    impute(data, 'Height (cm)', 'mean')
    impute(data, 'Weight (kg)', 'mean')
    impute(data, 'Age in decades', 'mean')
    impute(data, 'Amiodarone (Cordarone)', 'given', value=0)
    impute(data, 'Carbamazepine (Tegretol)', 'given', value=0)
    impute(data, 'Phenytoin (Dilantin)', 'given', value=0)
    impute(data, 'Rifampin or Rifampicin', 'given', value=0)


    a = data['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].to_numpy()
    data['VKORC1AG'] = [1 if elmt == 'A/G' else 0 for elmt in a]
    data['VKORC1AA'] = [1 if elmt == 'A/A' else 0 for elmt in a]
    V_nan = pd.isnull(a)
    data['VKORC1UN'] = [1 if V_nan[i] else 0 for i in range(len(a))]

    c = data['Combined QC CYP2C9'].to_numpy()
    data['CYP2C912'] = [1 if elmt == '*1/*2' else 0 for elmt in c]
    data['CYP2C913'] = [1 if elmt == '*1/*3' else 0 for elmt in c]
    data['CYP2C922'] = [1 if elmt == '*2/*2' else 0 for elmt in c]
    data['CYP2C923'] = [1 if elmt == '*2/*3' else 0 for elmt in c]
    data['CYP2C933'] = [1 if elmt == '*3/*3' else 0 for elmt in c]
    C_nan = pd.isnull(c)
    data['CYP2C9UN'] = [1 if C_nan[i] else 0 for i in range(len(c))]

    return data