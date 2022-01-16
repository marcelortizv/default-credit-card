import math
import numpy as np
import pandas as pd

def get_pay_amt_last_3m(df: pd.DataFrame):
    list_check = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']
    if not all(item in list(df.columns) for item in list_check ):
        return df
    else:
        pay_amt_last_3m = pd.Series(df["PAY_AMT1"] + \
                                    df["PAY_AMT2"] + \
                                    df["PAY_AMT3"])
        df['PAY_AMT_LAST_3M'] = pay_amt_last_3m
        return df

def get_pay_amt_last_6m(df: pd.DataFrame):
    list_check = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    if not all(item in list(df.columns) for item in list_check ):
        return df
    else:
        pay_amt_last_6m = pd.Series(df["PAY_AMT1"] + \
                                    df["PAY_AMT2"] + \
                                    df["PAY_AMT3"] + \
                                    df["PAY_AMT4"] + \
                                    df["PAY_AMT5"] + \
                                    df["PAY_AMT6"])
        df['PAY_AMT_LAST_6M'] = pay_amt_last_6m
        return df

def get_bill_amt_last_3m(df: pd.DataFrame):
    list_check = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']
    if not all(item in list(df.columns) for item in list_check ):
        return df
    else:
        pay_amt_last_3m = pd.Series(df["BILL_AMT1"] + \
                                    df["BILL_AMT2"] + \
                                    df["BILL_AMT3"])
        df['BILL_AMT_LAST_3M'] = pay_amt_last_3m
        return df

def get_bill_amt_last_6m(df: pd.DataFrame):
    list_check = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    if not all(item in list(df.columns) for item in list_check ):
        return df
    else:
        pay_amt_last_6m = pd.Series(df["BILL_AMT1"] + \
                                    df["BILL_AMT2"] + \
                                    df["BILL_AMT3"] + \
                                    df["BILL_AMT4"] + \
                                    df["BILL_AMT5"] + \
                                    df["BILL_AMT6"])
        df['BILL_AMT_LAST_6M'] = pay_amt_last_6m
        return df

def get_avg_repayment_3m(df: pd.DataFrame):
    list_check = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']
    if not all(item in list(df.columns) for item in list_check ):
        return df
    else:
        df['AVG_REPAYMENT_3M'] = df[list_check].mean(axis=1)
        return df

def get_avg_repayment_6m(df: pd.DataFrame):
    list_check = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    if not all(item in list(df.columns) for item in list_check):
        return df
    else:
        df['AVG_REPAYMENT_6M'] = df[list_check].mean(axis=1)
        return df

def get_pay_to_limit_ratio_last_3m(df: pd.DataFrame):
    if 'PAY_AMT_LAST_3M' in list(df.columns):
        df['PAY_TO_LIMIT_RATIO_LAST_3M'] = df['PAY_AMT_LAST_3M'] / df['LIMIT_BAL']
        return df
    else:
        return df

def get_pay_to_limit_ratio_last_6m(df: pd.DataFrame):
    if 'PAY_AMT_LAST_6M' in list(df.columns):
        df['PAY_TO_LIMIT_RATIO_LAST_6M'] = df['PAY_AMT_LAST_6M'] / df['LIMIT_BAL']
        return df
    else:
        return df

def get_pay_to_bill_amount_t(df: pd.DataFrame):

    idxs = [1,2,3,4,5,6]

    for idx in idxs:
        column_name = f"PAY_TO_BILL_{idx}"
        df[column_name] = df['PAY_AMT' + str(idx)] / (df['BILL_AMT' + str(idx)] + 1)

    return df

def get_pay_amount_std_last_6m(df: pd.DataFrame):
    list_check = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    if not all(item in list(df.columns) for item in list_check):
        return df
    else:
        df['PAY_AMOUNT_STD_LAST_6M'] = df[list_check].std(axis=1)
        return df

def get_bill_amount_std_last_6m(df: pd.DataFrame):
    list_check = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    if not all(item in list(df.columns) for item in list_check):
        return df
    else:
        df['BILL_AMOUNT_STD_LAST_6M'] = df[list_check].std(axis=1)
        return df

def age_category(age):
    if age <= 40:
        return '<=40'
    elif age <= 60:
        return '<=60'
    elif age <= 80:
        return '<=80'

def get_age_group(df: pd.DataFrame):
    # get age categories
    df['AGE'] = df['AGE'].apply(lambda x: age_category(x))
    # compute one-hot encoding to age variable
    data = pd.get_dummies(df, columns = ['AGE'])
    return data

def get_risky_groups(df: pd.DataFrame):
    # (Gender = Female) & (Education in ('University', 'Graduate School') &
    # (Marital Status in ('Single','Married'))
    df['RISKY_GROUP1'] = np.where(
                            (df['SEX'] == 2) &
                            ((df['EDUCATION'] == 1) | (df['EDUCATION'] == 2)) &
                            ((df['MARRIAGE'] == 1) | (df['MARRIAGE'] == 2)),
                            1, 0)
    # (Gender = Male) & (Education in ('University', 'Graduate School') &
    # (Marital Status = Married)
    df['RISKY_GROUP2'] = np.where(
                            (df['SEX'] == 1) &
                            ((df['EDUCATION'] == 1) | (df['EDUCATION'] == 2)) &
                            (df['MARRIAGE'] == 1),
                            1, 0)
    # (LIMIT_BAL >= 50K & LIMIT_BAL <= 200K) & (AGE_GROUP = (21, 40])
    df['RISKY_GROUP3'] = np.where(
                            (df['AGE_<=40'] == 1) &
                            ((df['LIMIT_BAL'] >= 50000) | (df['LIMIT_BAL'] <= 200000)),
                            1, 0)

    return df


def apply_feature_eng(data: pd.DataFrame):
    # PAY_AMT_LAST_3M
    new_df = get_pay_amt_last_3m(data)
    # PAY_AMT_LAST_6M
    new_df = get_pay_amt_last_6m(new_df)
    # BILL_AMT_LAST_3M
    new_df = get_bill_amt_last_3m(new_df)
    # BILL_AMT_LAST_6M
    new_df = get_bill_amt_last_6m(new_df)
    # AVG_REPAYMENT_3M
    new_df = get_avg_repayment_3m(new_df)
    # AVG_REPAYMENT_6M
    new_df = get_avg_repayment_6m(new_df)
    # PAY_TO_LIMIT_RATIO_LAST_3M
    new_df = get_pay_to_limit_ratio_last_3m(new_df)
    # PAY_TO_LIMIT_RATIO_LAST_6M
    new_df = get_pay_to_limit_ratio_last_6m(new_df)
    # PAY_TO_BILL_AMOUNT_t
    new_df = get_pay_to_bill_amount_t(new_df)
    # PAY_AMOUNT_STD_LAST_6M
    new_df = get_pay_amount_std_last_6m(new_df)
    # BILL_AMOUNT_STD_LAST_6M
    new_df = get_bill_amount_std_last_6m(new_df)
    # AGE_GROUP
    new_df = get_age_group(new_df)  # this function drop variable AGE
    # RISKY_GROUP
    new_df = get_risky_groups(new_df)

    new_df.drop(columns=['ID', 'SEX', 'EDUCATION', 'MARRIAGE'], inplace=True, axis=1)

    return new_df