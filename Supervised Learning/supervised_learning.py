import pandas as pd
import arff

def loadPhishingDataSet():
    arff_all = arff.load(open('datasets/phishing/Training Dataset.arff.txt'), 'rb')

    #  Put data into dataframe
    df_1_all = pd.DataFrame(arff_all["data"], columns=pd.DataFrame(arff_all["attributes"])[0])

    #  Split into training and testing sets
    split_ratio = .65
    split_point = int(len(df_1_all) * split_ratio)

    df_training = df_1_all[0:split_point]
    df_testing  = df_1_all[split_point:]
    
    return df_training, df_testing

def loadOccupancyDataSet():
    df_all = pd.read_csv("datasets/occupancy/datatraining.txt")
    df_all = df_all.drop("date", 1)
    #  Split into training and testing sets
    split_ratio = .65
    split_point = int(len(df_all) * split_ratio)

    df_training = df_all[0:split_point]
    df_testing  = df_all[split_point:]
    
    return df_training, df_testing

def loadBankDataSet():
    df_all = pd.read_csv("datasets/bank/bank-full.csv", delimiter=";")
    mapped_values = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
    for value in mapped_values:
        df_all[value] = df_all[value].map({j:i for i,j in enumerate(df_all[value].unique())})
       
    #  Split into training and testing sets
    split_ratio = .65
    split_point = int(len(df_all) * split_ratio)

    df_training = df_all[0:split_point]
    df_testing  = df_all[split_point:]

    return df_training, df_testing
