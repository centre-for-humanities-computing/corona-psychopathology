"""
A script for creating test train set
"""

import argparse
import os


#"\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\Corona_projekt_psyk_samlet_data_efter_3runde_12_april_inkl_pato_V2.xlsx"

path=os.getcwd()
os.chdir("E:\\Users\\adminanddan\\Desktop\\corona-psychopathology-master")
path=os.getcwd()
print(path)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from utils import resample as rs


def split_to_train_test(data="\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\Corona_projekt_psyk_samlet_data_efter_3runde_12_april_inkl_pato_V2.xlsx",
                        text_column="text",
                        label_column="labels",
                        resample="over",
                        perc_test=0.3,
                        sampling_strategy=1,
                        **kwargs):

    df = pd.read_excel(data)
    
    columns_toDrop = ['Psyko_pato','SFI_Navn','Datotid',\
                  'ADiagnoseKodeTekst','DiagnoseGruppeStreng','KOEN',\
                  'ALDER_KOR','KontakttypeEPJ','afdelingstekst',\
                  'Corona_COVID','virus_smitte_epidemi_pandemi']
    df = df.drop(columns_toDrop, axis=1) 
        
    df.loc[df['konklusion_efter_3_runde'] == 3, 'konklusion_efter_3_runde'] = 0
    df.loc[df['konklusion_efter_3_runde'] == 2, 'konklusion_efter_3_runde'] = 1
    df = df.rename(columns={"Fritekst": text_column, "konklusion_efter_3_runde": label_column})
 
    #splitter på patienter
    train_inds, test_inds = next(GroupShuffleSplit(random_state=42, test_size=0.3).split(df, groups=df["DW_EK_Borger"]))
    train, test = df.iloc[train_inds], df.iloc[test_inds]
    
    train = train.drop("DW_EK_Borger", axis=1)
    test = test.drop("DW_EK_Borger", axis=1)
    
    if resample:
        train = rs(train, "labels",
                   method=resample,
                   sampling_strategy=sampling_strategy,
                   **kwargs)
    
    train.to_csv("\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\train.csv")
    test.to_csv('\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\test.csv')
    return(train, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data",
                        help="filename",
                        default="\\\\TSCLIENT\X\Sites\\BETTRS\\documentLibrary\\BI data\\Data efter 3. runde screening\\Corona_projekt_psyk_samlet_data_efter_3runde_12_april_inkl_pato_V2.xlsx")
    parser.add_argument("-tc", "--text_column",
                        help="columns for text",
                        default="text")
    parser.add_argument("-lc", "--label_column",
                        help="columns for labels",
                        default="labels")
    parser.add_argument("-pt", "--perc_test",
                        help="percent data to use as train",
                        default=0.3, type=float)
    parser.add_argument("-rs", "--resample",
                        help="what method should you use to resample",
                        default=None)

    # parse and clean args:
    args = parser.parse_args()
    args = vars(args)  # make it into a dict

    print("Calling split_to_train_test with args:", args)
    split_to_train_test(**args)
