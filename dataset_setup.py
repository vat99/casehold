import pandas as pd
import numpy as np

def train_validate_test_split(df, train_percent=.64, validate_percent=.16, seed=1234):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def main():
    df = pd.read_csv("./unprocessed_data/casehold.csv")
    train, validate, test = train_validate_test_split(df)
    
    train.to_csv("./data_processed/train.csv", sep=',', header=True, index=False, encoding='utf-8')
    validate.to_csv("./data_processed/dev.csv", sep=',', header=True, index=False, encoding='utf-8')
    test.to_csv("./data_processed/test.csv", sep=',', header=True, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()