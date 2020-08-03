import pandas as pd

if __name__ == '__main__':
    fileName = r'./data/air_data.csv'
    data = pd.read_csv(fileName, encoding='gb18030')
    data.drop(['MEMBER_NO'], axis=1, inplace=True)
    data.describe()
    exploredFile = r'./data/data_explored.csv'
    data.to_csv(exploredFile, encoding='gb18030')
    dataFile = r'./data/new_data.csv'
    data.to_csv(dataFile, encoding='gb18030')
