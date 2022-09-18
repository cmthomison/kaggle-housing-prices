    """
    Functions used throughout the process to load/manipulate/push data.
    """

    import pandas as pd
    import os


    def load_train(path='~/Documents/projects/kaggle-housing-prices/data/train.csv')
        
        data = pd.read_csv(path)

        return data