import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_labels(label_path, fill=True):
    labels = pd.read_csv(label_path)

    labels['Sex'] = labels['Sex'].mask(
                            labels['Sex'] == 'Female', 1).mask(
                            labels['Sex'] == 'Male', -1).mask(
                            labels['Sex'] == 'Unknown', 0)
    labels['Frontal/Lateral'] = labels['Frontal/Lateral'].mask(
                            labels['Frontal/Lateral'] == 'Frontal', 1).mask(
                            labels['Frontal/Lateral'] == 'Lateral', -1)
    labels['AP/PA'] = labels['AP/PA'].mask(
                            labels['AP/PA'] == 'PA', 1).mask(
                            labels['AP/PA'] == 'AP', 0).fillna(-1)

    labels.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    labels.set_index('id', drop=True, inplace=True)
    
    if fill:
        mses = np.zeros((81,14))
        train, test = train_test_split(np.arange(labels.shape[0]))
        rs = np.linspace(-1.5, 1.5 ,81)
        cols = labels.columns[5:]

        fill = labels[cols]
        pred = np.asarray(fill.iloc[train]).mean(axis=0) * np.ones((len(test), fill.shape[1]))
        mses[-1,:] = np.mean((pred - np.asarray(fill.iloc[test]))**2, axis=0)

        for i, r in enumerate(rs):
            fill = labels[cols].fillna(r)
            filltrain = fill.iloc[train]
            filltest = fill.iloc[test]
            pred = np.asarray(filltrain).mean(axis=0) * np.ones((len(test), fill.shape[1]))
            mses[i,:] = np.mean((pred - np.asarray(filltest))**2, axis=0)

        fills = np.zeros(14)
        argmins = mses.argmin(axis=0)
        for n in range(14):
            if argmins[n] == len(rs):
                fills[n] = None
            else:
                fills[n] = rs[argmins[n]]

        for i, col in enumerate(labels.columns[5:]):
            labels[col] = labels[col].fillna(fills[i])
    
    return labels


if __name__ == "__main__":
    
    label_path = "/groups/CS156b/data/student_labels/train.csv"

    processed_labels = load_labels(label_path, fill=True)
    processed_labels.to_csv('/groups/CS156b/2023/BbbBbbB/labels.csv')