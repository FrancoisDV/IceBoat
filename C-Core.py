
# coding: utf-8

# Import relevant packages

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import ijson


#Constants

PATH_TO_TRAIN_DATA = '/home/francois/Notebooks/C-Core Iceberg/data/processed/train.json'
PATH_TO_TEST_DATA = '/home/francois/Notebooks/C-Core Iceberg/data/processed/test.json'

# Functions

#Returns a resized np.array of shape (75,75) of an input flatten image

def make_img_from_serie(Serie):
    return np.array(Serie).reshape(75, 75)

#takes a dataframe of a flatten img serie as an input and plot them
#legend labels are indications plotted on the left-hand side

def plot_data_frame(df,plot_labels,legend_labels):
    N_samples, _ = df.shape
    gs = gridspec.GridSpec(N_samples, len(plot_labels)+1,width_ratios=[1, 3, 3])
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(4*N_samples)
    j = 0
    for index_j,row in df.iterrows():
        legend = ''
        for k,legend_label in enumerate(legend_labels):
            legend += '%s : %s \n' % (legend_label, row[legend_label])
        ax = plt.subplot(gs[j, 0])
        ax.axis('off')
        ax.text(0,0,legend)
        for i,plot_label in enumerate(plot_labels):
            ax = plt.subplot(gs[j, i+1])
            ax.imshow(make_img_from_serie(row[plot_label]), cmap = 'gray')
        j+= 1   
    plt.show()

#Compares predictions and ground-truth, prints score and returns a boolean array of results

def compare_predictGT(predictions,ground_truth):
    score = (predictions == ground_truth)
    print float(sum(score))/len(ground_truth)
    return score

#shuffles df and splits it, returns 2 arrays learn and validation df. 
#rate is the ratio of learn data

def shuffle_and_split(df,rate):
    shuffle_df = df.sample(frac=1)
    l = len(df)
    split = np.floor(rate*l).astype(int)
    val_df = shuffle_df.iloc[split:l]
    learn_df = shuffle_df.iloc[0:split] 
    return learn_df, val_df


def test_module():
    # Import training data
    train_df = pd.read_json(PATH_TO_TRAIN_DATA)

    # Split data (learn and validation)
    learn_df, val_df = shuffle_and_split(train_df, 0.7)

    # Prepare Dataset for training and validation
    learn_df_bands_1 = np.stack(np.array(band) for band in learn_df['band_1'])
    learn_df_result = np.array(learn_df['is_iceberg'])
    val_df_bands_1 = np.stack(np.array(band) for band in val_df['band_1'])
    val_df_result = np.array(val_df['is_iceberg'])

    # Define and train Model
    """svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearSVC(C=1, loss='hinge')),
    ])
    
    svm_clf.fit(learn_df_bands_1, learn_df_result)

    # Get score
    #predictions = svm_clf.predict(val_df_bands_1)
    score = compare_predictGT(predictions, val_df_result)
    """
    # Save Model

    # Plot False results

    # plot_data_frame(false_df.sample(10),['band_1','band_2'],['id','inc_angle','is_iceberg'])


    # Import test data

    try:
        f = open(PATH_TO_TEST_DATA,'r')
    except IOError:
        print 'cannot open'
    i = 0
    for prefix, event, value in ijson.parse(f):

        if (event) == ("start_map"):
            id = ""
            band1 = []
            band2 = []
            angle = 0
        if prefix == "item.id":
            id = value
            print id
        if prefix == "item.band_2.item":
            band2.append(str(value))
        if prefix == "item.band_1.item":
            band1.append(str(value))
        if prefix == "item.inc_angle":
            angle = value
        #if event == "end_map":





if __name__ == "__main__":
    test_module()