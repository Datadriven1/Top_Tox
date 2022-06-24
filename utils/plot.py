import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pandas as pd

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i],ha="center",rotation = "vertical", va="top", fontsize="x-large")

def save_plot(model_name, accuracy, plot, plot_dir_path, dataset_name, data_used):
    unique_filename = get_unique_filename(plot)
    plt.subplots(figsize=(10,8))
    plt.bar(model_name, accuracy ,color ='green')
    addlabels(model_name,accuracy)
    plt.xlabel("Model Name")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name}\n{data_used}")
    plt.xticks(rotation = 20, fontsize=8)
    plotPath = os.path.join(plot_dir_path, unique_filename)
    plt.savefig(plotPath)

def analysis_plot(file_path, plot_dir_path,unique_filename, dataset_name):
    plt.subplots(figsize=(10,8))
    df = pd.read_csv(file_path)
    sns.barplot(data=df, y='Test accuracy', x='model_name', hue='features_used')
    plt.xticks(rotation = 20, fontsize=8)
    plt.title(dataset_name)
    plt.ylim(0, 1.15)
    plotPath = os.path.join(plot_dir_path, unique_filename)
    plt.savefig(plotPath)