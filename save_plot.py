from utils.plot import analysis_plot
from utils.all_utils import *
import argparse

def save_plot(config_path):
    config = read_config(config_path)

    artifacts_dir = config["artifacts"]["artifacts_dir"]

    plots_dir = config["artifacts"][f"plots_dir1"]

    plot_dir_path = os.path.join(artifacts_dir, plots_dir)

    analysis_plot('result/2171_paper.csv', plot_dir_path, "2171_paper", "2171_paper")

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", help="Path to config file", default="config.yaml")
    parsed_args = args.parse_args()
    
    save_plot(config_path=parsed_args.config)