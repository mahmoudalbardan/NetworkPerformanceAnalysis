import argparse
import pickle
import configparser



def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def parse_args():
    """
    Parse command line arguments.
    This function sets up the argument parser to handle command line
    input for the configuration file and retrain flag. It defines
    the expected arguments and their types.

    Returns
    -------
    Namespace
        A Namespace object containing the parsed command line arguments.
        - configuration : str
            Path to the configuration file (default: 'configuration.ini').
        - eda : str
            Flag indicating whether to do exploratory data analysis (plot histograms) ('true' or 'false'; default: 'false').
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=str,
                        help='configuration file', default='configuration.ini')
    args = parser.parse_args()
    return args


def get_config(configfile):
    """
    Read a configuration file.
    This function reads the specified configuration file using the
    configparser module and returns the configuration object.

    Parameters
    ----------
    configfile : str
        The path to the configuration file to read.

    Returns
    -------
    config : configparser.ConfigParser
        A ConfigParser object containing the configuration settings
        loaded from the specified file.
    """
    config = configparser.ConfigParser()
    config.read(configfile)
    return config
