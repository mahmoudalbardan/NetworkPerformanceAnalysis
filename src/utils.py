import argparse
import configparser
import pickle

def save_model(model, model_path):
    """
    Save a trained model to a specified file path .

    Parameters
    ----------
    model : object
        Trained model object to be saved.
    model_path : str
        Path where the model will be saved.
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    Load a trained model.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    object
        Loaded model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def parse_args():
    """
    Parse command line arguments.
    This function sets up the argument parser to handle command line
    input for the configuration file.

    Returns
    -------
    Namespace
        A Namespace object containing the parsed command line arguments.
        - configuration : str
            Path to the configuration file (default: "configuration.ini").
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=str,
                        help="configuration file", default="configuration.ini")
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
