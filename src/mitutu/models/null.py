
import joblib
import os

# function to save a model using joblib
def save_model(model, model_name = None, file_path = ''):
    """
    A function that saves any model in .joblib format, optionally, in a specified folder.

    Parameters
    ----------
    model:
        Model to be saved (sklearn model or models compatible with .joblib format)
    model_name: str
        Name to be assigned to the saved model file.
    file_path: str
        The path to the directory in which the model is to be stored.

    Returns
    -------
    """

    # save model in given location
    try:
        joblib.dump(model, file_path + f'{model_name}.joblib')
    except:
        print("An error occured while saving your model")
    
# function to load a model
def load_model(model_name: str, file_path: str = '', extension: str = '.joblib'):
    """
    Load a model saved in .joblib format from a specified directory.

    Parameters
    ----------
    model_name : str
        Name of the saved model file (without the .joblib extension).
    file_path : str, default=''
        Path to the directory where the model is stored.

    Returns
    -------
    model
        The loaded model object.

    Raises
    ------
    FileNotFoundError
        If the specified model file does not exist.
    Exception
        If there is an error during model loading.
    """

    full_path = os.path.join(file_path, f"{model_name}.{extension}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at: {full_path}")

    try:
        model = joblib.load(full_path)
        return model
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")

