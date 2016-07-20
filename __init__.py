from sklearn.pipeline import Pipeline
from base_preprocessor import BasePreprocessor
from base_transformer import BaseTransformer
from composite import CompositeModel

"""
Model Pipeline function has data preprocessing and Composite Tagging model instance.
"""


def get_instance():
    """
    :return: Pipe_line instance
    """
    return model_pipe


model_pipe = Pipeline(
        [('preproc', BasePreprocessor()), ('scaler', BaseTransformer()), ('model', CompositeModel())])
