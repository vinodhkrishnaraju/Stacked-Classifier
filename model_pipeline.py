from sklearn.pipeline import Pipeline
from base_preprocessor import BasePreprocessor
from base_transformer import BaseTransformer
from composite import PhaseOneTaggingModel

"""
Model Pipeline function has data preprocessing and Composite Tagging model instance.
"""


def getModelInstance():
    """
    :return: Pipe_line instance
    """
    return model_pipe

def getTransformInstance():
    """
    :return: Pipe_line instance
    """
    return transform_pipe


model_pipe = Pipeline(
        [('preproc', BasePreprocessor()), ('scaler', BaseTransformer()), ('model', PhaseOneTaggingModel())])

transform_pipe = Pipeline(
        [('preproc', BasePreprocessor()), ('scaler', BaseTransformer())])
