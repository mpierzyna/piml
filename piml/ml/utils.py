import piml

from sklearn.base import TransformerMixin


def get_custom_tf(ws: piml.Workspace, tf_name: str) -> TransformerMixin:
    """ Get custom transformer from `custom_code.py` in workspace. """
    if not tf_name.startswith("custom_code."):
        raise ValueError(f"Custom transformers must start with 'custom_code.' but got '{tf_name}'.")
    tf_name = tf_name[len("custom_code."):]

    try:
        return getattr(ws.custom_code, tf_name)
    except AttributeError:
        raise ValueError(f"Custom transformer '{tf_name}' not found in custom_code.py.")
