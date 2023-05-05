from abc import ABC
from hydra.utils import instantiate


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """
    
    def after_train_iter(self, trainer,**kwargs):
        pass

    def __getattr__(self, item):
        return super().__getattr__(item)



class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """

    def __init__(self, callbacks):
        self.callbacks = []
        if callbacks is not None:
            self.add_callbacks(callbacks)

    def add_callbacks(self, callbacks):
        """
        Add a list of callbacks to the callback handler
        :param callbacks: a list of :class:`TrainerCallback`
        """
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        """
        Add a callbacks to the callback handler
        :param callback: an instance of a subclass of :class:`TrainerCallback`
        """
        cb = instantiate(callback)
        self.callbacks.append(cb)

    def __iter__(self):
        return iter(self.callbacks)

    def clear_callbacks(self):
        self.callbacks = []

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)


