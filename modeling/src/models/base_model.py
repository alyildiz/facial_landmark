from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, output_file):
        self.output_file = output_file

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_over_image(self):
        pass
