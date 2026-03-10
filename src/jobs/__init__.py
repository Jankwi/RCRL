from abc import abstractmethod


class AbstractJob():
    @abstractmethod
    def run(self,):
        pass
