import re

class NamedObject:
    """
    Base class for objects that have a type name and instance name. 

    Attributes: 
    -----------
    name : str
        The name of this instance of the object, unique within the system.
    type_name : str | None
        The type name to use for this object in visualizations. If None, the snake case of class name is used.
    """
    type_name = None

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instance_count = 0
        if cls.type_name is None:
            cls.type_name = NamedObject._camel_to_snake(cls.__name__)

    def __init__(
            self, 
            name : str | None =None):
        cls = type(self)
        if name is None:
            self.name = f"{cls.type_name}{cls._instance_count}"
            cls._instance_count += 1
        else:
            self.name = name