

class Component:
    """
    Base class for a software or hardware component.

    Attributes:
    -----------
    name : str
        The name of this component, unique within the system.
    endpoints : dict[str, InterfaceEndpoint]
        The mapping of endpoint names to the endpoints 
        owned by this component.    
    """
    def __init__(self, name: str):
        self.name = name
        self.endpoints: dict[str, InterfaceEndpoint] = {}

    def add_endpoint(self, endpoint: InterfaceEndpoint) -> None:
        endpoint.comp = self
        self.endpoints[endpoint.name] = endpoint