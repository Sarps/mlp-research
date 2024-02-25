from dataclasses import dataclass
from typing import Any, Union, List, Tuple
from keras.models import Model
from keras.layers import Input, Layer
import keras
from jsonpath_ng import jsonpath, parse


@dataclass
class Connection:
    layer: str
    local_key: jsonpath.JSONPath
    foreign_key: jsonpath.JSONPath


@dataclass
class ConnectionGroup:
    name: str
    connections: list[Connection]
    index: int

    def __lt__(self, other):
        return self.index < other.index

    def append(self, src_layer: str, src_port: jsonpath.JSONPath, dest_port: jsonpath.JSONPath, idx: int):
        self.connections.append(Connection(src_layer, src_port, dest_port))
        self.index = idx


class Graph(Model):
    def __init__(
            self,
            inputs: Union[Input, list[Input]],
            layers: list[Layer],
            connections: list[tuple[str, str]],
            **kwargs
    ):
        """Initializes the GraphModel with separate inputs, layers, and direct connections.
        Args:
            inputs (Union[Input, List[Input]]): A single Input layer or a list of Input layers, each with a unique name.
            layers (List[Layer]): A list of Layer instances, each with a unique name.
            connections (List[Tuple[str, str]]): A list of tuples specifying the connections between layers,
                                                 represented as (source_layer_name, destination_layer_name).
        """
        print(inputs)
        if not isinstance(inputs, list):
            inputs = [inputs]
        used_input_names = {src for src, _ in connections}
        inputs = [inp for inp in inputs if inp.name in used_input_names]

        super(Graph, self).__init__(
            inputs=inputs,
            outputs=self.__build_model(inputs, layers, connections),
            **kwargs
        )

        self.graph_inputs = inputs
        self.graph_layers = layers
        self.connections = connections

    def __build_model(
            self,
            inputs: List[Input],
            layers: List[Layer],
            connections: list[tuple[str, str]]
    ) -> Any:
        layer_maps: dict[str, Layer] = {layer.name: layer for layer in layers}
        tensor_maps: dict[str, Any] = {inp.name: inp for inp in inputs}
        connection_groups = self.__get_connection_groups(connections)

        for group in connection_groups:
            dest_layer = layer_maps[group.name]
            dest_inputs: dict[str, str] = {}
            for conn in group.connections:
                input_val = ConnectionManager.find_or_fail(tensor_maps[conn.layer], conn.local_key)
                dest_inputs = ConnectionManager.add(dest_inputs, conn.foreign_key, input_val)

            tensor_maps[group.name] = dest_layer(**dest_inputs) if isinstance(dest_inputs, dict) \
                else dest_layer(dest_inputs)
        # Assuming the last group's destination is the model output
        return tensor_maps[connection_groups[-1].name]

    def __get_connection_groups(self, connections: list[tuple[str, str]]) -> list[ConnectionGroup]:
        # Group connections by destination while maintaining the order based on the last appearance
        grouped: dict[str, ConnectionGroup] = {}
        for idx, (src, dest) in enumerate(connections):
            dest_name, dest_port = ConnectionManager.parse_port(dest)
            src_layer, src_port = ConnectionManager.parse_port(src)
            if dest_name not in grouped:
                grouped[dest_name] = ConnectionGroup(dest_name, [], idx)
            grouped[dest_name].append(src_layer, src_port, dest_port, idx)
        return sorted(grouped.values())

    def get_config(self):
        base_config = super(Graph, self).get_config()
        config = {
            **base_config,
            'graph_inputs': [{'shape': inp.shape[1:], 'name': inp.name, 'dtype': inp.dtype} for inp in
                             self.graph_inputs],
            'graph_layers': [keras.layers.serialize(layer) for layer in self.graph_layers],
            'connections': self.connections,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_configs = config['graph_inputs']
        connections = config['connections']

        # Recreate the Input layers from the stored configurations
        inputs = [Input(shape=conf['shape'], name=conf['name'], dtype=conf['dtype']) for conf in input_configs]
        layers = [keras.layers.deserialize(layer_config) for layer_config in config['graph_layers']]
        print(layers)

        # Recreate other layers and connections...
        return cls(inputs=inputs, layers=layers, connections=connections)


class ConnectionManager:
    def __init__(self):
        pass

    @staticmethod
    def parse_port(output: str) -> tuple[str, jsonpath.JSONPath]:
        layer_name, index = ConnectionManager.__parse_connection_string(parse(output))
        return layer_name, jsonpath.Root() if index is None else index

    @staticmethod
    def find_or_fail(data: Any, path: jsonpath.JSONPath) -> Any:
        results = path.find(data)
        if len(results) != 1:
            raise Exception(f"Expected exactly one match for {path}, got {len(results)}")
        return results[0].value

    @staticmethod
    def add(data: Any, path: jsonpath.JSONPath, value: Any) -> Any:
        return path.update_or_create(data, value)

    @staticmethod
    def __parse_connection_string(child: jsonpath.JSONPath) -> tuple[str, Union[jsonpath.JSONPath, None]]:
        if isinstance(child, jsonpath.Fields):
            return ConnectionManager.__get_field_name(child), None
        if isinstance(child, jsonpath.Child):
            layer_name, right = ConnectionManager.__parse_connection_string(child.left)
            return layer_name, child.right if right is None else jsonpath.Child(right, child.right)
        else:
            raise Exception(f"Expected index or key, got ${child}")

    @staticmethod
    def __get_field_name(fields: jsonpath.Fields) -> str:
        if len(fields.fields) != 1:
            raise Exception(f"Expected exactly one field name, got {fields.fields}")
        return str(fields.fields[0])
