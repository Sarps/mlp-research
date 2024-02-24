from dataclasses import dataclass
from typing import Any, Union, List, Tuple, NamedTuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer


@dataclass
class Connection:
    src: Union[Tuple[str, int], str]
    dest: Union[Tuple[str, str], str]


@dataclass
class ConnectionGroup:
    name: str
    connections: List[Connection]
    index: int

    def __lt__(self, other):
        return self.index < other.index


class Graph(Model):
    def __init__(self, inputs: Union[Input, List[Input]], layers: List[Layer], connections: List[Connection], **kwargs):
        """Initializes the GraphModel with separate inputs, layers, and direct connections.

        Args:
            inputs (Union[Input, List[Input]]): A single Input layer or a list of Input layers, each with a unique name.
            layers (List[Layer]): A list of Layer instances, each with a unique name.
            connections (List[Tuple[str, str]]): A list of tuples specifying the connections between layers,
                                                 represented as (source_layer_name, destination_layer_name).
        """
        if not isinstance(inputs, list):
            inputs = [inputs]  # Ensure inputs is a list

        # Determine which inputs are used based on connections
        used_input_names = {conn.src for conn in connections}
        used_inputs = [inp for inp in inputs if inp.name in used_input_names]

        outputs = self.__build_model(used_inputs, layers, connections)
        super(Graph, self).__init__(inputs=used_inputs, outputs=outputs, **kwargs)

    def __build_model(self, inputs: List[Input], layers: List[Layer], connections: List[Connection]) -> Any:
        """Builds the model based on the specified inputs, layers, and connections."""
        layer_maps: dict[str, Layer] = {layer.name: layer for layer in layers}
        tensor_maps: dict[str, Any] = {inp.name: inp for inp in inputs}

        connection_groups = self.__get_connection_groups(connections)

        for group in connection_groups:
            dest_layer = layer_maps[group.name]
            src_outputs: List[tuple[str, Any]] = []
            for conn in group.connections:
                arg_name = conn.dest[1] if isinstance(conn.dest, (tuple, list)) else 0
                if isinstance(conn.src, str):
                    src_outputs.append((arg_name, tensor_maps[conn.src]))
                elif isinstance(conn.src, tuple):
                    src_name, src_index = conn.src
                    src_outputs.append((arg_name, tensor_maps[src_name][src_index]))

            if len(src_outputs) > 1:
                tensor_maps[group.name] = dest_layer(**dict(src_outputs))
            else:
                tensor_maps[group.name] = dest_layer(src_outputs[0][1])

        # Assuming the last group's destination is the model output
        return tensor_maps[connection_groups[-1].name]

    def __get_connection_groups(self, connections: List[Connection]) -> list[ConnectionGroup]:
        # Group connections by destination while maintaining the order based on the last appearance
        grouped: dict[str, ConnectionGroup] = {}
        for idx, connection in enumerate(connections):
            dest_name = connection.dest[0] if isinstance(connection.dest, (tuple, list)) else connection.dest
            if dest_name not in grouped:
                grouped[dest_name] = ConnectionGroup(dest_name, [], idx)
            grouped[dest_name].connections.append(connection)
            grouped[dest_name].index = idx
        return sorted(grouped.values())
