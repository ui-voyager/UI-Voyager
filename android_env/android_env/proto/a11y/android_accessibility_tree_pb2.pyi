from android_env.proto.a11y import android_accessibility_node_info_pb2 as _android_accessibility_node_info_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AndroidAccessibilityTree(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[_android_accessibility_node_info_pb2.AndroidAccessibilityNodeInfo]
    def __init__(self, nodes: _Optional[_Iterable[_Union[_android_accessibility_node_info_pb2.AndroidAccessibilityNodeInfo, _Mapping]]] = ...) -> None: ...
