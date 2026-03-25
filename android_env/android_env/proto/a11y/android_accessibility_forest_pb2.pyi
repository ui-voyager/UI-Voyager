from android_env.proto.a11y import android_accessibility_window_info_pb2 as _android_accessibility_window_info_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AndroidAccessibilityForest(_message.Message):
    __slots__ = ("windows",)
    WINDOWS_FIELD_NUMBER: _ClassVar[int]
    windows: _containers.RepeatedCompositeFieldContainer[_android_accessibility_window_info_pb2.AndroidAccessibilityWindowInfo]
    def __init__(self, windows: _Optional[_Iterable[_Union[_android_accessibility_window_info_pb2.AndroidAccessibilityWindowInfo, _Mapping]]] = ...) -> None: ...
