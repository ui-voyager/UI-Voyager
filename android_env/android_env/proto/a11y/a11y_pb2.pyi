from android_env.proto.a11y import android_accessibility_forest_pb2 as _android_accessibility_forest_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ForestResponse(_message.Message):
    __slots__ = ("error",)
    ERROR_FIELD_NUMBER: _ClassVar[int]
    error: str
    def __init__(self, error: _Optional[str] = ...) -> None: ...

class EventRequest(_message.Message):
    __slots__ = ("event",)
    class EventEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_FIELD_NUMBER: _ClassVar[int]
    event: _containers.ScalarMap[str, str]
    def __init__(self, event: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EventResponse(_message.Message):
    __slots__ = ("error",)
    ERROR_FIELD_NUMBER: _ClassVar[int]
    error: str
    def __init__(self, error: _Optional[str] = ...) -> None: ...

class ClientToServer(_message.Message):
    __slots__ = ("event", "forest")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    FOREST_FIELD_NUMBER: _ClassVar[int]
    event: EventRequest
    forest: _android_accessibility_forest_pb2.AndroidAccessibilityForest
    def __init__(self, event: _Optional[_Union[EventRequest, _Mapping]] = ..., forest: _Optional[_Union[_android_accessibility_forest_pb2.AndroidAccessibilityForest, _Mapping]] = ...) -> None: ...

class ServerToClient(_message.Message):
    __slots__ = ("get_forest",)
    class GetA11yForest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    GET_FOREST_FIELD_NUMBER: _ClassVar[int]
    get_forest: ServerToClient.GetA11yForest
    def __init__(self, get_forest: _Optional[_Union[ServerToClient.GetA11yForest, _Mapping]] = ...) -> None: ...
