from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ProtoRect(_message.Message):
    __slots__ = ("left", "top", "right", "bottom")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    left: int
    top: int
    right: int
    bottom: int
    def __init__(self, left: _Optional[int] = ..., top: _Optional[int] = ..., right: _Optional[int] = ..., bottom: _Optional[int] = ...) -> None: ...
