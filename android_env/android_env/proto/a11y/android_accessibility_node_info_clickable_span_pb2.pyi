from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AndroidAccessibilityNodeInfoClickableSpan(_message.Message):
    __slots__ = ("text", "url", "source", "start", "node_id")
    class SpanSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN_TYPE: _ClassVar[AndroidAccessibilityNodeInfoClickableSpan.SpanSource]
        TEXT: _ClassVar[AndroidAccessibilityNodeInfoClickableSpan.SpanSource]
        CONTENT_DESCRIPTION: _ClassVar[AndroidAccessibilityNodeInfoClickableSpan.SpanSource]
    UNKNOWN_TYPE: AndroidAccessibilityNodeInfoClickableSpan.SpanSource
    TEXT: AndroidAccessibilityNodeInfoClickableSpan.SpanSource
    CONTENT_DESCRIPTION: AndroidAccessibilityNodeInfoClickableSpan.SpanSource
    TEXT_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    text: str
    url: str
    source: AndroidAccessibilityNodeInfoClickableSpan.SpanSource
    start: int
    node_id: int
    def __init__(self, text: _Optional[str] = ..., url: _Optional[str] = ..., source: _Optional[_Union[AndroidAccessibilityNodeInfoClickableSpan.SpanSource, str]] = ..., start: _Optional[int] = ..., node_id: _Optional[int] = ...) -> None: ...
