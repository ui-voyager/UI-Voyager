from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SaveStateRequest(_message.Message):
    __slots__ = ("args",)
    class ArgsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ARGS_FIELD_NUMBER: _ClassVar[int]
    args: _containers.ScalarMap[str, str]
    def __init__(self, args: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LoadStateRequest(_message.Message):
    __slots__ = ("args",)
    class ArgsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ARGS_FIELD_NUMBER: _ClassVar[int]
    args: _containers.ScalarMap[str, str]
    def __init__(self, args: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SaveStateResponse(_message.Message):
    __slots__ = ("status", "error_message", "additional_info")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[SaveStateResponse.Status]
        OK: _ClassVar[SaveStateResponse.Status]
        ERROR: _ClassVar[SaveStateResponse.Status]
    UNDEFINED: SaveStateResponse.Status
    OK: SaveStateResponse.Status
    ERROR: SaveStateResponse.Status
    class AdditionalInfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_INFO_FIELD_NUMBER: _ClassVar[int]
    status: SaveStateResponse.Status
    error_message: str
    additional_info: _containers.ScalarMap[str, str]
    def __init__(self, status: _Optional[_Union[SaveStateResponse.Status, str]] = ..., error_message: _Optional[str] = ..., additional_info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LoadStateResponse(_message.Message):
    __slots__ = ("status", "error_message", "additional_info")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[LoadStateResponse.Status]
        OK: _ClassVar[LoadStateResponse.Status]
        NOT_FOUND: _ClassVar[LoadStateResponse.Status]
        ERROR: _ClassVar[LoadStateResponse.Status]
    UNDEFINED: LoadStateResponse.Status
    OK: LoadStateResponse.Status
    NOT_FOUND: LoadStateResponse.Status
    ERROR: LoadStateResponse.Status
    class AdditionalInfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_INFO_FIELD_NUMBER: _ClassVar[int]
    status: LoadStateResponse.Status
    error_message: str
    additional_info: _containers.ScalarMap[str, str]
    def __init__(self, status: _Optional[_Union[LoadStateResponse.Status, str]] = ..., error_message: _Optional[str] = ..., additional_info: _Optional[_Mapping[str, str]] = ...) -> None: ...
