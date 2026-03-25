from android_env.proto import adb_pb2 as _adb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AppScreen(_message.Message):
    __slots__ = ("activity", "view_hierarchy_path")
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    VIEW_HIERARCHY_PATH_FIELD_NUMBER: _ClassVar[int]
    activity: str
    view_hierarchy_path: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, activity: _Optional[str] = ..., view_hierarchy_path: _Optional[_Iterable[str]] = ...) -> None: ...

class WaitForAppScreen(_message.Message):
    __slots__ = ("app_screen", "timeout_sec")
    APP_SCREEN_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
    app_screen: AppScreen
    timeout_sec: float
    def __init__(self, app_screen: _Optional[_Union[AppScreen, _Mapping]] = ..., timeout_sec: _Optional[float] = ...) -> None: ...

class CheckInstall(_message.Message):
    __slots__ = ("package_name", "timeout_sec")
    PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
    package_name: str
    timeout_sec: float
    def __init__(self, package_name: _Optional[str] = ..., timeout_sec: _Optional[float] = ...) -> None: ...

class Sleep(_message.Message):
    __slots__ = ("time_sec",)
    TIME_SEC_FIELD_NUMBER: _ClassVar[int]
    time_sec: float
    def __init__(self, time_sec: _Optional[float] = ...) -> None: ...

class SuccessCondition(_message.Message):
    __slots__ = ("num_retries", "wait_for_app_screen", "check_install")
    NUM_RETRIES_FIELD_NUMBER: _ClassVar[int]
    WAIT_FOR_APP_SCREEN_FIELD_NUMBER: _ClassVar[int]
    CHECK_INSTALL_FIELD_NUMBER: _ClassVar[int]
    num_retries: int
    wait_for_app_screen: WaitForAppScreen
    check_install: CheckInstall
    def __init__(self, num_retries: _Optional[int] = ..., wait_for_app_screen: _Optional[_Union[WaitForAppScreen, _Mapping]] = ..., check_install: _Optional[_Union[CheckInstall, _Mapping]] = ...) -> None: ...

class SetupStep(_message.Message):
    __slots__ = ("success_condition", "adb_request", "sleep")
    SUCCESS_CONDITION_FIELD_NUMBER: _ClassVar[int]
    ADB_REQUEST_FIELD_NUMBER: _ClassVar[int]
    SLEEP_FIELD_NUMBER: _ClassVar[int]
    success_condition: SuccessCondition
    adb_request: _adb_pb2.AdbRequest
    sleep: Sleep
    def __init__(self, success_condition: _Optional[_Union[SuccessCondition, _Mapping]] = ..., adb_request: _Optional[_Union[_adb_pb2.AdbRequest, _Mapping]] = ..., sleep: _Optional[_Union[Sleep, _Mapping]] = ...) -> None: ...

class ArraySpec(_message.Message):
    __slots__ = ("name", "shape", "dtype")
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        INVALID_DATA_TYPE: _ClassVar[ArraySpec.DataType]
        FLOAT: _ClassVar[ArraySpec.DataType]
        DOUBLE: _ClassVar[ArraySpec.DataType]
        INT8: _ClassVar[ArraySpec.DataType]
        INT16: _ClassVar[ArraySpec.DataType]
        INT32: _ClassVar[ArraySpec.DataType]
        INT64: _ClassVar[ArraySpec.DataType]
        UINT8: _ClassVar[ArraySpec.DataType]
        UINT16: _ClassVar[ArraySpec.DataType]
        UINT32: _ClassVar[ArraySpec.DataType]
        UINT64: _ClassVar[ArraySpec.DataType]
        BOOL: _ClassVar[ArraySpec.DataType]
        STRING_U1: _ClassVar[ArraySpec.DataType]
        STRING_U16: _ClassVar[ArraySpec.DataType]
        STRING_U25: _ClassVar[ArraySpec.DataType]
        STRING_U250: _ClassVar[ArraySpec.DataType]
        STRING: _ClassVar[ArraySpec.DataType]
        OBJECT: _ClassVar[ArraySpec.DataType]
    INVALID_DATA_TYPE: ArraySpec.DataType
    FLOAT: ArraySpec.DataType
    DOUBLE: ArraySpec.DataType
    INT8: ArraySpec.DataType
    INT16: ArraySpec.DataType
    INT32: ArraySpec.DataType
    INT64: ArraySpec.DataType
    UINT8: ArraySpec.DataType
    UINT16: ArraySpec.DataType
    UINT32: ArraySpec.DataType
    UINT64: ArraySpec.DataType
    BOOL: ArraySpec.DataType
    STRING_U1: ArraySpec.DataType
    STRING_U16: ArraySpec.DataType
    STRING_U25: ArraySpec.DataType
    STRING_U250: ArraySpec.DataType
    STRING: ArraySpec.DataType
    OBJECT: ArraySpec.DataType
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: ArraySpec.DataType
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[ArraySpec.DataType, str]] = ...) -> None: ...

class LogParsingConfig(_message.Message):
    __slots__ = ("filters", "log_regexps")
    class LogRegexps(_message.Message):
        __slots__ = ("score", "reward", "episode_end", "extra", "json_extra", "reward_event")
        class RewardEvent(_message.Message):
            __slots__ = ("event", "reward")
            EVENT_FIELD_NUMBER: _ClassVar[int]
            REWARD_FIELD_NUMBER: _ClassVar[int]
            event: str
            reward: float
            def __init__(self, event: _Optional[str] = ..., reward: _Optional[float] = ...) -> None: ...
        SCORE_FIELD_NUMBER: _ClassVar[int]
        REWARD_FIELD_NUMBER: _ClassVar[int]
        EPISODE_END_FIELD_NUMBER: _ClassVar[int]
        EXTRA_FIELD_NUMBER: _ClassVar[int]
        JSON_EXTRA_FIELD_NUMBER: _ClassVar[int]
        REWARD_EVENT_FIELD_NUMBER: _ClassVar[int]
        score: str
        reward: _containers.RepeatedScalarFieldContainer[str]
        episode_end: _containers.RepeatedScalarFieldContainer[str]
        extra: _containers.RepeatedScalarFieldContainer[str]
        json_extra: _containers.RepeatedScalarFieldContainer[str]
        reward_event: _containers.RepeatedCompositeFieldContainer[LogParsingConfig.LogRegexps.RewardEvent]
        def __init__(self, score: _Optional[str] = ..., reward: _Optional[_Iterable[str]] = ..., episode_end: _Optional[_Iterable[str]] = ..., extra: _Optional[_Iterable[str]] = ..., json_extra: _Optional[_Iterable[str]] = ..., reward_event: _Optional[_Iterable[_Union[LogParsingConfig.LogRegexps.RewardEvent, _Mapping]]] = ...) -> None: ...
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    LOG_REGEXPS_FIELD_NUMBER: _ClassVar[int]
    filters: _containers.RepeatedScalarFieldContainer[str]
    log_regexps: LogParsingConfig.LogRegexps
    def __init__(self, filters: _Optional[_Iterable[str]] = ..., log_regexps: _Optional[_Union[LogParsingConfig.LogRegexps, _Mapping]] = ...) -> None: ...

class Task(_message.Message):
    __slots__ = ("id", "name", "description", "setup_steps", "reset_steps", "expected_app_screen", "max_episode_sec", "max_episode_steps", "log_parsing_config", "extras_spec")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SETUP_STEPS_FIELD_NUMBER: _ClassVar[int]
    RESET_STEPS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_APP_SCREEN_FIELD_NUMBER: _ClassVar[int]
    MAX_EPISODE_SEC_FIELD_NUMBER: _ClassVar[int]
    MAX_EPISODE_STEPS_FIELD_NUMBER: _ClassVar[int]
    LOG_PARSING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_SPEC_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    setup_steps: _containers.RepeatedCompositeFieldContainer[SetupStep]
    reset_steps: _containers.RepeatedCompositeFieldContainer[SetupStep]
    expected_app_screen: AppScreen
    max_episode_sec: float
    max_episode_steps: int
    log_parsing_config: LogParsingConfig
    extras_spec: _containers.RepeatedCompositeFieldContainer[ArraySpec]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., setup_steps: _Optional[_Iterable[_Union[SetupStep, _Mapping]]] = ..., reset_steps: _Optional[_Iterable[_Union[SetupStep, _Mapping]]] = ..., expected_app_screen: _Optional[_Union[AppScreen, _Mapping]] = ..., max_episode_sec: _Optional[float] = ..., max_episode_steps: _Optional[int] = ..., log_parsing_config: _Optional[_Union[LogParsingConfig, _Mapping]] = ..., extras_spec: _Optional[_Iterable[_Union[ArraySpec, _Mapping]]] = ...) -> None: ...
