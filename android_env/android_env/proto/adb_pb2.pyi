from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AdbRequest(_message.Message):
    __slots__ = ("install_apk", "start_activity", "force_stop", "tap", "press_button", "start_screen_pinning", "uninstall_package", "get_current_activity", "get_orientation", "push", "pull", "input_text", "settings", "generic", "package_manager", "dumpsys", "send_broadcast", "timeout_sec")
    class InstallApk(_message.Message):
        __slots__ = ("filesystem", "blob")
        class Filesystem(_message.Message):
            __slots__ = ("path",)
            PATH_FIELD_NUMBER: _ClassVar[int]
            path: str
            def __init__(self, path: _Optional[str] = ...) -> None: ...
        class Blob(_message.Message):
            __slots__ = ("contents",)
            CONTENTS_FIELD_NUMBER: _ClassVar[int]
            contents: bytes
            def __init__(self, contents: _Optional[bytes] = ...) -> None: ...
        FILESYSTEM_FIELD_NUMBER: _ClassVar[int]
        BLOB_FIELD_NUMBER: _ClassVar[int]
        filesystem: AdbRequest.InstallApk.Filesystem
        blob: AdbRequest.InstallApk.Blob
        def __init__(self, filesystem: _Optional[_Union[AdbRequest.InstallApk.Filesystem, _Mapping]] = ..., blob: _Optional[_Union[AdbRequest.InstallApk.Blob, _Mapping]] = ...) -> None: ...
    class StartActivity(_message.Message):
        __slots__ = ("full_activity", "extra_args", "force_stop")
        FULL_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
        EXTRA_ARGS_FIELD_NUMBER: _ClassVar[int]
        FORCE_STOP_FIELD_NUMBER: _ClassVar[int]
        full_activity: str
        extra_args: _containers.RepeatedScalarFieldContainer[str]
        force_stop: bool
        def __init__(self, full_activity: _Optional[str] = ..., extra_args: _Optional[_Iterable[str]] = ..., force_stop: bool = ...) -> None: ...
    class SendBroadcast(_message.Message):
        __slots__ = ("action", "component")
        ACTION_FIELD_NUMBER: _ClassVar[int]
        COMPONENT_FIELD_NUMBER: _ClassVar[int]
        action: str
        component: str
        def __init__(self, action: _Optional[str] = ..., component: _Optional[str] = ...) -> None: ...
    class UninstallPackage(_message.Message):
        __slots__ = ("package_name",)
        PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
        package_name: str
        def __init__(self, package_name: _Optional[str] = ...) -> None: ...
    class ForceStop(_message.Message):
        __slots__ = ("package_name",)
        PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
        package_name: str
        def __init__(self, package_name: _Optional[str] = ...) -> None: ...
    class Tap(_message.Message):
        __slots__ = ("x", "y")
        X_FIELD_NUMBER: _ClassVar[int]
        Y_FIELD_NUMBER: _ClassVar[int]
        x: int
        y: int
        def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...
    class PressButton(_message.Message):
        __slots__ = ("button",)
        class Button(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            HOME: _ClassVar[AdbRequest.PressButton.Button]
            BACK: _ClassVar[AdbRequest.PressButton.Button]
            ENTER: _ClassVar[AdbRequest.PressButton.Button]
        HOME: AdbRequest.PressButton.Button
        BACK: AdbRequest.PressButton.Button
        ENTER: AdbRequest.PressButton.Button
        BUTTON_FIELD_NUMBER: _ClassVar[int]
        button: AdbRequest.PressButton.Button
        def __init__(self, button: _Optional[_Union[AdbRequest.PressButton.Button, str]] = ...) -> None: ...
    class StartScreenPinning(_message.Message):
        __slots__ = ("full_activity",)
        FULL_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
        full_activity: str
        def __init__(self, full_activity: _Optional[str] = ...) -> None: ...
    class GetCurrentActivity(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetOrientationRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Push(_message.Message):
        __slots__ = ("content", "path")
        CONTENT_FIELD_NUMBER: _ClassVar[int]
        PATH_FIELD_NUMBER: _ClassVar[int]
        content: bytes
        path: str
        def __init__(self, content: _Optional[bytes] = ..., path: _Optional[str] = ...) -> None: ...
    class Pull(_message.Message):
        __slots__ = ("path",)
        PATH_FIELD_NUMBER: _ClassVar[int]
        path: str
        def __init__(self, path: _Optional[str] = ...) -> None: ...
    class InputText(_message.Message):
        __slots__ = ("text",)
        TEXT_FIELD_NUMBER: _ClassVar[int]
        text: str
        def __init__(self, text: _Optional[str] = ...) -> None: ...
    class SettingsRequest(_message.Message):
        __slots__ = ("name_space", "get", "put", "delete_key", "reset", "list")
        class Namespace(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            UNKNOWN: _ClassVar[AdbRequest.SettingsRequest.Namespace]
            SYSTEM: _ClassVar[AdbRequest.SettingsRequest.Namespace]
            SECURE: _ClassVar[AdbRequest.SettingsRequest.Namespace]
            GLOBAL: _ClassVar[AdbRequest.SettingsRequest.Namespace]
        UNKNOWN: AdbRequest.SettingsRequest.Namespace
        SYSTEM: AdbRequest.SettingsRequest.Namespace
        SECURE: AdbRequest.SettingsRequest.Namespace
        GLOBAL: AdbRequest.SettingsRequest.Namespace
        class Get(_message.Message):
            __slots__ = ("key",)
            KEY_FIELD_NUMBER: _ClassVar[int]
            key: str
            def __init__(self, key: _Optional[str] = ...) -> None: ...
        class Put(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        class Delete(_message.Message):
            __slots__ = ("key",)
            KEY_FIELD_NUMBER: _ClassVar[int]
            key: str
            def __init__(self, key: _Optional[str] = ...) -> None: ...
        class Reset(_message.Message):
            __slots__ = ("package_name", "mode")
            class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = ()
                UNKNOWN: _ClassVar[AdbRequest.SettingsRequest.Reset.Mode]
                UNTRUSTED_DEFAULTS: _ClassVar[AdbRequest.SettingsRequest.Reset.Mode]
                UNTRUSTED_CLEAR: _ClassVar[AdbRequest.SettingsRequest.Reset.Mode]
                TRUSTED_DEFAULTS: _ClassVar[AdbRequest.SettingsRequest.Reset.Mode]
            UNKNOWN: AdbRequest.SettingsRequest.Reset.Mode
            UNTRUSTED_DEFAULTS: AdbRequest.SettingsRequest.Reset.Mode
            UNTRUSTED_CLEAR: AdbRequest.SettingsRequest.Reset.Mode
            TRUSTED_DEFAULTS: AdbRequest.SettingsRequest.Reset.Mode
            PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
            MODE_FIELD_NUMBER: _ClassVar[int]
            package_name: str
            mode: AdbRequest.SettingsRequest.Reset.Mode
            def __init__(self, package_name: _Optional[str] = ..., mode: _Optional[_Union[AdbRequest.SettingsRequest.Reset.Mode, str]] = ...) -> None: ...
        class List(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        NAME_SPACE_FIELD_NUMBER: _ClassVar[int]
        GET_FIELD_NUMBER: _ClassVar[int]
        PUT_FIELD_NUMBER: _ClassVar[int]
        DELETE_KEY_FIELD_NUMBER: _ClassVar[int]
        RESET_FIELD_NUMBER: _ClassVar[int]
        LIST_FIELD_NUMBER: _ClassVar[int]
        name_space: AdbRequest.SettingsRequest.Namespace
        get: AdbRequest.SettingsRequest.Get
        put: AdbRequest.SettingsRequest.Put
        delete_key: AdbRequest.SettingsRequest.Delete
        reset: AdbRequest.SettingsRequest.Reset
        list: AdbRequest.SettingsRequest.List
        def __init__(self, name_space: _Optional[_Union[AdbRequest.SettingsRequest.Namespace, str]] = ..., get: _Optional[_Union[AdbRequest.SettingsRequest.Get, _Mapping]] = ..., put: _Optional[_Union[AdbRequest.SettingsRequest.Put, _Mapping]] = ..., delete_key: _Optional[_Union[AdbRequest.SettingsRequest.Delete, _Mapping]] = ..., reset: _Optional[_Union[AdbRequest.SettingsRequest.Reset, _Mapping]] = ..., list: _Optional[_Union[AdbRequest.SettingsRequest.List, _Mapping]] = ...) -> None: ...
    class GenericRequest(_message.Message):
        __slots__ = ("args",)
        ARGS_FIELD_NUMBER: _ClassVar[int]
        args: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, args: _Optional[_Iterable[str]] = ...) -> None: ...
    class PackageManagerRequest(_message.Message):
        __slots__ = ("list", "clear", "grant")
        class List(_message.Message):
            __slots__ = ("features", "libraries", "packages")
            class Features(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class Libraries(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class Packages(_message.Message):
                __slots__ = ("filter", "options")
                FILTER_FIELD_NUMBER: _ClassVar[int]
                OPTIONS_FIELD_NUMBER: _ClassVar[int]
                filter: str
                options: _containers.RepeatedScalarFieldContainer[str]
                def __init__(self, filter: _Optional[str] = ..., options: _Optional[_Iterable[str]] = ...) -> None: ...
            FEATURES_FIELD_NUMBER: _ClassVar[int]
            LIBRARIES_FIELD_NUMBER: _ClassVar[int]
            PACKAGES_FIELD_NUMBER: _ClassVar[int]
            features: AdbRequest.PackageManagerRequest.List.Features
            libraries: AdbRequest.PackageManagerRequest.List.Libraries
            packages: AdbRequest.PackageManagerRequest.List.Packages
            def __init__(self, features: _Optional[_Union[AdbRequest.PackageManagerRequest.List.Features, _Mapping]] = ..., libraries: _Optional[_Union[AdbRequest.PackageManagerRequest.List.Libraries, _Mapping]] = ..., packages: _Optional[_Union[AdbRequest.PackageManagerRequest.List.Packages, _Mapping]] = ...) -> None: ...
        class Clear(_message.Message):
            __slots__ = ("package_name", "user_id")
            PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
            USER_ID_FIELD_NUMBER: _ClassVar[int]
            package_name: str
            user_id: str
            def __init__(self, package_name: _Optional[str] = ..., user_id: _Optional[str] = ...) -> None: ...
        class Grant(_message.Message):
            __slots__ = ("package_name", "permissions")
            PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
            PERMISSIONS_FIELD_NUMBER: _ClassVar[int]
            package_name: str
            permissions: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, package_name: _Optional[str] = ..., permissions: _Optional[_Iterable[str]] = ...) -> None: ...
        LIST_FIELD_NUMBER: _ClassVar[int]
        CLEAR_FIELD_NUMBER: _ClassVar[int]
        GRANT_FIELD_NUMBER: _ClassVar[int]
        list: AdbRequest.PackageManagerRequest.List
        clear: AdbRequest.PackageManagerRequest.Clear
        grant: AdbRequest.PackageManagerRequest.Grant
        def __init__(self, list: _Optional[_Union[AdbRequest.PackageManagerRequest.List, _Mapping]] = ..., clear: _Optional[_Union[AdbRequest.PackageManagerRequest.Clear, _Mapping]] = ..., grant: _Optional[_Union[AdbRequest.PackageManagerRequest.Grant, _Mapping]] = ...) -> None: ...
    class DumpsysRequest(_message.Message):
        __slots__ = ("service", "args", "list_only", "timeout_sec", "timeout_ms", "pid", "proto", "priority", "skip_services")
        class PriorityLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            UNSET: _ClassVar[AdbRequest.DumpsysRequest.PriorityLevel]
            NORMAL: _ClassVar[AdbRequest.DumpsysRequest.PriorityLevel]
            HIGH: _ClassVar[AdbRequest.DumpsysRequest.PriorityLevel]
            CRITICAL: _ClassVar[AdbRequest.DumpsysRequest.PriorityLevel]
        UNSET: AdbRequest.DumpsysRequest.PriorityLevel
        NORMAL: AdbRequest.DumpsysRequest.PriorityLevel
        HIGH: AdbRequest.DumpsysRequest.PriorityLevel
        CRITICAL: AdbRequest.DumpsysRequest.PriorityLevel
        SERVICE_FIELD_NUMBER: _ClassVar[int]
        ARGS_FIELD_NUMBER: _ClassVar[int]
        LIST_ONLY_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
        PID_FIELD_NUMBER: _ClassVar[int]
        PROTO_FIELD_NUMBER: _ClassVar[int]
        PRIORITY_FIELD_NUMBER: _ClassVar[int]
        SKIP_SERVICES_FIELD_NUMBER: _ClassVar[int]
        service: str
        args: _containers.RepeatedScalarFieldContainer[str]
        list_only: bool
        timeout_sec: int
        timeout_ms: int
        pid: bool
        proto: bool
        priority: AdbRequest.DumpsysRequest.PriorityLevel
        skip_services: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, service: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., list_only: bool = ..., timeout_sec: _Optional[int] = ..., timeout_ms: _Optional[int] = ..., pid: bool = ..., proto: bool = ..., priority: _Optional[_Union[AdbRequest.DumpsysRequest.PriorityLevel, str]] = ..., skip_services: _Optional[_Iterable[str]] = ...) -> None: ...
    INSTALL_APK_FIELD_NUMBER: _ClassVar[int]
    START_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    FORCE_STOP_FIELD_NUMBER: _ClassVar[int]
    TAP_FIELD_NUMBER: _ClassVar[int]
    PRESS_BUTTON_FIELD_NUMBER: _ClassVar[int]
    START_SCREEN_PINNING_FIELD_NUMBER: _ClassVar[int]
    UNINSTALL_PACKAGE_FIELD_NUMBER: _ClassVar[int]
    GET_CURRENT_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    GET_ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    PUSH_FIELD_NUMBER: _ClassVar[int]
    PULL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    GENERIC_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_MANAGER_FIELD_NUMBER: _ClassVar[int]
    DUMPSYS_FIELD_NUMBER: _ClassVar[int]
    SEND_BROADCAST_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
    install_apk: AdbRequest.InstallApk
    start_activity: AdbRequest.StartActivity
    force_stop: AdbRequest.ForceStop
    tap: AdbRequest.Tap
    press_button: AdbRequest.PressButton
    start_screen_pinning: AdbRequest.StartScreenPinning
    uninstall_package: AdbRequest.UninstallPackage
    get_current_activity: AdbRequest.GetCurrentActivity
    get_orientation: AdbRequest.GetOrientationRequest
    push: AdbRequest.Push
    pull: AdbRequest.Pull
    input_text: AdbRequest.InputText
    settings: AdbRequest.SettingsRequest
    generic: AdbRequest.GenericRequest
    package_manager: AdbRequest.PackageManagerRequest
    dumpsys: AdbRequest.DumpsysRequest
    send_broadcast: AdbRequest.SendBroadcast
    timeout_sec: float
    def __init__(self, install_apk: _Optional[_Union[AdbRequest.InstallApk, _Mapping]] = ..., start_activity: _Optional[_Union[AdbRequest.StartActivity, _Mapping]] = ..., force_stop: _Optional[_Union[AdbRequest.ForceStop, _Mapping]] = ..., tap: _Optional[_Union[AdbRequest.Tap, _Mapping]] = ..., press_button: _Optional[_Union[AdbRequest.PressButton, _Mapping]] = ..., start_screen_pinning: _Optional[_Union[AdbRequest.StartScreenPinning, _Mapping]] = ..., uninstall_package: _Optional[_Union[AdbRequest.UninstallPackage, _Mapping]] = ..., get_current_activity: _Optional[_Union[AdbRequest.GetCurrentActivity, _Mapping]] = ..., get_orientation: _Optional[_Union[AdbRequest.GetOrientationRequest, _Mapping]] = ..., push: _Optional[_Union[AdbRequest.Push, _Mapping]] = ..., pull: _Optional[_Union[AdbRequest.Pull, _Mapping]] = ..., input_text: _Optional[_Union[AdbRequest.InputText, _Mapping]] = ..., settings: _Optional[_Union[AdbRequest.SettingsRequest, _Mapping]] = ..., generic: _Optional[_Union[AdbRequest.GenericRequest, _Mapping]] = ..., package_manager: _Optional[_Union[AdbRequest.PackageManagerRequest, _Mapping]] = ..., dumpsys: _Optional[_Union[AdbRequest.DumpsysRequest, _Mapping]] = ..., send_broadcast: _Optional[_Union[AdbRequest.SendBroadcast, _Mapping]] = ..., timeout_sec: _Optional[float] = ...) -> None: ...

class AdbResponse(_message.Message):
    __slots__ = ("status", "error_message", "stats", "get_current_activity", "start_activity", "press_button", "push", "pull", "input_text", "settings", "generic", "package_manager", "get_orientation", "dumpsys")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[AdbResponse.Status]
        OK: _ClassVar[AdbResponse.Status]
        UNKNOWN_COMMAND: _ClassVar[AdbResponse.Status]
        FAILED_PRECONDITION: _ClassVar[AdbResponse.Status]
        INTERNAL_ERROR: _ClassVar[AdbResponse.Status]
        ADB_ERROR: _ClassVar[AdbResponse.Status]
        TIMEOUT: _ClassVar[AdbResponse.Status]
    UNDEFINED: AdbResponse.Status
    OK: AdbResponse.Status
    UNKNOWN_COMMAND: AdbResponse.Status
    FAILED_PRECONDITION: AdbResponse.Status
    INTERNAL_ERROR: AdbResponse.Status
    ADB_ERROR: AdbResponse.Status
    TIMEOUT: AdbResponse.Status
    class StatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class GetCurrentActivityResponse(_message.Message):
        __slots__ = ("full_activity",)
        FULL_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
        full_activity: str
        def __init__(self, full_activity: _Optional[str] = ...) -> None: ...
    class GetOrientationResponse(_message.Message):
        __slots__ = ("orientation",)
        ORIENTATION_FIELD_NUMBER: _ClassVar[int]
        orientation: int
        def __init__(self, orientation: _Optional[int] = ...) -> None: ...
    class StartActivityResponse(_message.Message):
        __slots__ = ("full_activity", "output")
        FULL_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        full_activity: str
        output: bytes
        def __init__(self, full_activity: _Optional[str] = ..., output: _Optional[bytes] = ...) -> None: ...
    class PressButtonResponse(_message.Message):
        __slots__ = ("output",)
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        output: bytes
        def __init__(self, output: _Optional[bytes] = ...) -> None: ...
    class PushResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class PullResponse(_message.Message):
        __slots__ = ("content",)
        CONTENT_FIELD_NUMBER: _ClassVar[int]
        content: bytes
        def __init__(self, content: _Optional[bytes] = ...) -> None: ...
    class InputTextResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class SettingsResponse(_message.Message):
        __slots__ = ("output",)
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        output: bytes
        def __init__(self, output: _Optional[bytes] = ...) -> None: ...
    class GenericResponse(_message.Message):
        __slots__ = ("output",)
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        output: bytes
        def __init__(self, output: _Optional[bytes] = ...) -> None: ...
    class PackageManagerResponse(_message.Message):
        __slots__ = ("output", "list")
        class List(_message.Message):
            __slots__ = ("items",)
            ITEMS_FIELD_NUMBER: _ClassVar[int]
            items: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, items: _Optional[_Iterable[str]] = ...) -> None: ...
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        LIST_FIELD_NUMBER: _ClassVar[int]
        output: bytes
        list: AdbResponse.PackageManagerResponse.List
        def __init__(self, output: _Optional[bytes] = ..., list: _Optional[_Union[AdbResponse.PackageManagerResponse.List, _Mapping]] = ...) -> None: ...
    class DumpsysResponse(_message.Message):
        __slots__ = ("output",)
        OUTPUT_FIELD_NUMBER: _ClassVar[int]
        output: bytes
        def __init__(self, output: _Optional[bytes] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    GET_CURRENT_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    START_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    PRESS_BUTTON_FIELD_NUMBER: _ClassVar[int]
    PUSH_FIELD_NUMBER: _ClassVar[int]
    PULL_FIELD_NUMBER: _ClassVar[int]
    INPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    GENERIC_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_MANAGER_FIELD_NUMBER: _ClassVar[int]
    GET_ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    DUMPSYS_FIELD_NUMBER: _ClassVar[int]
    status: AdbResponse.Status
    error_message: str
    stats: _containers.ScalarMap[str, float]
    get_current_activity: AdbResponse.GetCurrentActivityResponse
    start_activity: AdbResponse.StartActivityResponse
    press_button: AdbResponse.PressButtonResponse
    push: AdbResponse.PushResponse
    pull: AdbResponse.PullResponse
    input_text: AdbResponse.InputTextResponse
    settings: AdbResponse.SettingsResponse
    generic: AdbResponse.GenericResponse
    package_manager: AdbResponse.PackageManagerResponse
    get_orientation: AdbResponse.GetOrientationResponse
    dumpsys: AdbResponse.DumpsysResponse
    def __init__(self, status: _Optional[_Union[AdbResponse.Status, str]] = ..., error_message: _Optional[str] = ..., stats: _Optional[_Mapping[str, float]] = ..., get_current_activity: _Optional[_Union[AdbResponse.GetCurrentActivityResponse, _Mapping]] = ..., start_activity: _Optional[_Union[AdbResponse.StartActivityResponse, _Mapping]] = ..., press_button: _Optional[_Union[AdbResponse.PressButtonResponse, _Mapping]] = ..., push: _Optional[_Union[AdbResponse.PushResponse, _Mapping]] = ..., pull: _Optional[_Union[AdbResponse.PullResponse, _Mapping]] = ..., input_text: _Optional[_Union[AdbResponse.InputTextResponse, _Mapping]] = ..., settings: _Optional[_Union[AdbResponse.SettingsResponse, _Mapping]] = ..., generic: _Optional[_Union[AdbResponse.GenericResponse, _Mapping]] = ..., package_manager: _Optional[_Union[AdbResponse.PackageManagerResponse, _Mapping]] = ..., get_orientation: _Optional[_Union[AdbResponse.GetOrientationResponse, _Mapping]] = ..., dumpsys: _Optional[_Union[AdbResponse.DumpsysResponse, _Mapping]] = ...) -> None: ...
