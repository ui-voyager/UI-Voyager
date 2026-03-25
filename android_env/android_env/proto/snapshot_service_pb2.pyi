from android_env.proto import snapshot_pb2 as _snapshot_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SnapshotPackage(_message.Message):
    __slots__ = ("snapshot_id", "payload", "success", "err", "format", "path")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TARGZ: _ClassVar[SnapshotPackage.Format]
        TAR: _ClassVar[SnapshotPackage.Format]
        DIRECTORY: _ClassVar[SnapshotPackage.Format]
    TARGZ: SnapshotPackage.Format
    TAR: SnapshotPackage.Format
    DIRECTORY: SnapshotPackage.Format
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    snapshot_id: str
    payload: bytes
    success: bool
    err: bytes
    format: SnapshotPackage.Format
    path: str
    def __init__(self, snapshot_id: _Optional[str] = ..., payload: _Optional[bytes] = ..., success: bool = ..., err: _Optional[bytes] = ..., format: _Optional[_Union[SnapshotPackage.Format, str]] = ..., path: _Optional[str] = ...) -> None: ...

class SnapshotFilter(_message.Message):
    __slots__ = ("statusFilter",)
    class LoadStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CompatibleOnly: _ClassVar[SnapshotFilter.LoadStatus]
        All: _ClassVar[SnapshotFilter.LoadStatus]
    CompatibleOnly: SnapshotFilter.LoadStatus
    All: SnapshotFilter.LoadStatus
    STATUSFILTER_FIELD_NUMBER: _ClassVar[int]
    statusFilter: SnapshotFilter.LoadStatus
    def __init__(self, statusFilter: _Optional[_Union[SnapshotFilter.LoadStatus, str]] = ...) -> None: ...

class SnapshotDetails(_message.Message):
    __slots__ = ("snapshot_id", "details", "status", "size")
    class LoadStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Compatible: _ClassVar[SnapshotDetails.LoadStatus]
        Incompatible: _ClassVar[SnapshotDetails.LoadStatus]
        Loaded: _ClassVar[SnapshotDetails.LoadStatus]
    Compatible: SnapshotDetails.LoadStatus
    Incompatible: SnapshotDetails.LoadStatus
    Loaded: SnapshotDetails.LoadStatus
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    snapshot_id: str
    details: _snapshot_pb2.Snapshot
    status: SnapshotDetails.LoadStatus
    size: int
    def __init__(self, snapshot_id: _Optional[str] = ..., details: _Optional[_Union[_snapshot_pb2.Snapshot, _Mapping]] = ..., status: _Optional[_Union[SnapshotDetails.LoadStatus, str]] = ..., size: _Optional[int] = ...) -> None: ...

class SnapshotList(_message.Message):
    __slots__ = ("snapshots",)
    SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    snapshots: _containers.RepeatedCompositeFieldContainer[SnapshotDetails]
    def __init__(self, snapshots: _Optional[_Iterable[_Union[SnapshotDetails, _Mapping]]] = ...) -> None: ...

class IceboxTarget(_message.Message):
    __slots__ = ("pid", "package_name", "snapshot_id", "failed", "err", "max_snapshot_number")
    PID_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    ERR_FIELD_NUMBER: _ClassVar[int]
    MAX_SNAPSHOT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    pid: int
    package_name: str
    snapshot_id: str
    failed: bool
    err: str
    max_snapshot_number: int
    def __init__(self, pid: _Optional[int] = ..., package_name: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., failed: bool = ..., err: _Optional[str] = ..., max_snapshot_number: _Optional[int] = ...) -> None: ...
