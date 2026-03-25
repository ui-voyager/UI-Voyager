from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Image(_message.Message):
    __slots__ = ("type", "path", "present", "size", "modification_time")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        IMAGE_TYPE_UNKNOWN: _ClassVar[Image.Type]
        IMAGE_TYPE_KERNEL: _ClassVar[Image.Type]
        IMAGE_TYPE_KERNEL_RANCHU: _ClassVar[Image.Type]
        IMAGE_TYPE_SYSTEM: _ClassVar[Image.Type]
        IMAGE_TYPE_SYSTEM_COPY: _ClassVar[Image.Type]
        IMAGE_TYPE_DATA: _ClassVar[Image.Type]
        IMAGE_TYPE_DATA_COPY: _ClassVar[Image.Type]
        IMAGE_TYPE_RAMDISK: _ClassVar[Image.Type]
        IMAGE_TYPE_SDCARD: _ClassVar[Image.Type]
        IMAGE_TYPE_CACHE: _ClassVar[Image.Type]
        IMAGE_TYPE_VENDOR: _ClassVar[Image.Type]
        IMAGE_TYPE_ENCRYPTION_KEY: _ClassVar[Image.Type]
    IMAGE_TYPE_UNKNOWN: Image.Type
    IMAGE_TYPE_KERNEL: Image.Type
    IMAGE_TYPE_KERNEL_RANCHU: Image.Type
    IMAGE_TYPE_SYSTEM: Image.Type
    IMAGE_TYPE_SYSTEM_COPY: Image.Type
    IMAGE_TYPE_DATA: Image.Type
    IMAGE_TYPE_DATA_COPY: Image.Type
    IMAGE_TYPE_RAMDISK: Image.Type
    IMAGE_TYPE_SDCARD: Image.Type
    IMAGE_TYPE_CACHE: Image.Type
    IMAGE_TYPE_VENDOR: Image.Type
    IMAGE_TYPE_ENCRYPTION_KEY: Image.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    PRESENT_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    MODIFICATION_TIME_FIELD_NUMBER: _ClassVar[int]
    type: Image.Type
    path: str
    present: bool
    size: int
    modification_time: int
    def __init__(self, type: _Optional[_Union[Image.Type, str]] = ..., path: _Optional[str] = ..., present: bool = ..., size: _Optional[int] = ..., modification_time: _Optional[int] = ...) -> None: ...

class Host(_message.Message):
    __slots__ = ("gpu_driver", "hypervisor")
    GPU_DRIVER_FIELD_NUMBER: _ClassVar[int]
    HYPERVISOR_FIELD_NUMBER: _ClassVar[int]
    gpu_driver: str
    hypervisor: int
    def __init__(self, gpu_driver: _Optional[str] = ..., hypervisor: _Optional[int] = ...) -> None: ...

class Config(_message.Message):
    __slots__ = ("enabled_features", "selected_renderer", "cpu_core_count", "ram_size_bytes")
    ENABLED_FEATURES_FIELD_NUMBER: _ClassVar[int]
    SELECTED_RENDERER_FIELD_NUMBER: _ClassVar[int]
    CPU_CORE_COUNT_FIELD_NUMBER: _ClassVar[int]
    RAM_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    enabled_features: _containers.RepeatedScalarFieldContainer[int]
    selected_renderer: int
    cpu_core_count: int
    ram_size_bytes: int
    def __init__(self, enabled_features: _Optional[_Iterable[int]] = ..., selected_renderer: _Optional[int] = ..., cpu_core_count: _Optional[int] = ..., ram_size_bytes: _Optional[int] = ...) -> None: ...

class SaveStats(_message.Message):
    __slots__ = ("incremental", "duration", "ram_changed_bytes")
    INCREMENTAL_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    RAM_CHANGED_BYTES_FIELD_NUMBER: _ClassVar[int]
    incremental: int
    duration: int
    ram_changed_bytes: int
    def __init__(self, incremental: _Optional[int] = ..., duration: _Optional[int] = ..., ram_changed_bytes: _Optional[int] = ...) -> None: ...

class Snapshot(_message.Message):
    __slots__ = ("version", "creation_time", "images", "host", "config", "failed_to_load_reason_code", "guest_data_partition_mounted", "rotation", "invalid_loads", "successful_loads", "logical_name", "parent", "description", "save_stats", "folded", "launch_parameters", "emulator_build_id", "system_image_build_id")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    FAILED_TO_LOAD_REASON_CODE_FIELD_NUMBER: _ClassVar[int]
    GUEST_DATA_PARTITION_MOUNTED_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    INVALID_LOADS_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_LOADS_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_NAME_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SAVE_STATS_FIELD_NUMBER: _ClassVar[int]
    FOLDED_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    EMULATOR_BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_IMAGE_BUILD_ID_FIELD_NUMBER: _ClassVar[int]
    version: int
    creation_time: int
    images: _containers.RepeatedCompositeFieldContainer[Image]
    host: Host
    config: Config
    failed_to_load_reason_code: int
    guest_data_partition_mounted: bool
    rotation: int
    invalid_loads: int
    successful_loads: int
    logical_name: str
    parent: str
    description: str
    save_stats: _containers.RepeatedCompositeFieldContainer[SaveStats]
    folded: bool
    launch_parameters: _containers.RepeatedScalarFieldContainer[str]
    emulator_build_id: str
    system_image_build_id: str
    def __init__(self, version: _Optional[int] = ..., creation_time: _Optional[int] = ..., images: _Optional[_Iterable[_Union[Image, _Mapping]]] = ..., host: _Optional[_Union[Host, _Mapping]] = ..., config: _Optional[_Union[Config, _Mapping]] = ..., failed_to_load_reason_code: _Optional[int] = ..., guest_data_partition_mounted: bool = ..., rotation: _Optional[int] = ..., invalid_loads: _Optional[int] = ..., successful_loads: _Optional[int] = ..., logical_name: _Optional[str] = ..., parent: _Optional[str] = ..., description: _Optional[str] = ..., save_stats: _Optional[_Iterable[_Union[SaveStats, _Mapping]]] = ..., folded: bool = ..., launch_parameters: _Optional[_Iterable[str]] = ..., emulator_build_id: _Optional[str] = ..., system_image_build_id: _Optional[str] = ...) -> None: ...
