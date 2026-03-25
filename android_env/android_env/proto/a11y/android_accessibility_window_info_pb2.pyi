from android_env.proto.a11y import android_accessibility_tree_pb2 as _android_accessibility_tree_pb2
from android_env.proto.a11y import rect_pb2 as _rect_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AndroidAccessibilityWindowInfo(_message.Message):
    __slots__ = ("bounds_in_screen", "display_id", "id", "layer", "title", "window_type", "is_accessibility_focused", "is_active", "is_focused", "is_in_picture_in_picture_mode", "tree")
    class WindowType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN_TYPE: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
        TYPE_APPLICATION: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
        TYPE_INPUT_METHOD: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
        TYPE_SYSTEM: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
        TYPE_ACCESSIBILITY_OVERLAY: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
        TYPE_SPLIT_SCREEN_DIVIDER: _ClassVar[AndroidAccessibilityWindowInfo.WindowType]
    UNKNOWN_TYPE: AndroidAccessibilityWindowInfo.WindowType
    TYPE_APPLICATION: AndroidAccessibilityWindowInfo.WindowType
    TYPE_INPUT_METHOD: AndroidAccessibilityWindowInfo.WindowType
    TYPE_SYSTEM: AndroidAccessibilityWindowInfo.WindowType
    TYPE_ACCESSIBILITY_OVERLAY: AndroidAccessibilityWindowInfo.WindowType
    TYPE_SPLIT_SCREEN_DIVIDER: AndroidAccessibilityWindowInfo.WindowType
    BOUNDS_IN_SCREEN_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_ACCESSIBILITY_FOCUSED_FIELD_NUMBER: _ClassVar[int]
    IS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    IS_FOCUSED_FIELD_NUMBER: _ClassVar[int]
    IS_IN_PICTURE_IN_PICTURE_MODE_FIELD_NUMBER: _ClassVar[int]
    TREE_FIELD_NUMBER: _ClassVar[int]
    bounds_in_screen: _rect_pb2.ProtoRect
    display_id: int
    id: int
    layer: int
    title: str
    window_type: AndroidAccessibilityWindowInfo.WindowType
    is_accessibility_focused: bool
    is_active: bool
    is_focused: bool
    is_in_picture_in_picture_mode: bool
    tree: _android_accessibility_tree_pb2.AndroidAccessibilityTree
    def __init__(self, bounds_in_screen: _Optional[_Union[_rect_pb2.ProtoRect, _Mapping]] = ..., display_id: _Optional[int] = ..., id: _Optional[int] = ..., layer: _Optional[int] = ..., title: _Optional[str] = ..., window_type: _Optional[_Union[AndroidAccessibilityWindowInfo.WindowType, str]] = ..., is_accessibility_focused: bool = ..., is_active: bool = ..., is_focused: bool = ..., is_in_picture_in_picture_mode: bool = ..., tree: _Optional[_Union[_android_accessibility_tree_pb2.AndroidAccessibilityTree, _Mapping]] = ...) -> None: ...
