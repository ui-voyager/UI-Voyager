from android_env.proto.a11y import android_accessibility_action_pb2 as _android_accessibility_action_pb2
from android_env.proto.a11y import android_accessibility_node_info_clickable_span_pb2 as _android_accessibility_node_info_clickable_span_pb2
from android_env.proto.a11y import rect_pb2 as _rect_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AndroidAccessibilityNodeInfo(_message.Message):
    __slots__ = ("unique_id", "bounds_in_screen", "class_name", "content_description", "hint_text", "package_name", "text", "text_selection_start", "text_selection_end", "view_id_resource_name", "window_id", "is_checkable", "is_checked", "is_clickable", "is_editable", "is_enabled", "is_focusable", "is_focused", "is_long_clickable", "is_password", "is_scrollable", "is_selected", "is_visible_to_user", "actions", "child_ids", "clickable_spans", "depth", "labeled_by_id", "label_for_id", "drawing_order", "tooltip_text")
    UNIQUE_ID_FIELD_NUMBER: _ClassVar[int]
    BOUNDS_IN_SCREEN_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    HINT_TEXT_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TEXT_SELECTION_START_FIELD_NUMBER: _ClassVar[int]
    TEXT_SELECTION_END_FIELD_NUMBER: _ClassVar[int]
    VIEW_ID_RESOURCE_NAME_FIELD_NUMBER: _ClassVar[int]
    WINDOW_ID_FIELD_NUMBER: _ClassVar[int]
    IS_CHECKABLE_FIELD_NUMBER: _ClassVar[int]
    IS_CHECKED_FIELD_NUMBER: _ClassVar[int]
    IS_CLICKABLE_FIELD_NUMBER: _ClassVar[int]
    IS_EDITABLE_FIELD_NUMBER: _ClassVar[int]
    IS_ENABLED_FIELD_NUMBER: _ClassVar[int]
    IS_FOCUSABLE_FIELD_NUMBER: _ClassVar[int]
    IS_FOCUSED_FIELD_NUMBER: _ClassVar[int]
    IS_LONG_CLICKABLE_FIELD_NUMBER: _ClassVar[int]
    IS_PASSWORD_FIELD_NUMBER: _ClassVar[int]
    IS_SCROLLABLE_FIELD_NUMBER: _ClassVar[int]
    IS_SELECTED_FIELD_NUMBER: _ClassVar[int]
    IS_VISIBLE_TO_USER_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    CHILD_IDS_FIELD_NUMBER: _ClassVar[int]
    CLICKABLE_SPANS_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    LABELED_BY_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FOR_ID_FIELD_NUMBER: _ClassVar[int]
    DRAWING_ORDER_FIELD_NUMBER: _ClassVar[int]
    TOOLTIP_TEXT_FIELD_NUMBER: _ClassVar[int]
    unique_id: int
    bounds_in_screen: _rect_pb2.ProtoRect
    class_name: str
    content_description: str
    hint_text: str
    package_name: str
    text: str
    text_selection_start: int
    text_selection_end: int
    view_id_resource_name: str
    window_id: int
    is_checkable: bool
    is_checked: bool
    is_clickable: bool
    is_editable: bool
    is_enabled: bool
    is_focusable: bool
    is_focused: bool
    is_long_clickable: bool
    is_password: bool
    is_scrollable: bool
    is_selected: bool
    is_visible_to_user: bool
    actions: _containers.RepeatedCompositeFieldContainer[_android_accessibility_action_pb2.AndroidAccessibilityAction]
    child_ids: _containers.RepeatedScalarFieldContainer[int]
    clickable_spans: _containers.RepeatedCompositeFieldContainer[_android_accessibility_node_info_clickable_span_pb2.AndroidAccessibilityNodeInfoClickableSpan]
    depth: int
    labeled_by_id: int
    label_for_id: int
    drawing_order: int
    tooltip_text: str
    def __init__(self, unique_id: _Optional[int] = ..., bounds_in_screen: _Optional[_Union[_rect_pb2.ProtoRect, _Mapping]] = ..., class_name: _Optional[str] = ..., content_description: _Optional[str] = ..., hint_text: _Optional[str] = ..., package_name: _Optional[str] = ..., text: _Optional[str] = ..., text_selection_start: _Optional[int] = ..., text_selection_end: _Optional[int] = ..., view_id_resource_name: _Optional[str] = ..., window_id: _Optional[int] = ..., is_checkable: bool = ..., is_checked: bool = ..., is_clickable: bool = ..., is_editable: bool = ..., is_enabled: bool = ..., is_focusable: bool = ..., is_focused: bool = ..., is_long_clickable: bool = ..., is_password: bool = ..., is_scrollable: bool = ..., is_selected: bool = ..., is_visible_to_user: bool = ..., actions: _Optional[_Iterable[_Union[_android_accessibility_action_pb2.AndroidAccessibilityAction, _Mapping]]] = ..., child_ids: _Optional[_Iterable[int]] = ..., clickable_spans: _Optional[_Iterable[_Union[_android_accessibility_node_info_clickable_span_pb2.AndroidAccessibilityNodeInfoClickableSpan, _Mapping]]] = ..., depth: _Optional[int] = ..., labeled_by_id: _Optional[int] = ..., label_for_id: _Optional[int] = ..., drawing_order: _Optional[int] = ..., tooltip_text: _Optional[str] = ...) -> None: ...
