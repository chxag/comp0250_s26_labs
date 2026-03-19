# generated from rosidl_generator_py/resource/_idl.py.em
# with input from cw1_world_spawner:srv/Task1Service.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Task1Service_Request(type):
    """Metaclass of message 'Task1Service_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw1_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_world_spawner.srv.Task1Service_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__task1_service__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__task1_service__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__task1_service__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__task1_service__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__task1_service__request

            from geometry_msgs.msg import PointStamped
            if PointStamped.__class__._TYPE_SUPPORT is None:
                PointStamped.__class__.__import_type_support__()

            from geometry_msgs.msg import PoseStamped
            if PoseStamped.__class__._TYPE_SUPPORT is None:
                PoseStamped.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Task1Service_Request(metaclass=Metaclass_Task1Service_Request):
    """Message class 'Task1Service_Request'."""

    __slots__ = [
        '_object_loc',
        '_goal_loc',
    ]

    _fields_and_field_types = {
        'object_loc': 'geometry_msgs/PoseStamped',
        'goal_loc': 'geometry_msgs/PointStamped',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'PoseStamped'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'PointStamped'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from geometry_msgs.msg import PoseStamped
        self.object_loc = kwargs.get('object_loc', PoseStamped())
        from geometry_msgs.msg import PointStamped
        self.goal_loc = kwargs.get('goal_loc', PointStamped())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.object_loc != other.object_loc:
            return False
        if self.goal_loc != other.goal_loc:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def object_loc(self):
        """Message field 'object_loc'."""
        return self._object_loc

    @object_loc.setter
    def object_loc(self, value):
        if __debug__:
            from geometry_msgs.msg import PoseStamped
            assert \
                isinstance(value, PoseStamped), \
                "The 'object_loc' field must be a sub message of type 'PoseStamped'"
        self._object_loc = value

    @builtins.property
    def goal_loc(self):
        """Message field 'goal_loc'."""
        return self._goal_loc

    @goal_loc.setter
    def goal_loc(self, value):
        if __debug__:
            from geometry_msgs.msg import PointStamped
            assert \
                isinstance(value, PointStamped), \
                "The 'goal_loc' field must be a sub message of type 'PointStamped'"
        self._goal_loc = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_Task1Service_Response(type):
    """Metaclass of message 'Task1Service_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw1_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_world_spawner.srv.Task1Service_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__task1_service__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__task1_service__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__task1_service__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__task1_service__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__task1_service__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Task1Service_Response(metaclass=Metaclass_Task1Service_Response):
    """Message class 'Task1Service_Response'."""

    __slots__ = [
    ]

    _fields_and_field_types = {
    }

    SLOT_TYPES = (
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)


class Metaclass_Task1Service(type):
    """Metaclass of service 'Task1Service'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw1_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_world_spawner.srv.Task1Service')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__task1_service

            from cw1_world_spawner.srv import _task1_service
            if _task1_service.Metaclass_Task1Service_Request._TYPE_SUPPORT is None:
                _task1_service.Metaclass_Task1Service_Request.__import_type_support__()
            if _task1_service.Metaclass_Task1Service_Response._TYPE_SUPPORT is None:
                _task1_service.Metaclass_Task1Service_Response.__import_type_support__()


class Task1Service(metaclass=Metaclass_Task1Service):
    from cw1_world_spawner.srv._task1_service import Task1Service_Request as Request
    from cw1_world_spawner.srv._task1_service import Task1Service_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
