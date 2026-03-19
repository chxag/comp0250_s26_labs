# generated from rosidl_generator_py/resource/_idl.py.em
# with input from cw1_team_x:srv/Example.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Example_Request(type):
    """Metaclass of message 'Example_Request'."""

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
            module = import_type_support('cw1_team_x')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_team_x.srv.Example_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__example__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__example__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__example__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__example__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__example__request

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Example_Request(metaclass=Metaclass_Example_Request):
    """Message class 'Example_Request'."""

    __slots__ = [
        '_example_float',
        '_example_int',
        '_example_string',
        '_example_pose',
        '_example_std_vector_of_poses',
    ]

    _fields_and_field_types = {
        'example_float': 'double',
        'example_int': 'int32',
        'example_string': 'string',
        'example_pose': 'geometry_msgs/Pose',
        'example_std_vector_of_poses': 'sequence<geometry_msgs/Pose>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.example_float = kwargs.get('example_float', float())
        self.example_int = kwargs.get('example_int', int())
        self.example_string = kwargs.get('example_string', str())
        from geometry_msgs.msg import Pose
        self.example_pose = kwargs.get('example_pose', Pose())
        self.example_std_vector_of_poses = kwargs.get('example_std_vector_of_poses', [])

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
        if self.example_float != other.example_float:
            return False
        if self.example_int != other.example_int:
            return False
        if self.example_string != other.example_string:
            return False
        if self.example_pose != other.example_pose:
            return False
        if self.example_std_vector_of_poses != other.example_std_vector_of_poses:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def example_float(self):
        """Message field 'example_float'."""
        return self._example_float

    @example_float.setter
    def example_float(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'example_float' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'example_float' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._example_float = value

    @builtins.property
    def example_int(self):
        """Message field 'example_int'."""
        return self._example_int

    @example_int.setter
    def example_int(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'example_int' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'example_int' field must be an integer in [-2147483648, 2147483647]"
        self._example_int = value

    @builtins.property
    def example_string(self):
        """Message field 'example_string'."""
        return self._example_string

    @example_string.setter
    def example_string(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'example_string' field must be of type 'str'"
        self._example_string = value

    @builtins.property
    def example_pose(self):
        """Message field 'example_pose'."""
        return self._example_pose

    @example_pose.setter
    def example_pose(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'example_pose' field must be a sub message of type 'Pose'"
        self._example_pose = value

    @builtins.property
    def example_std_vector_of_poses(self):
        """Message field 'example_std_vector_of_poses'."""
        return self._example_std_vector_of_poses

    @example_std_vector_of_poses.setter
    def example_std_vector_of_poses(self, value):
        if __debug__:
            from geometry_msgs.msg import Pose
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Pose) for v in value) and
                 True), \
                "The 'example_std_vector_of_poses' field must be a set or sequence and each value of type 'Pose'"
        self._example_std_vector_of_poses = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_Example_Response(type):
    """Metaclass of message 'Example_Response'."""

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
            module = import_type_support('cw1_team_x')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_team_x.srv.Example_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__example__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__example__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__example__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__example__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__example__response

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Example_Response(metaclass=Metaclass_Example_Response):
    """Message class 'Example_Response'."""

    __slots__ = [
        '_success',
        '_example_std_vector_of_points',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
        'example_std_vector_of_points': 'sequence<geometry_msgs/Point>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())
        self.example_std_vector_of_points = kwargs.get('example_std_vector_of_points', [])

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
        if self.success != other.success:
            return False
        if self.example_std_vector_of_points != other.example_std_vector_of_points:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value

    @builtins.property
    def example_std_vector_of_points(self):
        """Message field 'example_std_vector_of_points'."""
        return self._example_std_vector_of_points

    @example_std_vector_of_points.setter
    def example_std_vector_of_points(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Point) for v in value) and
                 True), \
                "The 'example_std_vector_of_points' field must be a set or sequence and each value of type 'Point'"
        self._example_std_vector_of_points = value


class Metaclass_Example(type):
    """Metaclass of service 'Example'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw1_team_x')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw1_team_x.srv.Example')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__example

            from cw1_team_x.srv import _example
            if _example.Metaclass_Example_Request._TYPE_SUPPORT is None:
                _example.Metaclass_Example_Request.__import_type_support__()
            if _example.Metaclass_Example_Response._TYPE_SUPPORT is None:
                _example.Metaclass_Example_Response.__import_type_support__()


class Example(metaclass=Metaclass_Example):
    from cw1_team_x.srv._example import Example_Request as Request
    from cw1_team_x.srv._example import Example_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
