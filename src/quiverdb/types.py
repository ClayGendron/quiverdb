"""Composable validated types for GraphModel.

These types resolve to base Python types for SQLAlchemy compatibility
while providing Pydantic-style validation.

Usage:
    from quiverdb.types import Str, Int, Float, Email, Url, Positive, MaxLen
    from quiverdb.types import EmailStr, HttpUrl, PositiveInt  # Pre-built

    class User(GraphNode):
        # Composable syntax
        email: Str[Email]
        url: Str[Url, MaxLen(2000)]
        age: Int[Positive, Lt(150)]
        score: Float[Ge(0), Le(100)]

        # Or use pre-built types
        email: EmailStr
        url: HttpUrl
        count: PositiveInt
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)

from pydantic import EmailStr as PydanticEmailStr
from pydantic import GetCoreSchemaHandler
from pydantic import HttpUrl as PydanticHttpUrl
from pydantic import validate_email
from pydantic_core import CoreSchema, core_schema

# ============ SCHEMA UTILITIES ============

# Schema types that support direct constraints (e.g., min_length, gt, le, etc.)
_BASE_SCHEMA_TYPES = frozenset({
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "decimal",
    "list",
    "set",
    "frozenset",
    "dict",
})


def _find_base_schema(schema: CoreSchema) -> CoreSchema:
    """Find the innermost base schema that supports constraints.

    When validators wrap the base schema (e.g., function-after wrapping str),
    constraints like min_length must be applied to the inner base schema,
    not the outer wrapper.

    Args:
        schema: A Pydantic CoreSchema, possibly with nested wrappers.

    Returns:
        The innermost schema that supports constraints.
    """
    schema_type = schema.get("type")

    # If this is a base schema type, return it
    if schema_type in _BASE_SCHEMA_TYPES:
        return schema

    # If this schema wraps another schema, recurse into it
    if "schema" in schema:
        return _find_base_schema(schema["schema"])

    # Fallback: return the schema as-is if no nested schema found
    return schema


# ============ NUMERIC CONSTRAINTS ============


@dataclass(frozen=True)
class Gt:
    """Greater than constraint.

    Usage:
        age: Int[Gt(0)]        # age > 0
        score: Float[Gt(0.5)]  # score > 0.5
    """

    value: float

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["gt"] = self.value
        return schema


@dataclass(frozen=True)
class Ge:
    """Greater than or equal constraint.

    Usage:
        age: Int[Ge(0)]        # age >= 0
        score: Float[Ge(0.0)]  # score >= 0.0
    """

    value: float

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["ge"] = self.value
        return schema


@dataclass(frozen=True)
class Lt:
    """Less than constraint.

    Usage:
        age: Int[Lt(150)]       # age < 150
        score: Float[Lt(100)]   # score < 100
    """

    value: float

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["lt"] = self.value
        return schema


@dataclass(frozen=True)
class Le:
    """Less than or equal constraint.

    Usage:
        age: Int[Le(150)]       # age <= 150
        score: Float[Le(100)]   # score <= 100
    """

    value: float

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["le"] = self.value
        return schema


@dataclass(frozen=True)
class MultipleOf:
    """Multiple of constraint.

    Usage:
        quantity: Int[MultipleOf(5)]    # Must be divisible by 5
        price: Float[MultipleOf(0.01)]  # Two decimal places
    """

    value: float

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["multiple_of"] = self.value
        return schema


class Positive:
    """Positive number constraint (> 0).

    Usage:
        count: Int[Positive]
        amount: Float[Positive]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["gt"] = 0
        return schema


class NonNegative:
    """Non-negative number constraint (>= 0).

    Usage:
        count: Int[NonNegative]
        balance: Float[NonNegative]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["ge"] = 0
        return schema


class Negative:
    """Negative number constraint (< 0).

    Usage:
        debt: Float[Negative]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["lt"] = 0
        return schema


class NonPositive:
    """Non-positive number constraint (<= 0).

    Usage:
        adjustment: Float[NonPositive]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["le"] = 0
        return schema


# ============ STRING/COLLECTION LENGTH CONSTRAINTS ============


@dataclass(frozen=True)
class MinLen:
    """Minimum length constraint for strings/collections.

    Usage:
        name: Str[MinLen(1)]           # At least 1 character
        tags: list[str][MinLen(1)]     # At least 1 item
    """

    length: int

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["min_length"] = self.length
        return schema


@dataclass(frozen=True)
class MaxLen:
    """Maximum length constraint for strings/collections.

    Usage:
        name: Str[MaxLen(100)]         # At most 100 characters
        tags: list[str][MaxLen(10)]    # At most 10 items
    """

    length: int

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["max_length"] = self.length
        return schema


@dataclass(frozen=True)
class Len:
    """Exact or range length constraint.

    Usage:
        code: Str[Len(6)]              # Exactly 6 characters
        name: Str[Len(1, 100)]         # Between 1 and 100 characters
    """

    min_length: int
    max_length: int | None = None

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        base_schema = _find_base_schema(schema)
        if self.max_length is None:
            # Exact length
            base_schema["min_length"] = self.min_length
            base_schema["max_length"] = self.min_length
        else:
            base_schema["min_length"] = self.min_length
            base_schema["max_length"] = self.max_length
        return schema


# ============ STRING FORMAT VALIDATORS ============


class Email:
    """Email format validator.

    Usage:
        email: Str[Email]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        validate_email(v)
        return v


class Url:
    """HTTP/HTTPS URL validator.

    Usage:
        website: Str[Url]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        PydanticHttpUrl(v)
        return v


class NonEmpty:
    """Non-empty string constraint (not empty or whitespace only).

    Usage:
        name: Str[NonEmpty]
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("String cannot be empty or whitespace")
        return v


class Trimmed:
    """Strips whitespace from string.

    Usage:
        name: Str[Trimmed]  # "  hello  " -> "hello"
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        return v.strip()


class Lowercase:
    """Converts string to lowercase.

    Usage:
        username: Str[Lowercase]  # "Hello" -> "hello"
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        return v.lower()


class Uppercase:
    """Converts string to uppercase.

    Usage:
        code: Str[Uppercase]  # "hello" -> "HELLO"
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate, handler(source_type)
        )

    def _validate(self, v: str) -> str:
        return v.upper()


@dataclass(frozen=True)
class Pattern:
    """Regex pattern constraint.

    Usage:
        phone: Str[Pattern(r"^\\d{3}-\\d{3}-\\d{4}$")]
    """

    pattern: str

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)
        _find_base_schema(schema)["pattern"] = self.pattern
        return schema


# ============ COMPOSABLE BASE TYPES ============

if TYPE_CHECKING:
    # Type stubs for type checkers - these help IDEs and type checkers understand
    # that Str[...] returns a str type, Int[...] returns an int type, etc.
    # At runtime, these are replaced by the actual _ConstrainedType subclasses.

    class Str:
        """String type with optional constraints."""

        _base_type: type[str]

        def __class_getitem__(cls, constraints: Any) -> type[str]: ...

    class Int:
        """Integer type with optional constraints."""

        _base_type: type[int]

        def __class_getitem__(cls, constraints: Any) -> type[int]: ...

    class Float:
        """Float type with optional constraints."""

        _base_type: type[float]

        def __class_getitem__(cls, constraints: Any) -> type[float]: ...

    class Bool:
        """Boolean type."""

        _base_type: type[bool]

        def __class_getitem__(cls, constraints: Any) -> type[bool]: ...

    class Bytes:
        """Bytes type with optional constraints."""

        _base_type: type[bytes]

        def __class_getitem__(cls, constraints: Any) -> type[bytes]: ...

    class DateTime:
        """Datetime type."""

        _base_type: type[datetime]

        def __class_getitem__(cls, constraints: Any) -> type[datetime]: ...

    class Date:
        """Date type."""

        _base_type: type[date]

        def __class_getitem__(cls, constraints: Any) -> type[date]: ...

    class Time:
        """Time type."""

        _base_type: type[time]

        def __class_getitem__(cls, constraints: Any) -> type[time]: ...

    class Dec:
        """Decimal type with optional constraints."""

        _base_type: type[Decimal]

        def __class_getitem__(cls, constraints: Any) -> type[Decimal]: ...

else:
    # Runtime implementations

    class _ConstrainedType:
        """Base for subscriptable constrained types like Str[Email]."""

        _base_type: type

        def __class_getitem__(cls, constraints: Any) -> type:
            """Enable Str[Email, MaxLen(100)] syntax."""
            if not isinstance(constraints, tuple):
                constraints = (constraints,)

            # Instantiate constraint classes that aren't already instances
            processed = []
            for c in constraints:
                if isinstance(c, type):
                    processed.append(c())
                else:
                    processed.append(c)

            return Annotated[cls._base_type, *processed]

    class Str(_ConstrainedType):
        """String type with optional constraints.

        Usage:
            email: Str[Email]
            name: Str[MinLen(1), MaxLen(100)]
            code: Str[Pattern(r"^[A-Z]{3}$")]
        """

        _base_type = str

    class Int(_ConstrainedType):
        """Integer type with optional constraints.

        Usage:
            age: Int[Positive, Lt(150)]
            count: Int[Ge(0)]
            quantity: Int[MultipleOf(5)]
        """

        _base_type = int

    class Float(_ConstrainedType):
        """Float type with optional constraints.

        Usage:
            score: Float[Ge(0), Le(100)]
            temperature: Float[Gt(-273.15)]
            price: Float[Positive, MultipleOf(0.01)]
        """

        _base_type = float

    class Bool(_ConstrainedType):
        """Boolean type (primarily for documentation/consistency)."""

        _base_type = bool

    class Bytes(_ConstrainedType):
        """Bytes type with optional constraints.

        Usage:
            data: Bytes[MaxLen(1024)]
        """

        _base_type = bytes

    class DateTime(_ConstrainedType):
        """Datetime type."""

        _base_type = datetime

    class Date(_ConstrainedType):
        """Date type."""

        _base_type = date

    class Time(_ConstrainedType):
        """Time type."""

        _base_type = time

    class Dec(_ConstrainedType):
        """Decimal type with optional constraints.

        Usage:
            price: Dec[Positive]
            balance: Dec[Ge(0)]
        """

        _base_type = Decimal


# ============ PRE-BUILT TYPE ALIASES ============

# String types
EmailStr: TypeAlias = Annotated[str, Email()]
HttpUrl: TypeAlias = Annotated[str, Url()]
NonEmptyStr: TypeAlias = Annotated[str, NonEmpty()]
TrimmedStr: TypeAlias = Annotated[str, Trimmed()]

# Integer types
PositiveInt: TypeAlias = Annotated[int, Positive()]
NonNegativeInt: TypeAlias = Annotated[int, NonNegative()]
NegativeInt: TypeAlias = Annotated[int, Negative()]

# Float types
PositiveFloat: TypeAlias = Annotated[float, Positive()]
NonNegativeFloat: TypeAlias = Annotated[float, NonNegative()]
NegativeFloat: TypeAlias = Annotated[float, Negative()]

# Common ranges
Percentage: TypeAlias = Annotated[float, Ge(0), Le(100)]
UnitInterval: TypeAlias = Annotated[float, Ge(0), Le(1)]


# ============ TYPE EXTRACTION UTILITIES ============

# Supported base types for SQLAlchemy
SUPPORTED_BASE_TYPES = frozenset(
    {
        str,
        int,
        float,
        bool,
        bytes,
        datetime,
        date,
        time,
        Decimal,
        list,
        dict,
    }
)


def get_base_type(annotation: Any) -> type | None:
    """Extract the base Python type from a constrained annotation.

    This is used by GraphModel's metaclass to get SQLAlchemy-compatible types.

    Examples:
        get_base_type(Str[Email]) -> str
        get_base_type(EmailStr) -> str
        get_base_type(Int[Positive]) -> int
        get_base_type(str | None) -> str

    Args:
        annotation: A type annotation, possibly with constraints.

    Returns:
        The base Python type, or None if it cannot be determined.
    """
    origin = get_origin(annotation)

    # Handle Annotated[str, ...] -> str
    if origin is Annotated:
        return get_args(annotation)[0]

    # Handle Optional[str] / str | None -> str
    # Note: types.UnionType is the runtime type for X | Y syntax (Python 3.10+)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return get_base_type(non_none_args[0])
        return None

    # Handle _ConstrainedType subclasses
    if hasattr(annotation, "_base_type"):
        return annotation._base_type

    # Handle plain types
    if isinstance(annotation, type):
        return annotation

    return None


def is_supported_type(annotation: Any) -> bool:
    """Check if an annotation resolves to a SQLAlchemy-supported type.

    Args:
        annotation: A type annotation to check.

    Returns:
        True if the base type is supported by SQLAlchemy.
    """
    base = get_base_type(annotation)
    return base in SUPPORTED_BASE_TYPES if base else False
