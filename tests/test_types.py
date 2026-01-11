"""Tests for quiverdb.types module."""

from typing import Annotated, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from quiverdb.types import (
    # Composable base types
    Bool,
    Bytes,
    Date,
    DateTime,
    Dec,
    Float,
    Int,
    Str,
    Time,
    # Numeric constraints
    Ge,
    Gt,
    Le,
    Lt,
    MultipleOf,
    Negative,
    NonNegative,
    NonPositive,
    Positive,
    # Length constraints
    Len,
    MaxLen,
    MinLen,
    # String validators
    Email,
    Lowercase,
    NonEmpty,
    Pattern,
    Trimmed,
    Uppercase,
    Url,
    # Pre-built types
    EmailStr,
    HttpUrl,
    NegativeFloat,
    NegativeInt,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Percentage,
    PositiveFloat,
    PositiveInt,
    TrimmedStr,
    UnitInterval,
    # Utilities
    get_base_type,
    is_supported_type,
)


class TestComposableTypes:
    """Tests for Str, Int, Float, etc. subscript syntax."""

    def test_str_with_single_constraint(self) -> None:
        """Str[Email] produces Annotated[str, Email()]."""
        result = Str[Email]
        assert get_origin(result) is Annotated
        assert get_args(result)[0] is str
        assert isinstance(get_args(result)[1], Email)

    def test_str_with_multiple_constraints(self) -> None:
        """Str[MinLen(1), MaxLen(100)] produces Annotated with multiple constraints."""
        result = Str[MinLen(1), MaxLen(100)]
        assert get_origin(result) is Annotated
        args = get_args(result)
        assert args[0] is str
        assert isinstance(args[1], MinLen)
        assert args[1].length == 1
        assert isinstance(args[2], MaxLen)
        assert args[2].length == 100

    def test_int_with_constraints(self) -> None:
        """Int[Positive, Lt(150)] produces Annotated[int, ...]."""
        result = Int[Positive, Lt(150)]
        assert get_origin(result) is Annotated
        args = get_args(result)
        assert args[0] is int
        assert isinstance(args[1], Positive)
        assert isinstance(args[2], Lt)
        assert args[2].value == 150

    def test_float_with_constraints(self) -> None:
        """Float[Ge(0), Le(100)] produces Annotated[float, ...]."""
        result = Float[Ge(0), Le(100)]
        assert get_origin(result) is Annotated
        args = get_args(result)
        assert args[0] is float

    def test_constraint_class_auto_instantiated(self) -> None:
        """Constraint classes are auto-instantiated when passed as types."""
        result = Str[NonEmpty]  # NonEmpty without ()
        args = get_args(result)
        assert isinstance(args[1], NonEmpty)

    def test_all_base_types_exist(self) -> None:
        """All base type wrappers have correct _base_type."""
        from datetime import date, datetime, time
        from decimal import Decimal

        assert Str._base_type is str
        assert Int._base_type is int
        assert Float._base_type is float
        assert Bool._base_type is bool
        assert Bytes._base_type is bytes
        assert DateTime._base_type is datetime
        assert Date._base_type is date
        assert Time._base_type is time
        assert Dec._base_type is Decimal


class TestNumericConstraints:
    """Tests for Gt, Ge, Lt, Le, Positive, etc."""

    def test_gt_constraint_valid(self) -> None:
        """Gt constraint accepts values greater than threshold."""

        class Model(BaseModel):
            value: Int[Gt(0)]

        m = Model(value=1)
        assert m.value == 1

        m = Model(value=100)
        assert m.value == 100

    def test_gt_constraint_invalid(self) -> None:
        """Gt constraint rejects values at or below threshold."""

        class Model(BaseModel):
            value: Int[Gt(0)]

        with pytest.raises(ValidationError):
            Model(value=0)

        with pytest.raises(ValidationError):
            Model(value=-1)

    def test_ge_constraint_valid(self) -> None:
        """Ge constraint accepts values greater than or equal to threshold."""

        class Model(BaseModel):
            value: Int[Ge(0)]

        assert Model(value=0).value == 0
        assert Model(value=1).value == 1

    def test_ge_constraint_invalid(self) -> None:
        """Ge constraint rejects values below threshold."""

        class Model(BaseModel):
            value: Int[Ge(0)]

        with pytest.raises(ValidationError):
            Model(value=-1)

    def test_lt_constraint_valid(self) -> None:
        """Lt constraint accepts values less than threshold."""

        class Model(BaseModel):
            value: Int[Lt(100)]

        assert Model(value=99).value == 99
        assert Model(value=0).value == 0

    def test_lt_constraint_invalid(self) -> None:
        """Lt constraint rejects values at or above threshold."""

        class Model(BaseModel):
            value: Int[Lt(100)]

        with pytest.raises(ValidationError):
            Model(value=100)

        with pytest.raises(ValidationError):
            Model(value=101)

    def test_le_constraint_valid(self) -> None:
        """Le constraint accepts values less than or equal to threshold."""

        class Model(BaseModel):
            value: Int[Le(100)]

        assert Model(value=100).value == 100
        assert Model(value=99).value == 99

    def test_le_constraint_invalid(self) -> None:
        """Le constraint rejects values above threshold."""

        class Model(BaseModel):
            value: Int[Le(100)]

        with pytest.raises(ValidationError):
            Model(value=101)

    def test_positive_constraint(self) -> None:
        """Positive constraint requires value > 0."""

        class Model(BaseModel):
            value: Int[Positive]

        assert Model(value=1).value == 1

        with pytest.raises(ValidationError):
            Model(value=0)

        with pytest.raises(ValidationError):
            Model(value=-1)

    def test_non_negative_constraint(self) -> None:
        """NonNegative constraint requires value >= 0."""

        class Model(BaseModel):
            value: Int[NonNegative]

        assert Model(value=0).value == 0
        assert Model(value=1).value == 1

        with pytest.raises(ValidationError):
            Model(value=-1)

    def test_negative_constraint(self) -> None:
        """Negative constraint requires value < 0."""

        class Model(BaseModel):
            value: Int[Negative]

        assert Model(value=-1).value == -1

        with pytest.raises(ValidationError):
            Model(value=0)

        with pytest.raises(ValidationError):
            Model(value=1)

    def test_non_positive_constraint(self) -> None:
        """NonPositive constraint requires value <= 0."""

        class Model(BaseModel):
            value: Int[NonPositive]

        assert Model(value=0).value == 0
        assert Model(value=-1).value == -1

        with pytest.raises(ValidationError):
            Model(value=1)

    def test_multiple_of_constraint(self) -> None:
        """MultipleOf constraint requires value to be divisible."""

        class Model(BaseModel):
            value: Int[MultipleOf(5)]

        assert Model(value=0).value == 0
        assert Model(value=5).value == 5
        assert Model(value=10).value == 10

        with pytest.raises(ValidationError):
            Model(value=3)

    def test_combined_numeric_constraints(self) -> None:
        """Multiple numeric constraints can be combined."""

        class Model(BaseModel):
            value: Int[Gt(0), Lt(100), MultipleOf(5)]

        assert Model(value=5).value == 5
        assert Model(value=95).value == 95

        with pytest.raises(ValidationError):
            Model(value=0)  # Not > 0

        with pytest.raises(ValidationError):
            Model(value=100)  # Not < 100

        with pytest.raises(ValidationError):
            Model(value=7)  # Not multiple of 5

    def test_float_constraints(self) -> None:
        """Numeric constraints work with floats."""

        class Model(BaseModel):
            value: Float[Gt(0.0), Le(1.0)]

        assert Model(value=0.5).value == 0.5
        assert Model(value=1.0).value == 1.0

        with pytest.raises(ValidationError):
            Model(value=0.0)

        with pytest.raises(ValidationError):
            Model(value=1.1)


class TestLengthConstraints:
    """Tests for MinLen, MaxLen, Len."""

    def test_min_len_valid(self) -> None:
        """MinLen accepts strings at or above minimum length."""

        class Model(BaseModel):
            value: Str[MinLen(3)]

        assert Model(value="abc").value == "abc"
        assert Model(value="abcd").value == "abcd"

    def test_min_len_invalid(self) -> None:
        """MinLen rejects strings below minimum length."""

        class Model(BaseModel):
            value: Str[MinLen(3)]

        with pytest.raises(ValidationError):
            Model(value="ab")

        with pytest.raises(ValidationError):
            Model(value="")

    def test_max_len_valid(self) -> None:
        """MaxLen accepts strings at or below maximum length."""

        class Model(BaseModel):
            value: Str[MaxLen(5)]

        assert Model(value="").value == ""
        assert Model(value="abcde").value == "abcde"

    def test_max_len_invalid(self) -> None:
        """MaxLen rejects strings above maximum length."""

        class Model(BaseModel):
            value: Str[MaxLen(5)]

        with pytest.raises(ValidationError):
            Model(value="abcdef")

    def test_len_exact(self) -> None:
        """Len(n) requires exactly n characters."""

        class Model(BaseModel):
            value: Str[Len(5)]

        assert Model(value="abcde").value == "abcde"

        with pytest.raises(ValidationError):
            Model(value="abcd")

        with pytest.raises(ValidationError):
            Model(value="abcdef")

    def test_len_range(self) -> None:
        """Len(min, max) requires length in range."""

        class Model(BaseModel):
            value: Str[Len(2, 5)]

        assert Model(value="ab").value == "ab"
        assert Model(value="abcde").value == "abcde"

        with pytest.raises(ValidationError):
            Model(value="a")

        with pytest.raises(ValidationError):
            Model(value="abcdef")

    def test_combined_length_constraints(self) -> None:
        """MinLen and MaxLen can be combined."""

        class Model(BaseModel):
            value: Str[MinLen(2), MaxLen(5)]

        assert Model(value="ab").value == "ab"
        assert Model(value="abcde").value == "abcde"

        with pytest.raises(ValidationError):
            Model(value="a")

        with pytest.raises(ValidationError):
            Model(value="abcdef")


class TestStringFormatValidators:
    """Tests for Email, Url, NonEmpty, etc."""

    def test_email_valid(self) -> None:
        """Email validator accepts valid email addresses."""

        class Model(BaseModel):
            value: Str[Email]

        assert Model(value="test@example.com").value == "test@example.com"
        assert Model(value="user.name@domain.org").value == "user.name@domain.org"

    def test_email_invalid(self) -> None:
        """Email validator rejects invalid email addresses."""

        class Model(BaseModel):
            value: Str[Email]

        with pytest.raises(ValidationError):
            Model(value="not-an-email")

        with pytest.raises(ValidationError):
            Model(value="@domain.com")

        with pytest.raises(ValidationError):
            Model(value="user@")

    def test_url_valid(self) -> None:
        """Url validator accepts valid HTTP/HTTPS URLs."""

        class Model(BaseModel):
            value: Str[Url]

        assert Model(value="https://example.com").value == "https://example.com"
        assert Model(value="http://localhost:8080").value == "http://localhost:8080"
        assert (
            Model(value="https://example.com/path?q=1").value
            == "https://example.com/path?q=1"
        )

    def test_url_invalid(self) -> None:
        """Url validator rejects invalid URLs."""

        class Model(BaseModel):
            value: Str[Url]

        with pytest.raises(ValidationError):
            Model(value="not-a-url")

        with pytest.raises(ValidationError):
            Model(value="ftp://example.com")  # Not HTTP/HTTPS

    def test_non_empty_valid(self) -> None:
        """NonEmpty accepts non-empty, non-whitespace strings."""

        class Model(BaseModel):
            value: Str[NonEmpty]

        assert Model(value="hello").value == "hello"
        assert Model(value=" hello ").value == " hello "

    def test_non_empty_invalid(self) -> None:
        """NonEmpty rejects empty or whitespace-only strings."""

        class Model(BaseModel):
            value: Str[NonEmpty]

        with pytest.raises(ValidationError):
            Model(value="")

        with pytest.raises(ValidationError):
            Model(value="   ")

        with pytest.raises(ValidationError):
            Model(value="\t\n")

    def test_trimmed_transforms_string(self) -> None:
        """Trimmed strips leading/trailing whitespace."""

        class Model(BaseModel):
            value: Str[Trimmed]

        assert Model(value="  hello  ").value == "hello"
        assert Model(value="\t\nhello\t\n").value == "hello"
        assert Model(value="hello").value == "hello"

    def test_lowercase_transforms_string(self) -> None:
        """Lowercase converts string to lowercase."""

        class Model(BaseModel):
            value: Str[Lowercase]

        assert Model(value="HELLO").value == "hello"
        assert Model(value="Hello World").value == "hello world"
        assert Model(value="hello").value == "hello"

    def test_uppercase_transforms_string(self) -> None:
        """Uppercase converts string to uppercase."""

        class Model(BaseModel):
            value: Str[Uppercase]

        assert Model(value="hello").value == "HELLO"
        assert Model(value="Hello World").value == "HELLO WORLD"
        assert Model(value="HELLO").value == "HELLO"

    def test_pattern_valid(self) -> None:
        """Pattern validator accepts strings matching regex."""

        class Model(BaseModel):
            value: Str[Pattern(r"^\d{3}-\d{4}$")]

        assert Model(value="123-4567").value == "123-4567"

    def test_pattern_invalid(self) -> None:
        """Pattern validator rejects strings not matching regex."""

        class Model(BaseModel):
            value: Str[Pattern(r"^\d{3}-\d{4}$")]

        with pytest.raises(ValidationError):
            Model(value="1234567")

        with pytest.raises(ValidationError):
            Model(value="abc-defg")

    def test_combined_string_constraints(self) -> None:
        """Multiple string constraints can be combined in any order."""

        # Order 1: Trimmed first
        class Model1(BaseModel):
            value: Str[Trimmed, NonEmpty, MinLen(3)]

        # Order 2: MinLen first
        class Model2(BaseModel):
            value: Str[MinLen(3), Trimmed, NonEmpty]

        # Both should accept valid input and trim it
        assert Model1(value="  hello  ").value == "hello"
        assert Model2(value="  hello  ").value == "hello"

        # Both should reject whitespace-only (fails NonEmpty after trim)
        with pytest.raises(ValidationError):
            Model1(value="   ")

        with pytest.raises(ValidationError):
            Model2(value="   ")

        # Both should reject too-short input (fails MinLen)
        with pytest.raises(ValidationError):
            Model1(value="ab")

        with pytest.raises(ValidationError):
            Model2(value="ab")


class TestPreBuiltTypes:
    """Tests for EmailStr, HttpUrl, PositiveInt, etc."""

    def test_email_str(self) -> None:
        """EmailStr validates email addresses."""

        class Model(BaseModel):
            value: EmailStr

        assert Model(value="test@example.com").value == "test@example.com"

        with pytest.raises(ValidationError):
            Model(value="invalid")

    def test_http_url(self) -> None:
        """HttpUrl validates URLs."""

        class Model(BaseModel):
            value: HttpUrl

        assert Model(value="https://example.com").value == "https://example.com"

        with pytest.raises(ValidationError):
            Model(value="invalid")

    def test_non_empty_str(self) -> None:
        """NonEmptyStr rejects empty strings."""

        class Model(BaseModel):
            value: NonEmptyStr

        assert Model(value="hello").value == "hello"

        with pytest.raises(ValidationError):
            Model(value="")

    def test_trimmed_str(self) -> None:
        """TrimmedStr strips whitespace."""

        class Model(BaseModel):
            value: TrimmedStr

        assert Model(value="  hello  ").value == "hello"

    def test_positive_int(self) -> None:
        """PositiveInt requires value > 0."""

        class Model(BaseModel):
            value: PositiveInt

        assert Model(value=1).value == 1

        with pytest.raises(ValidationError):
            Model(value=0)

    def test_non_negative_int(self) -> None:
        """NonNegativeInt requires value >= 0."""

        class Model(BaseModel):
            value: NonNegativeInt

        assert Model(value=0).value == 0

        with pytest.raises(ValidationError):
            Model(value=-1)

    def test_negative_int(self) -> None:
        """NegativeInt requires value < 0."""

        class Model(BaseModel):
            value: NegativeInt

        assert Model(value=-1).value == -1

        with pytest.raises(ValidationError):
            Model(value=0)

    def test_positive_float(self) -> None:
        """PositiveFloat requires value > 0."""

        class Model(BaseModel):
            value: PositiveFloat

        assert Model(value=0.1).value == 0.1

        with pytest.raises(ValidationError):
            Model(value=0.0)

    def test_non_negative_float(self) -> None:
        """NonNegativeFloat requires value >= 0."""

        class Model(BaseModel):
            value: NonNegativeFloat

        assert Model(value=0.0).value == 0.0

        with pytest.raises(ValidationError):
            Model(value=-0.1)

    def test_negative_float(self) -> None:
        """NegativeFloat requires value < 0."""

        class Model(BaseModel):
            value: NegativeFloat

        assert Model(value=-0.1).value == -0.1

        with pytest.raises(ValidationError):
            Model(value=0.0)

    def test_percentage(self) -> None:
        """Percentage requires value in [0, 100]."""

        class Model(BaseModel):
            value: Percentage

        assert Model(value=0).value == 0
        assert Model(value=50.5).value == 50.5
        assert Model(value=100).value == 100

        with pytest.raises(ValidationError):
            Model(value=-1)

        with pytest.raises(ValidationError):
            Model(value=101)

    def test_unit_interval(self) -> None:
        """UnitInterval requires value in [0, 1]."""

        class Model(BaseModel):
            value: UnitInterval

        assert Model(value=0).value == 0
        assert Model(value=0.5).value == 0.5
        assert Model(value=1).value == 1

        with pytest.raises(ValidationError):
            Model(value=-0.1)

        with pytest.raises(ValidationError):
            Model(value=1.1)


class TestGetBaseType:
    """Tests for get_base_type utility function."""

    def test_annotated_type(self) -> None:
        """get_base_type extracts base from Annotated types."""
        assert get_base_type(Str[Email]) is str
        assert get_base_type(Int[Positive]) is int
        assert get_base_type(Float[Ge(0)]) is float

    def test_pre_built_types(self) -> None:
        """get_base_type works with pre-built type aliases."""
        assert get_base_type(EmailStr) is str
        assert get_base_type(HttpUrl) is str
        assert get_base_type(PositiveInt) is int
        assert get_base_type(Percentage) is float

    def test_plain_types(self) -> None:
        """get_base_type returns plain types as-is."""
        assert get_base_type(str) is str
        assert get_base_type(int) is int
        assert get_base_type(float) is float

    def test_optional_types(self) -> None:
        """get_base_type extracts base from Optional types."""
        assert get_base_type(str | None) is str
        assert get_base_type(int | None) is int

    def test_constrained_type_class(self) -> None:
        """get_base_type works with _ConstrainedType classes."""
        assert get_base_type(Str) is str
        assert get_base_type(Int) is int

    def test_unsupported_returns_none(self) -> None:
        """get_base_type returns None for unsupported types."""
        assert get_base_type(str | int) is None  # Union of non-None types


class TestIsSupportedType:
    """Tests for is_supported_type utility function."""

    def test_supported_base_types(self) -> None:
        """is_supported_type returns True for supported types."""
        assert is_supported_type(str) is True
        assert is_supported_type(int) is True
        assert is_supported_type(float) is True
        assert is_supported_type(bool) is True
        assert is_supported_type(bytes) is True

    def test_supported_datetime_types(self) -> None:
        """is_supported_type returns True for datetime types."""
        from datetime import date, datetime, time

        assert is_supported_type(datetime) is True
        assert is_supported_type(date) is True
        assert is_supported_type(time) is True

    def test_supported_constrained_types(self) -> None:
        """is_supported_type returns True for constrained types."""
        assert is_supported_type(Str[Email]) is True
        assert is_supported_type(Int[Positive]) is True
        assert is_supported_type(EmailStr) is True

    def test_supported_collection_types(self) -> None:
        """is_supported_type returns True for JSON-serializable types."""
        assert is_supported_type(list) is True
        assert is_supported_type(dict) is True

    def test_unsupported_types(self) -> None:
        """is_supported_type returns False for unsupported types."""
        assert is_supported_type(str | int) is False  # Non-optional union


class TestComplexModels:
    """Integration tests with complex Pydantic models."""

    def test_model_with_all_constraint_types(self) -> None:
        """Model with various constraint types validates correctly."""

        class UserProfile(BaseModel):
            username: Str[Trimmed, Lowercase, MinLen(3), MaxLen(20)]
            email: EmailStr
            age: Int[Ge(0), Le(150)]
            website: HttpUrl | None = None
            score: Percentage = 0.0

        user = UserProfile(
            username="  JohnDoe  ",
            email="john@example.com",
            age=30,
            website="https://johndoe.com",
            score=85.5,
        )

        assert user.username == "johndoe"  # Trimmed and lowercased
        assert user.email == "john@example.com"
        assert user.age == 30
        assert user.website == "https://johndoe.com"
        assert user.score == 85.5

    def test_model_validation_errors(self) -> None:
        """Model reports all validation errors."""

        class StrictModel(BaseModel):
            name: Str[NonEmpty, MinLen(2)]
            count: PositiveInt
            ratio: UnitInterval

        with pytest.raises(ValidationError) as exc_info:
            StrictModel(name="", count=-1, ratio=2.0)

        errors = exc_info.value.errors()
        assert len(errors) >= 3  # At least 3 validation errors

    def test_nested_models(self) -> None:
        """Constrained types work in nested models."""

        class Address(BaseModel):
            street: Str[NonEmpty]
            city: Str[NonEmpty]
            zip_code: Str[Pattern(r"^\d{5}$")]

        class Person(BaseModel):
            name: Str[NonEmpty]
            email: EmailStr
            address: Address

        person = Person(
            name="Alice",
            email="alice@example.com",
            address=Address(street="123 Main St", city="Boston", zip_code="02101"),
        )

        assert person.name == "Alice"
        assert person.address.zip_code == "02101"
