"""GraphModel - SQLModel extension for graph-structured data."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
)

from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

# ============ CONTEXT VARIABLES ============

_graph_init: ContextVar[bool] = ContextVar("graph_init", default=True)


@contextmanager
def partial_graph_init() -> Generator[None, None, None]:
    """Skip full graph validation during ORM reconstruction."""
    token = _graph_init.set(False)
    try:
        yield
    finally:
        _graph_init.reset(token)


# ============ RESERVED NAMES ============

def _get_reserved_field_names() -> frozenset[str]:
    """Get field names reserved by SQLModel."""
    reserved: set[str] = set()
    for name in dir(SQLModel):
        if not name.startswith("_"):
            reserved.add(name)
    return frozenset(reserved)


RESERVED_FIELD_NAMES = _get_reserved_field_names()


# ============ HELPERS ============

def _is_table_model(cls: type[SQLModel]) -> bool:
    """Check if class is a SQLModel table."""
    return cls.model_config.get("table", False) is True


def _validate_reserved_field_names(cls: type) -> None:
    """Ensure class doesn't use reserved SQLModel field names."""
    own_annotations = getattr(cls, "__annotations__", {})
    for field_name in own_annotations:
        if field_name in RESERVED_FIELD_NAMES:
            raise TypeError(
                f"Field name '{field_name}' in '{cls.__name__}' is reserved by "
                f"SQLModel. Please choose a different field name."
            )


def _detect_model_type(cls: type) -> Literal["node", "edge"]:
    """Detect if class is a node or edge based on field presence."""
    fields: set[str] = set()
    for base in cls.__mro__:
        if hasattr(base, "__annotations__"):
            fields.update(base.__annotations__.keys())

    # Node detection takes precedence
    if "node" in fields:
        return "node"

    if "source" in fields and "target" in fields:
        return "edge"

    raise TypeError(
        f"Cannot determine model type for '{cls.__name__}'. "
        f"A node model must have a 'node' field and an edge model must have "
        f"'source' and 'target' fields."
    )


def _get_table_base(cls: type) -> type[SQLModel] | None:
    """Walk MRO to find the table base class."""
    for base in cls.__mro__[1:]:
        if hasattr(base, "_table_base") and base._table_base is not None:
            return base._table_base
    return None


def _validate_no_new_fields(cls: type[SQLModel], table_base: type[SQLModel]) -> None:
    """Ensure subclass doesn't introduce fields not on the table base."""
    table_fields = set(table_base.model_fields.keys())
    subclass_fields = set(cls.model_fields.keys())
    new_fields = subclass_fields - table_fields

    if new_fields:
        tablename = getattr(table_base, "__tablename__", table_base.__name__)
        raise TypeError(
            f"'{cls.__name__}' introduces new fields {new_fields} not present on "
            f"'{tablename}' table. Subclasses can only constrain "
            f"existing fields (e.g., make Optional fields required)."
        )


def _is_optional_annotation(annotation: Any) -> bool:
    """Check if annotation is Optional (Union with None)."""
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def _parent_is_table(bases: tuple[type, ...]) -> bool:
    """Check if any parent class is a SQLModel table."""
    for base in bases:
        config = getattr(base, "model_config", {})
        if config.get("table", False):
            return True
    return False


# ============ DECORATOR ============

def graph_type(name: str) -> Callable[[type[GraphModel]], type[GraphModel]]:
    """Decorator to set a custom type name for registry."""
    def decorator(cls: type[GraphModel]) -> type[GraphModel]:
        # Remove old registration (class name)
        old_name = cls.__name__
        if old_name in GraphModel._type_registry:
            del GraphModel._type_registry[old_name]
        # Set custom name and re-register
        cls._type_name = name
        GraphModel._type_registry[name] = cls
        return cls
    return decorator


# ============ METACLASS ============

class GraphModelMetaclass(SQLModelMetaclass):
    """Metaclass that extends SQLModel with graph-specific behavior."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> type:
        # Check if this will be a table class
        is_table = kwargs.get("table", False)

        # Set __tablename__ before SQLModel processes the class
        if is_table:
            if table_name:
                namespace["__tablename__"] = table_name
            elif "__tablename__" not in namespace:
                # Auto-detect default table name based on fields
                annotations = namespace.get("__annotations__", {})
                if "node" in annotations:
                    namespace["__tablename__"] = "nodes"
                elif "source" in annotations and "target" in annotations:
                    namespace["__tablename__"] = "edges"

        # For non-table subclasses of table classes, fix SQLModel/Pydantic integration
        # Issue: Subclasses inherit model_config["table"]=True from parent, causing
        # SQLAlchemy instrumentation issues. Also, parent's InstrumentedAttribute
        # becomes the default, making required fields appear optional.
        if not is_table and _parent_is_table(bases):
            # Explicitly mark as non-table to avoid SQLAlchemy instrumentation
            if "model_config" not in namespace:
                namespace["model_config"] = {}
            namespace["model_config"]["table"] = False

            # For any field with non-Optional annotation and no default value,
            # add Field(...) to properly mark it as required
            annotations = namespace.get("__annotations__", {})
            for field_name, annotation in annotations.items():
                if field_name in namespace:
                    continue  # Already has a value/default
                if _is_optional_annotation(annotation):
                    continue  # Optional fields don't need explicit required marker
                namespace[field_name] = Field(...)

        # Let SQLModel's metaclass create the class
        new_cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip setup for the base GraphModel class itself
        if name == "GraphModel" and bases == (SQLModel,):
            return new_cls

        # Validate reserved field names
        _validate_reserved_field_names(new_cls)

        # Detect model type (node vs edge)
        try:
            new_cls.model_type = _detect_model_type(new_cls)
        except TypeError:
            if is_table:
                # Table classes MUST have node or source/target fields
                raise
            # Skip for non-table classes without required fields

        if is_table:
            # This is a table class - mark it as the base
            new_cls._table_base = new_cls
        else:
            # This is a type subclass - inherit table base from parent
            parent_base = _get_table_base(new_cls)
            new_cls._table_base = parent_base

            if parent_base is not None:
                # Validate no new fields introduced
                _validate_no_new_fields(
                    cast("type[SQLModel]", new_cls), parent_base
                )

                # Register in type registry (on GraphModel class)
                type_name = getattr(new_cls, "_type_name", None) or name
                # Access registry through the GraphModel base class
                for base in new_cls.__mro__:
                    is_graph_model = base.__name__ == "GraphModel"
                    if is_graph_model and hasattr(base, "_type_registry"):
                        base._type_registry[type_name] = new_cls
                        break

        return new_cls


# ============ GRAPHMODEL ============

class GraphModel(SQLModel, metaclass=GraphModelMetaclass):
    """SQLModel extension for graph-structured data.

    Provides:
    - Automatic node/edge detection based on fields
    - Type registry for subclass validation
    - from_dict() for type-aware instantiation

    Node models must have a 'node' field.
    Edge models must have 'source' and 'target' fields.

    Usage:
        class Node(GraphModel, table=True):
            node: str = Field(primary_key=True)
            type: str

        class Person(Node):
            pass  # Registered as type "Person"
    """

    # Class variables with explicit typing
    model_type: ClassVar[Literal["node", "edge"]]
    _table_base: ClassVar[type[GraphModel] | None]
    _type_name: ClassVar[str | None]
    _type_registry: ClassVar[dict[str, type[GraphModel]]]

    _table_base = None
    _type_name = None
    _type_registry = {}  # Regular dict - caller must maintain references

    @classmethod
    def get_type_name(cls) -> str:
        """Get the registered type name for this class."""
        return cls._type_name or cls.__name__

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        strict: bool = True,
    ) -> GraphModel:
        """Create instance from dict, auto-detecting type.

        If 'type' key is present and matches a registered subclass,
        validates against that subclass's schema.

        Args:
            data: Dictionary of field values.
            strict: If True, raise on validation failure. If False, fall back to base.

        Returns:
            Instance of the appropriate GraphModel subclass.

        Raises:
            ValidationError: If required fields are missing for the detected type.
        """
        type_str = data.get("type")

        # Find the target class for validation
        target_cls: type[GraphModel]
        if type_str and type_str in cls._type_registry:
            target_cls = cls._type_registry[type_str]
        else:
            target_cls = cls

        # Get the table base for instantiation
        table_base = target_cls._table_base or target_cls
        table_fields = set(table_base.model_fields.keys())

        # Separate table fields from extras
        table_data: dict[str, Any] = {}
        extra_data: dict[str, Any] = {}

        for key, value in data.items():
            if key in table_fields:
                table_data[key] = value
            else:
                extra_data[key] = value

        # Store extras in properties if that field exists
        if extra_data and "properties" in table_fields:
            existing_props = table_data.get("properties", {})
            if isinstance(existing_props, dict):
                table_data["properties"] = {**existing_props, **extra_data}
            else:
                table_data["properties"] = extra_data

        # Auto-set type field if not provided
        if "type" in table_fields and "type" not in table_data:
            table_data["type"] = target_cls.get_type_name()

        # Build validation data (table fields + expanded properties)
        validation_data = {**table_data}
        if "properties" in table_data and isinstance(table_data["properties"], dict):
            validation_data.update(table_data["properties"])

        # Validate against target class schema if it's a subclass
        # This validates required fields, types, etc. per the subclass rules.
        # The result is discarded - we only use the table base for instantiation.
        if target_cls is not table_base:
            target_cls.__pydantic_validator__.validate_python(validation_data)

        # Create and return base table instance
        return table_base(**table_data)

    @classmethod
    def get_registered_types(cls) -> dict[str, type[GraphModel]]:
        """Get all registered type subclasses."""
        return dict(cls._type_registry)

    def __init__(self, **data: Any) -> None:
        """Initialize with optional graph validation."""
        if _graph_init.get():
            super().__init__(**data)
        else:
            # Partial init for ORM reconstruction
            super().__init__(**data)
