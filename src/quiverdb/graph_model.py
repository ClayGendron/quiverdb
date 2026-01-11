"""GraphModel - SQLModel extension for graph data with polymorphic inheritance."""

from __future__ import annotations

import contextlib
import warnings
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

from sqlalchemy import event
from sqlalchemy.orm import ORMExecuteState, Session
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.sql import Select

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


def _has_type_field(cls: type) -> bool:
    """Check if class has a 'type' field in its annotations."""
    for base in cls.__mro__:
        if hasattr(base, "__annotations__") and "type" in base.__annotations__:
            return True
    return False


def _get_polymorphic_entity(stmt: Any) -> tuple[Any, Any]:
    """Extract polymorphic subclass entity from a select statement."""
    # Check column_descriptions (covers select(Entity))
    for desc in stmt.column_descriptions:
        entity = desc.get("entity")
        if entity and hasattr(entity, "__mapper__"):
            mapper = entity.__mapper__
            if mapper.inherits and mapper.polymorphic_identity:
                return entity, mapper

    # Check froms (covers select_from(Entity))
    if hasattr(stmt, "froms"):
        for frm in stmt.froms:
            if hasattr(frm, "entity_namespace"):
                entity = frm.entity_namespace
                if hasattr(entity, "__mapper__"):
                    mapper = entity.__mapper__
                    if mapper.inherits and mapper.polymorphic_identity:
                        return entity, mapper

    return None, None


# ============ EVENT LISTENER ============

_event_listener_registered = False


def _register_polymorphic_event_listener() -> None:
    """Register SQLAlchemy event listener for auto-filtering polymorphic queries.

    When you query `select(Document)`, this automatically adds
    `WHERE type = 'Document'` to filter by the polymorphic identity.
    """
    global _event_listener_registered
    if _event_listener_registered:
        return
    _event_listener_registered = True

    @event.listens_for(Session, "do_orm_execute")
    def _add_polymorphic_filter(orm_execute_state: ORMExecuteState) -> None:
        if not orm_execute_state.is_select:
            return

        stmt = orm_execute_state.statement
        entity, mapper = _get_polymorphic_entity(stmt)

        if entity and mapper:
            select_stmt = cast("Select[Any]", stmt)
            new_stmt = select_stmt.where(
                mapper.polymorphic_on == mapper.polymorphic_identity
            )
            orm_execute_state.statement = new_stmt


# Register on module load
_register_polymorphic_event_listener()


# ============ DECORATOR ============


def graph_type(name: str) -> Callable[[type[GraphModel]], type[GraphModel]]:
    """Set a custom type name for a GraphModel subclass.

    This decorator must be applied at class definition time. SQLAlchemy's
    polymorphic identity cannot be changed after the mapper is configured.

    The custom name affects:
    - The type registry (for model_validate dispatch)
    - The polymorphic identity (for SELECT auto-filtering)
    - The auto-set 'type' field value on instantiation

    Usage:
        @graph_type("CustomName")
        class MyDocument(Node):
            pass

        doc = MyDocument(node="1")
        assert doc.type == "CustomName"

    Raises:
        TypeError: If applied to a non-subclass or if name is already registered.
    """

    def decorator(cls: type[GraphModel]) -> type[GraphModel]:
        old_name = cls.__name__
        table_base = cls._table_base

        # Validate this is a proper type subclass
        if table_base is None:
            raise TypeError(
                f"@graph_type can only be applied to subclasses of a table class, "
                f"not to '{cls.__name__}'."
            )

        # Check for duplicate names (but allow if it's the same class being renamed)
        if name in GraphModel._type_registry:
            existing_cls = GraphModel._type_registry[name]
            if existing_cls is not cls:
                raise TypeError(
                    f"Type name '{name}' is already registered by "
                    f"'{existing_cls.__name__}'."
                )

        # Update type registry: remove old name, add new name
        if old_name in GraphModel._type_registry:
            del GraphModel._type_registry[old_name]
        cls._type_name = name
        GraphModel._type_registry[name] = cls

        return cls

    return decorator


# ============ METACLASS ============


class GraphModelMetaclass(SQLModelMetaclass):
    """Metaclass that extends SQLModel with graph-specific behavior.

    Key features:
    - Auto-detects node vs edge models based on field names
    - Sets up SQLAlchemy polymorphic inheritance for type subclasses
    - Auto-configures __mapper_args__ for tables with 'type' field
    - Registers type subclasses for select() auto-filtering
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> type:
        is_table = kwargs.get("table", False)
        is_type_subclass = not is_table and _parent_is_table(bases)

        # Set __tablename__ before SQLModel processes the class
        if is_table:
            if table_name:
                namespace["__tablename__"] = table_name
            elif "__tablename__" not in namespace:
                annotations = namespace.get("__annotations__", {})
                if "node" in annotations:
                    namespace["__tablename__"] = "nodes"
                elif "source" in annotations and "target" in annotations:
                    namespace["__tablename__"] = "edges"

            # Auto-configure __mapper_args__ for polymorphic inheritance
            # if the table has a 'type' field
            annotations = namespace.get("__annotations__", {})
            if "type" in annotations and "__mapper_args__" not in namespace:
                namespace["__mapper_args__"] = {
                    "polymorphic_on": "type",
                    "polymorphic_identity": name,
                }

        # For type subclasses, mark required fields properly
        if is_type_subclass:
            annotations = namespace.get("__annotations__", {})
            for field_name, annotation in annotations.items():
                if field_name in namespace:
                    continue
                if _is_optional_annotation(annotation):
                    continue
                namespace[field_name] = Field(...)

        # Let SQLModel's metaclass create the class
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Field name .* shadows")
            new_cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip setup for the base GraphModel class itself
        if name == "GraphModel" and bases == (SQLModel,):
            return new_cls

        _validate_reserved_field_names(new_cls)

        # Detect model type (node vs edge)
        try:
            new_cls.model_type = _detect_model_type(new_cls)
        except TypeError:
            if is_table:
                raise
            # Skip for non-table classes without required fields

        if is_table:
            new_cls._table_base = new_cls
        else:
            parent_base = _get_table_base(new_cls)
            new_cls._table_base = parent_base

            if parent_base is not None:
                _validate_no_new_fields(cast("type[SQLModel]", new_cls), parent_base)

                # Get the type name (respects @graph_type decorator)
                type_name = getattr(new_cls, "_type_name", None) or name

                # Check for duplicate type names
                if type_name in GraphModel._type_registry:
                    existing_cls = GraphModel._type_registry[type_name]
                    raise TypeError(
                        f"Type name '{type_name}' is already registered by "
                        f"'{existing_cls.__name__}'. Use @graph_type('UniqueName') "
                        f"to set a different type name for '{name}'."
                    )

                # Register in type registry
                GraphModel._type_registry[type_name] = new_cls

                # Register with SQLAlchemy for polymorphic inheritance
                if _has_type_field(parent_base):
                    _register_polymorphic_subclass(new_cls, parent_base, type_name)

        return new_cls


def _register_polymorphic_subclass(
    cls: type, table_base: type[SQLModel], type_name: str
) -> None:
    """Register a type subclass with SQLAlchemy's polymorphic mapper."""
    # Suppress exceptions - class may already be mapped in test environments
    with contextlib.suppress(Exception):
        # Runtime attributes added by SQLAlchemy, not in type stubs
        registry = SQLModel._sa_registry  # type: ignore[attr-defined]
        table = table_base.__table__  # type: ignore[attr-defined]
        registry.map_imperatively(
            cls,
            table,
            inherits=table_base,
            polymorphic_identity=type_name,
        )


# ============ GRAPHMODEL ============


class GraphModel(SQLModel, metaclass=GraphModelMetaclass):
    """SQLModel extension for graph-structured data with polymorphic inheritance.

    Provides:
    - Automatic node/edge detection based on fields
    - Polymorphic inheritance: `select(Document)` auto-filters by type
    - Type registry for validation dispatch
    - Auto-set type field on subclass instantiation

    Node models must have a 'node' field.
    Edge models must have 'source' and 'target' fields.

    Usage:
        class Node(GraphModel, table=True):
            node: str = Field(primary_key=True)
            type: str
            name: str | None = None

        class Document(Node):
            name: str  # Make name required for documents

        # Create instances - type is auto-set
        doc = Document(node="doc-1", name="Report")
        assert doc.type == "Document"

        # Query with auto-filtering
        with Session(engine) as session:
            # Automatically adds WHERE type = 'Document'
            docs = session.exec(select(Document)).all()
    """

    model_type: ClassVar[Literal["node", "edge"]]
    _table_base: ClassVar[type[GraphModel] | None]
    _type_name: ClassVar[str | None]
    _type_registry: ClassVar[dict[str, type[GraphModel]]]

    _table_base = None
    _type_name = None
    _type_registry = {}

    def __init__(self, **data: Any) -> None:
        """Initialize GraphModel instance.

        For type subclasses, automatically sets the 'type' field
        to the class name (or @graph_type name) if not provided.
        """
        # Auto-set type for subclasses
        table_base = self.__class__._table_base
        if table_base is not None and table_base is not self.__class__:
            data.setdefault("type", self.__class__.get_type_name())

        super().__init__(**data)

    @classmethod
    def get_type_name(cls) -> str:
        """Get the registered type name for this class."""
        return cls._type_name or cls.__name__

    @classmethod
    def model_validate(  # type: ignore[override]
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
        update: dict[str, Any] | None = None,
    ) -> GraphModel:
        """Validate and create instance with type-aware dispatch.

        If 'type' key matches a registered subclass, dispatches to that
        class for validation. Delegates to SQLModel's validation to ensure
        proper Pydantic handling.

        Args:
            obj: Data to validate (dict or object with attributes).
            strict: Whether to enforce strict validation.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context for validation.
            update: Additional fields to update/override.

        Returns:
            Instance of the appropriate type class.

        Raises:
            ValidationError: If validation fails.
        """
        # Determine type for dispatch
        if isinstance(obj, dict):
            merged = {**obj, **(update or {})}
            type_str = merged.get("type")
        else:
            type_str = getattr(obj, "type", None)

        # Find target class based on type
        if type_str and type_str in cls._type_registry:
            target_cls = cls._type_registry[type_str]
        else:
            target_cls = cls

        # Delegate to SQLModel's model_validate for proper Pydantic handling
        return SQLModel.model_validate.__func__(
            target_cls,
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            update=update,
        )

    @classmethod
    def get_registered_types(cls) -> dict[str, type[GraphModel]]:
        """Get all registered type subclasses."""
        return dict(cls._type_registry)
