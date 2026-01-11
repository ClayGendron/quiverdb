"""Tests for GraphModel behavior."""


import pytest
from sqlmodel import Field

from quiverdb.graph_model import (
    GraphModel,
    graph_type,
)

# Keep references to test classes to prevent WeakValueDictionary GC
_test_classes: list = []


class TestModelTypeDetection:
    """Tests for node vs edge detection based on fields."""

    def test_detects_node_from_node_field(self) -> None:
        """Class with 'node' field is detected as node type."""

        class TestNode(GraphModel, table=True, table_name="test_nodes_1"):
            node: str = Field(primary_key=True)
            type: str

        _test_classes.append(TestNode)
        assert TestNode.model_type == "node"

    def test_detects_edge_from_source_target_fields(self) -> None:
        """Class with 'source' and 'target' fields is detected as edge type."""

        class TestEdge(GraphModel, table=True, table_name="test_edges_1"):
            source: str = Field(primary_key=True)
            target: str = Field(primary_key=True)
            type: str

        _test_classes.append(TestEdge)
        assert TestEdge.model_type == "edge"

    def test_node_takes_precedence_over_edge(self) -> None:
        """If both node and source/target exist, node takes precedence."""

        class Hybrid(GraphModel, table=True, table_name="hybrid_1"):
            node: str = Field(primary_key=True)
            source: str
            target: str

        _test_classes.append(Hybrid)
        assert Hybrid.model_type == "node"

    def test_raises_for_unknown_model_type(self) -> None:
        """Raises TypeError if neither node nor source/target fields exist."""
        with pytest.raises(TypeError, match="Cannot determine model type"):

            class Invalid(GraphModel, table=True, table_name="invalid_1"):
                id: int = Field(primary_key=True)
                name: str


class TestReservedFieldNames:
    """Tests for reserved field name validation."""

    def test_rejects_reserved_field_names(self) -> None:
        """Fields that conflict with SQLModel attributes are rejected."""
        with pytest.raises(TypeError, match="is reserved by SQLModel"):

            class BadModel(GraphModel, table=True, table_name="bad_1"):
                node: str = Field(primary_key=True)
                metadata: str  # 'metadata' is reserved by SQLModel

    def test_allows_valid_field_names(self) -> None:
        """Normal field names are allowed."""

        class GoodModel(GraphModel, table=True, table_name="good_1"):
            node: str = Field(primary_key=True)
            type: str
            title: str | None = None
            content: str | None = None

        _test_classes.append(GoodModel)
        assert GoodModel.model_type == "node"


class TestTableName:
    """Tests for table name assignment."""

    def test_explicit_table_name_via_parameter(self) -> None:
        """Table name can be set via 'table_name' parameter."""

        class MyNode(GraphModel, table=True, table_name="custom_nodes"):
            node: str = Field(primary_key=True)

        _test_classes.append(MyNode)
        assert MyNode.__tablename__ == "custom_nodes"

    def test_default_table_name_for_node(self) -> None:
        """Nodes default to 'nodes' table name."""

        class DefaultNode(GraphModel, table=True):
            node: str = Field(primary_key=True)

        _test_classes.append(DefaultNode)
        assert DefaultNode.__tablename__ == "nodes"

    def test_default_table_name_for_edge(self) -> None:
        """Edges default to 'edges' table name."""

        class DefaultEdge(GraphModel, table=True):
            source: str = Field(primary_key=True)
            target: str = Field(primary_key=True)

        _test_classes.append(DefaultEdge)
        assert DefaultEdge.__tablename__ == "edges"


class TestTypeRegistry:
    """Tests for type subclass registration."""

    def test_subclass_registered_by_class_name(self) -> None:
        """Non-table subclasses are registered in _type_registry."""

        class BaseNode(GraphModel, table=True, table_name="reg_test_nodes"):
            node: str = Field(primary_key=True)
            type: str
            title: str | None = None

        class Document(BaseNode):
            title: str  # make required

        _test_classes.extend([BaseNode, Document])
        assert "Document" in GraphModel._type_registry
        assert GraphModel._type_registry["Document"] is Document

    def test_table_classes_not_in_registry(self) -> None:
        """Table classes (table=True) are not added to _type_registry."""

        class TableNode(GraphModel, table=True, table_name="table_reg_nodes"):
            node: str = Field(primary_key=True)

        _test_classes.append(TableNode)
        assert "TableNode" not in GraphModel._type_registry


class TestSubclassFieldValidation:
    """Tests that subclasses cannot introduce new fields."""

    def test_subclass_can_make_optional_required(self) -> None:
        """Subclasses can make Optional fields required."""

        class FieldNode(GraphModel, table=True, table_name="field_test_nodes"):
            node: str = Field(primary_key=True)
            type: str
            title: str | None = None

        class StrictDoc(FieldNode):
            title: str  # now required

        _test_classes.extend([FieldNode, StrictDoc])
        # Should not raise
        assert StrictDoc.model_type == "node"

    def test_subclass_cannot_add_new_fields(self) -> None:
        """Subclasses cannot introduce fields not on the base table."""

        class ConstrainedNode(GraphModel, table=True, table_name="constrained_nodes"):
            node: str = Field(primary_key=True)
            type: str

        _test_classes.append(ConstrainedNode)

        with pytest.raises(TypeError, match="introduces new fields"):

            class BadSubclass(ConstrainedNode):
                new_field: str  # not on base table


class TestAutoTypeField:
    """Tests for automatic type field setting in subclass instantiation."""

    def test_subclass_auto_sets_type_field(self) -> None:
        """Instantiating a type subclass auto-sets the type field."""

        class AutoNode(GraphModel, table=True, table_name="auto_nodes"):
            node: str = Field(primary_key=True)
            type: str
            name: str | None = None

        class AutoDocument(AutoNode):
            name: str

        _test_classes.extend([AutoNode, AutoDocument])

        doc = AutoDocument(node="doc-1", name="Report")
        assert doc.type == "AutoDocument"

    def test_direct_instantiation_follows_sqlmodel_behavior(self) -> None:
        """Direct instantiation doesn't validate (SQLModel table behavior).

        Use model_validate() for validation. Direct instantiation of table
        classes bypasses Pydantic validation - this is standard SQLModel behavior.
        """

        class DirectNode(GraphModel, table=True, table_name="direct_nodes"):
            node: str = Field(primary_key=True)
            type: str
            name: str | None = None

        class DirectStrictDoc(DirectNode):
            name: str  # required in schema, but direct instantiation won't validate

        _test_classes.extend([DirectNode, DirectStrictDoc])

        # Direct instantiation doesn't raise - this is SQLModel behavior
        doc = DirectStrictDoc(node="doc-1")  # missing name, but no validation
        assert doc.type == "DirectStrictDoc"
        assert doc.name is None  # field is missing, defaulted to None

    def test_subclass_explicit_type_preserved(self) -> None:
        """Explicit type value is preserved when provided."""

        class ExplicitNode(GraphModel, table=True, table_name="explicit_nodes"):
            node: str = Field(primary_key=True)
            type: str
            name: str | None = None

        class Report(ExplicitNode):
            name: str

        _test_classes.extend([ExplicitNode, Report])

        # Explicit type should be preserved
        doc = Report(node="doc-1", name="Test", type="CustomType")
        assert doc.type == "CustomType"

    def test_custom_type_name_decorator_auto_sets(self) -> None:
        """@graph_type decorator name is used for auto-set type field."""

        class DecNode(GraphModel, table=True, table_name="dec_nodes"):
            node: str = Field(primary_key=True)
            type: str

        @graph_type("MyCustomType")
        class CustomDoc(DecNode):
            pass

        _test_classes.extend([DecNode, CustomDoc])

        doc = CustomDoc(node="doc-1")
        assert doc.type == "MyCustomType"


class TestModelValidate:
    """Tests for model_validate type detection and instantiation."""

    def test_model_validate_detects_type(self) -> None:
        """model_validate uses 'type' field to find the right subclass."""

        class ValidateNode(GraphModel, table=True, table_name="validate_nodes"):
            node: str = Field(primary_key=True)
            type: str
            title: str | None = None

        class Paper(ValidateNode):
            title: str

        _test_classes.extend([ValidateNode, Paper])

        instance = ValidateNode.model_validate({
            "node": "paper-1",
            "type": "Paper",
            "title": "My Paper",
        })

        assert instance.type == "Paper"
        assert instance.title == "My Paper"

    def test_model_validate_validates_required_fields(self) -> None:
        """model_validate raises if required fields for the type are missing."""
        from pydantic import ValidationError

        class ValidNode(GraphModel, table=True, table_name="valid_nodes"):
            node: str = Field(primary_key=True)
            type: str
            email: str | None = None

        class Person(ValidNode):
            email: str  # required for Person

        _test_classes.extend([ValidNode, Person])

        with pytest.raises(ValidationError):
            ValidNode.model_validate({
                "node": "person-1",
                "type": "Person",
                # missing email
            })

    def test_model_validate_falls_back_to_base(self) -> None:
        """model_validate uses base class if type not in registry."""

        class FallbackNode(GraphModel, table=True, table_name="fallback_nodes"):
            node: str = Field(primary_key=True)
            type: str

        _test_classes.append(FallbackNode)

        instance = FallbackNode.model_validate({
            "node": "unknown-1",
            "type": "UnknownType",
        })

        assert instance.type == "UnknownType"

    def test_model_validate_leaves_type_unset_for_base(self) -> None:
        """model_validate does NOT auto-set type when called on base class."""

        class BaseTypeNode(GraphModel, table=True, table_name="base_type_nodes"):
            node: str = Field(primary_key=True)
            type: str | None = None

        _test_classes.append(BaseTypeNode)

        instance = BaseTypeNode.model_validate({
            "node": "node-1",
        })

        assert instance.type is None


class TestGraphTypeDecorator:
    """Tests for @graph_type decorator custom naming."""

    def test_custom_type_name_via_decorator(self) -> None:
        """@graph_type decorator sets custom registration name."""

        class DecoratorNode(GraphModel, table=True, table_name="decorator_nodes"):
            node: str = Field(primary_key=True)
            type: str

        @graph_type("CustomPersonType")
        class PersonWithCustomName(DecoratorNode):
            pass

        _test_classes.extend([DecoratorNode, PersonWithCustomName])

        assert "CustomPersonType" in GraphModel._type_registry
        assert "PersonWithCustomName" not in GraphModel._type_registry

    def test_model_validate_uses_custom_type_name(self) -> None:
        """model_validate recognizes custom type names from decorator."""

        class CustomNode(GraphModel, table=True, table_name="custom_type_nodes"):
            node: str = Field(primary_key=True)
            type: str
            role: str | None = None

        @graph_type("AdminUser")
        class Admin(CustomNode):
            role: str

        _test_classes.extend([CustomNode, Admin])

        instance = CustomNode.model_validate({
            "node": "admin-1",
            "type": "AdminUser",
            "role": "superuser",
        })

        assert instance.type == "AdminUser"
        assert instance.role == "superuser"


class TestGetRegisteredTypes:
    """Tests for get_registered_types helper."""

    def test_returns_all_registered_types(self) -> None:
        """get_registered_types returns dict of all registered subclasses."""

        class RegistryNode(GraphModel, table=True, table_name="registry_nodes"):
            node: str = Field(primary_key=True)
            type: str

        class TypeA(RegistryNode):
            pass

        class TypeB(RegistryNode):
            pass

        _test_classes.extend([RegistryNode, TypeA, TypeB])

        registered = GraphModel.get_registered_types()
        assert "TypeA" in registered
        assert "TypeB" in registered


class TestDuplicateTypeDetection:
    """Tests for duplicate type name detection."""

    def test_duplicate_type_name_raises_error(self) -> None:
        """Registering the same type name twice raises TypeError."""

        class DupNode(GraphModel, table=True, table_name="dup_nodes"):
            node: str = Field(primary_key=True)
            type: str

        class UniqueType(DupNode):
            pass

        _test_classes.extend([DupNode, UniqueType])

        # Attempting to register another class with same name should fail
        with pytest.raises(TypeError, match="already registered"):

            class UniqueType(DupNode):  # noqa: F811 - intentional redefinition
                pass

    def test_graph_type_decorator_duplicate_raises_error(self) -> None:
        """@graph_type with existing name raises TypeError."""

        class DecDupNode(GraphModel, table=True, table_name="dec_dup_nodes"):
            node: str = Field(primary_key=True)
            type: str

        class FirstType(DecDupNode):
            pass

        _test_classes.extend([DecDupNode, FirstType])

        # Trying to use @graph_type with an existing name should fail
        with pytest.raises(TypeError, match="already registered"):

            @graph_type("FirstType")
            class SecondType(DecDupNode):
                pass
