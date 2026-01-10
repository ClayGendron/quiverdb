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


class TestFromDict:
    """Tests for from_dict type detection and instantiation."""

    def test_from_dict_detects_type(self) -> None:
        """from_dict uses 'type' field to find the right subclass."""

        class DictNode(GraphModel, table=True, table_name="dict_nodes"):
            node: str = Field(primary_key=True)
            type: str
            title: str | None = None

        class Article(DictNode):
            title: str

        _test_classes.extend([DictNode, Article])

        instance = DictNode.from_dict({
            "node": "article-1",
            "type": "Article",
            "title": "My Article",
        })

        # Should validate against Article schema
        assert instance.type == "Article"
        assert instance.title == "My Article"

    def test_from_dict_validates_required_fields(self) -> None:
        """from_dict raises if required fields for the type are missing."""
        from pydantic import ValidationError

        class ValidNode(GraphModel, table=True, table_name="valid_nodes"):
            node: str = Field(primary_key=True)
            type: str
            email: str | None = None

        class Person(ValidNode):
            email: str  # required for Person

        _test_classes.extend([ValidNode, Person])

        with pytest.raises(ValidationError):
            ValidNode.from_dict({
                "node": "person-1",
                "type": "Person",
                # missing email
            })

    def test_from_dict_falls_back_to_base(self) -> None:
        """from_dict uses base class if type not in registry."""

        class FallbackNode(GraphModel, table=True, table_name="fallback_nodes"):
            node: str = Field(primary_key=True)
            type: str

        _test_classes.append(FallbackNode)

        instance = FallbackNode.from_dict({
            "node": "unknown-1",
            "type": "UnknownType",
        })

        assert instance.type == "UnknownType"

    def test_from_dict_auto_sets_type(self) -> None:
        """from_dict auto-sets type field if not provided."""

        class AutoTypeNode(GraphModel, table=True, table_name="auto_type_nodes"):
            node: str = Field(primary_key=True)
            type: str
            name: str | None = None

        class Widget(AutoTypeNode):
            name: str

        _test_classes.extend([AutoTypeNode, Widget])

        # Register Widget, then create via from_dict without explicit type
        # Since we can't auto-detect without type field, this tests the base case
        instance = AutoTypeNode.from_dict({
            "node": "widget-1",
        })

        assert instance.type == "AutoTypeNode"


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

    def test_from_dict_uses_custom_type_name(self) -> None:
        """from_dict recognizes custom type names from decorator."""

        class CustomNode(GraphModel, table=True, table_name="custom_type_nodes"):
            node: str = Field(primary_key=True)
            type: str
            role: str | None = None

        @graph_type("AdminUser")
        class Admin(CustomNode):
            role: str

        _test_classes.extend([CustomNode, Admin])

        instance = CustomNode.from_dict({
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
