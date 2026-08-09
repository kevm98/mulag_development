from ROM_pipeline.data import DEFAULT_SCHEMA


def test_schema_dimensions_and_ordering():
    DEFAULT_SCHEMA.validate()
    assert DEFAULT_SCHEMA.input_dim == 24
    assert DEFAULT_SCHEMA.command_dim == 8
    assert DEFAULT_SCHEMA.target_dim == 16
    assert DEFAULT_SCHEMA.input_feature_names[16] == "DesiredDeltaExtension_1"
