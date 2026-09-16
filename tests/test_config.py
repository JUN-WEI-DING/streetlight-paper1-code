"""Test configuration module."""

from streetlight.config.base import Config, get_config


def test_config_singleton():
    """Test that Config implements singleton pattern."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


def test_get_config():
    """Test get_config function."""
    config = get_config()
    assert isinstance(config, Config)


def test_config_get():
    """Test Config.get method with dot notation."""
    config = Config()

    # Test getting existing value
    value = config.get('emission_factors.Coal')
    assert value == 0.820

    # Test getting nested value
    value = config.get('simulation.standardized_installation.n_lights')
    assert value == 64

    # Test getting non-existent value with default
    value = config.get('nonexistent.key', 'default')
    assert value == 'default'


def test_config_data_dir():
    """Test Config.data_dir property."""
    config = Config()
    assert config.data_dir is not None


def test_config_power_dir():
    """Test Config.power_dir property."""
    config = Config()
    assert config.power_dir is not None


def test_config_emission_factors():
    """Test Config.emission_factors property."""
    config = Config()
    ef = config.emission_factors
    assert isinstance(ef, dict)
    assert 'Coal' in ef
    assert 'LNG' in ef
    assert 'Solar' in ef


def test_config_regions():
    """Test Config.regions property."""
    config = Config()
    regions = config.regions
    assert isinstance(regions, list)
    assert len(regions) == 7
    assert 'north' in regions
    assert 'island_penghu' in regions
    assert 'island_kinmen' in regions
    assert 'island_lienchiang' in regions


def test_config_region_mapping():
    """Test Config.region_mapping property."""
    config = Config()
    mapping = config.region_mapping
    assert isinstance(mapping, dict)
    assert mapping['north'] == 'N'
    assert mapping['central'] == 'C'
    assert mapping['island_penghu'] == 'IP'
    assert mapping['island_kinmen'] == 'IK'
    assert mapping['island_lienchiang'] == 'IL'
