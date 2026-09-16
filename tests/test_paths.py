"""
Tests for streetlight.paths module.

Verifies that path resolution works correctly on any machine.
"""

import tempfile
from pathlib import Path

from streetlight import paths


class TestProjectRoot:
    """Tests for project_root() function."""

    def test_project_root_returns_path(self):
        """Test that project_root returns a Path object."""
        root = paths.project_root()
        assert isinstance(root, Path)

    def test_project_root_exists(self):
        """Test that project_root points to an existing directory."""
        root = paths.project_root()
        assert root.exists()
        assert root.is_dir()

    def test_project_root_has_git(self):
        """Test that project_root contains .git directory."""
        root = paths.project_root()
        # Should have .git since we're in a git repo
        assert (root / ".git").exists()

    def test_project_root_env_override(self, monkeypatch):
        """Test that STREETLIGHT_ROOT environment variable overrides default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("STREETLIGHT_ROOT", tmpdir)
            root = paths.project_root()
            assert root == Path(tmpdir)


class TestSrcDir:
    """Tests for src_dir() function."""

    def test_src_dir_returns_path(self):
        """Test that src_dir returns a Path object."""
        src = paths.src_dir()
        assert isinstance(src, Path)

    def test_src_dir_exists(self):
        """Test that src_dir points to an existing directory."""
        src = paths.src_dir()
        assert src.exists()
        assert src.is_dir()


class TestDataDir:
    """Tests for data_dir() function."""

    def test_data_dir_returns_path(self):
        """Test that data_dir returns a Path object."""
        data = paths.data_dir()
        assert isinstance(data, Path)

    def test_data_dir_is_relative_to_root(self):
        """Test that data_dir is under project root."""
        root = paths.project_root()
        data = paths.data_dir()
        # Should be under project root or resolve to same
        assert data.is_relative_to(root) or data == root / "data"

    def test_data_dir_env_override(self, monkeypatch):
        """Test that STREETLIGHT_DATA_DIR environment variable overrides default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("STREETLIGHT_DATA_DIR", tmpdir)
            data = paths.data_dir()
            assert data == Path(tmpdir)


class TestConfigDir:
    """Tests for config_dir() function."""

    def test_config_dir_returns_path(self):
        """Test that config_dir returns a Path object."""
        config = paths.config_dir()
        assert isinstance(config, Path)

    def test_config_dir_exists(self):
        """Test that config_dir points to an existing directory."""
        config = paths.config_dir()
        assert config.exists()
        assert config.is_dir()


class TestOutputsDir:
    """Tests for outputs_dir() function."""

    def test_outputs_dir_returns_path(self):
        """Test that outputs_dir returns a Path object."""
        outputs = paths.outputs_dir()
        assert isinstance(outputs, Path)

    def test_outputs_dir_env_override(self, monkeypatch):
        """Test that STREETLIGHT_OUTPUTS_DIR environment variable overrides default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("STREETLIGHT_OUTPUTS_DIR", tmpdir)
            outputs = paths.outputs_dir()
            assert outputs == Path(tmpdir)

class TestScriptsDir:
    """Tests for scripts_dir() function."""

    def test_scripts_dir_returns_path(self):
        """Test that scripts_dir returns a Path object."""
        scripts = paths.scripts_dir()
        assert isinstance(scripts, Path)

class TestLogsDir:
    """Tests for logs_dir() function."""

    def test_logs_dir_returns_path(self):
        """Test that logs_dir returns a Path object."""
        logs = paths.logs_dir()
        assert isinstance(logs, Path)


class TestSpecializedPaths:
    """Tests for specialized path functions."""

    def test_power_data_dir(self):
        """Test that power_data_dir returns correct path."""
        power = paths.power_data_dir()
        assert isinstance(power, Path)
        assert "power" in str(power)

    def test_county_shapefile_path(self):
        """Test that county_shapefile_path returns correct path."""
        shp = paths.county_shapefile_path()
        assert isinstance(shp, Path)
        assert shp.suffix == ".shp"
        assert shp.name == "county_boundaries.shp"
        assert "data/geo" in shp.as_posix()

    def test_streetlight_list_dir(self):
        """Test that streetlight_list_dir returns correct path."""
        sl_list = paths.streetlight_list_dir()
        assert isinstance(sl_list, Path)
        assert "streetlight_list" in str(sl_list)

class TestEnsureDir:
    """Tests for ensure_dir() function."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test that ensure_dir creates a new directory."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()

        result = paths.ensure_dir(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_dir_existing_directory(self, tmp_path):
        """Test that ensure_dir works with existing directory."""
        result = paths.ensure_dir(tmp_path)
        assert result == tmp_path
        assert tmp_path.exists()
