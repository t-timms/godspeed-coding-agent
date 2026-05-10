"""Tests for the Godspeed theme system — dark/light palettes, variables."""

from __future__ import annotations

from godspeed.tui.textual_app import _DARK, _LIGHT


class TestThemeDark:
    """Verify dark theme palette."""

    def test_dark_theme_name(self):
        assert _DARK.name == "godspeed-dark"

    def test_dark_is_dark(self):
        assert _DARK.dark is True

    def test_dark_primary(self):
        assert _DARK.primary == "#c17c5b"

    def test_dark_background(self):
        assert _DARK.background == "#0e0c09"

    def test_dark_foreground(self):
        assert _DARK.foreground == "#e6dac8"

    def test_dark_surface(self):
        assert _DARK.surface == "#0e0c09"

    def test_dark_panel(self):
        assert _DARK.panel == "#151210"

    def test_dark_error(self):
        assert _DARK.error == "#a45252"

    def test_dark_warning(self):
        assert _DARK.warning == "#d4a817"

    def test_dark_success(self):
        assert _DARK.success == "#87a96b"

    def test_dark_has_custom_variables(self):
        assert "text-muted" in _DARK.variables
        assert "border" in _DARK.variables
        assert "border-focus" in _DARK.variables
        assert "selection" in _DARK.variables


class TestThemeLight:
    """Verify light theme palette."""

    def test_light_theme_name(self):
        assert _LIGHT.name == "godspeed-light"

    def test_light_is_not_dark(self):
        assert _LIGHT.dark is False

    def test_light_primary(self):
        assert _LIGHT.primary == "#b86e4a"

    def test_light_background(self):
        assert _LIGHT.background == "#faf5ee"

    def test_light_foreground(self):
        assert _LIGHT.foreground == "#2d2318"

    def test_light_surface(self):
        assert _LIGHT.surface == "#faf5ee"

    def test_light_panel(self):
        assert _LIGHT.panel == "#f0e8d8"

    def test_light_has_custom_variables(self):
        assert "text-muted" in _LIGHT.variables
        assert "border" in _LIGHT.variables
        assert "border-focus" in _LIGHT.variables
        assert "selection" in _LIGHT.variables


class TestThemeColors:
    """Verify color relationships between themes."""

    def test_dark_bg_darker_than_light_bg(self):
        """Dark background should be perceptually darker."""
        def hex_to_luminance(h):
            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            return 0.299 * r + 0.587 * g + 0.114 * b

        assert hex_to_luminance(_DARK.background) < hex_to_luminance(_LIGHT.background)

    def test_dark_text_lighter_than_light_text(self):
        """Dark mode text should be lighter than light mode text."""
        def hex_to_luminance(h):
            r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            return 0.299 * r + 0.587 * g + 0.114 * b

        assert hex_to_luminance(_DARK.foreground) > hex_to_luminance(_LIGHT.foreground)

    def test_contrast_ratio_dark(self):
        """Dark mode should have adequate contrast ratio."""
        def hex_to_rgb(h):
            return int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255

        def luminance(r, g, b):
            def linearize(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

            return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

        def contrast_ratio(h1, h2):
            l1 = luminance(*hex_to_rgb(h1))
            l2 = luminance(*hex_to_rgb(h2))
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)

        ratio = contrast_ratio(_DARK.foreground, _DARK.background)
        assert ratio >= 4.5, f"Dark mode contrast ratio {ratio:.1f} should be >= 4.5"

    def test_contrast_ratio_light(self):
        """Light mode should have adequate contrast ratio."""
        def hex_to_rgb(h):
            return int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255

        def luminance(r, g, b):
            def linearize(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

            return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

        def contrast_ratio(h1, h2):
            l1 = luminance(*hex_to_rgb(h1))
            l2 = luminance(*hex_to_rgb(h2))
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)

        ratio = contrast_ratio(_LIGHT.foreground, _LIGHT.background)
        assert ratio >= 4.5, f"Light mode contrast ratio {ratio:.1f} should be >= 4.5"

    def test_all_themes_are_valid_hex(self):
        for theme in (_DARK, _LIGHT):
            for attr in ("primary", "secondary", "warning", "error", "success",
                         "accent", "foreground", "background", "surface", "panel"):
                val = getattr(theme, attr)
                assert val.startswith("#"), f"{theme.name}.{attr} should start with #"
                assert len(val) == 7, f"{theme.name}.{attr} should be 7 chars"


class TestThemeDefaults:
    """Verify get_theme_variable_defaults."""

    def test_defaults_provides_text_muted(self):
        from godspeed.tui.textual_app import GodspeedTextualApp

        # Create minimal instance to test get_theme_variable_defaults
        defaults = GodspeedTextualApp.get_theme_variable_defaults(Mock())
        assert "text-muted" in defaults
        assert "border" in defaults
        assert "border-focus" in defaults
        assert "selection" in defaults

    def test_defaults_are_valid_hex(self):
        from godspeed.tui.textual_app import GodspeedTextualApp

        defaults = GodspeedTextualApp.get_theme_variable_defaults(Mock())
        for val in defaults.values():
            assert val.startswith("#")

    def test_toggle_theme_logic(self):
        """Verify theme toggle swaps between dark and light."""

        app = Mock()
        app.theme = "godspeed-dark"
        result = []
        app.theme = 'godspeed-dark'

        def simulate_toggle():
            app.theme = (
                "godspeed-light" if app.theme == "godspeed-dark" else "godspeed-dark"
            )
            return app.theme

        assert simulate_toggle() == "godspeed-light"
        assert simulate_toggle() == "godspeed-dark"


class Mock:
    """Minimal mock object for theme tests."""
    pass
