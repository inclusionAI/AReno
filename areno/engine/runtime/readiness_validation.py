"""Input validation for readiness configuration.

Validates user inputs before expensive initialization to provide
clear, human-readable error messages.
"""

from __future__ import annotations

from typing import Any


class ValidationError(Exception):
    """Validation error with human-readable message."""

    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error: {field} - {message}. Got: {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured output."""
        return {
            "field": self.field,
            "value": str(self.value),
            "message": self.message,
        }


class ReadinessConfigValidator:
    """Validator for readiness configuration options."""

    # Valid ranges for configuration parameters
    TIMEOUT_MIN = 1
    TIMEOUT_MAX = 3600  # 1 hour
    PROBE_INTERVAL_MIN = 1
    PROBE_INTERVAL_MAX = 60

    def __init__(self):
        """Initialize validator."""
        self._errors: list[ValidationError] = []

    def validate_timeout(self, timeout: Any) -> int:
        """Validate readiness timeout value.

        Args:
            timeout: Timeout value (should be positive integer)

        Returns:
            Validated timeout in seconds

        Raises:
            ValidationError: If validation fails
        """
        return self._validate_positive_int(
            "readiness-timeout",
            timeout,
            self.TIMEOUT_MIN,
            self.TIMEOUT_MAX,
        )

    def validate_probe_interval(self, interval: Any) -> int:
        """Validate probe interval value.

        Args:
            interval: Interval value (should be positive integer)

        Returns:
            Validated interval in seconds

        Raises:
            ValidationError: If validation fails
        """
        return self._validate_positive_int(
            "readiness.probe_interval_seconds",
            interval,
            self.PROBE_INTERVAL_MIN,
            self.PROBE_INTERVAL_MAX,
        )

    def validate_enabled(self, enabled: Any) -> bool:
        """Validate enabled flag.

        Args:
            enabled: Enabled flag value

        Returns:
            Validated boolean value

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(enabled, bool):
            return enabled

        if isinstance(enabled, str):
            lower = enabled.lower()
            if lower in ("true", "1", "yes", "on"):
                return True
            if lower in ("false", "0", "no", "off"):
                return False

        raise ValidationError(
            "readiness.enabled",
            enabled,
            "must be a boolean or boolean-like string (true/false, 1/0, yes/no, on/off)",
        )

    def validate_config_dict(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate full readiness configuration dictionary.

        Args:
            config: Configuration dictionary from YAML/CLI

        Returns:
            Validated and normalized configuration

        Raises:
            ValidationError: If validation fails
        """
        result: dict[str, Any] = {}

        # Validate enabled
        if "enabled" in config:
            result["enabled"] = self.validate_enabled(config["enabled"])
        else:
            result["enabled"] = False

        # Validate timeout
        if "timeout_per_stage_seconds" in config:
            result["timeout_per_stage_seconds"] = self.validate_timeout(
                config["timeout_per_stage_seconds"]
            )
        else:
            result["timeout_per_stage_seconds"] = 30

        # Validate probe interval
        if "probe_interval_seconds" in config:
            result["probe_interval_seconds"] = self.validate_probe_interval(
                config["probe_interval_seconds"]
            )
        else:
            result["probe_interval_seconds"] = 2

        return result

    def _validate_positive_int(
        self,
        field: str,
        value: Any,
        min_val: int,
        max_val: int,
    ) -> int:
        """Validate a positive integer value.

        Args:
            field: Field name for error messages
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        # Try to convert to int
        if isinstance(value, bool):
            raise ValidationError(
                field,
                value,
                f"must be a positive integer (not a boolean)",
            )

        if isinstance(value, (int, float)):
            int_val = int(value)
            if int_val != value:  # Was a float with decimal part
                raise ValidationError(
                    field,
                    value,
                    f"must be a positive integer (not a float)",
                )
        elif isinstance(value, str):
            try:
                int_val = int(value)
            except ValueError:
                raise ValidationError(
                    field,
                    value,
                    f"must be a positive integer (invalid format)",
                )
        else:
            raise ValidationError(
                field,
                value,
                f"must be a positive integer (got {type(value).__name__})",
            )

        # Check range
        if int_val < min_val:
            raise ValidationError(
                field,
                value,
                f"must be at least {min_val}",
            )

        if int_val > max_val:
            raise ValidationError(
                field,
                value,
                f"must be at most {max_val}",
            )

        return int_val


def validate_readiness_options(
    enabled: bool | None = None,
    timeout: Any = None,
    probe_interval: Any = None,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate readiness options from CLI or config.

    This is the main entry point for validation. It validates all options
    and returns a normalized configuration dictionary.

    Args:
        enabled: Whether readiness is enabled (CLI flag)
        timeout: Timeout per stage in seconds (CLI option)
        probe_interval: Probe interval in seconds (CLI option)
        config_dict: Full configuration from YAML file

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If any validation fails
    """
    validator = ReadinessConfigValidator()

    # Start with config dict if provided
    if config_dict is not None:
        result = validator.validate_config_dict(config_dict)
    else:
        result = {"enabled": False, "timeout_per_stage_seconds": 30, "probe_interval_seconds": 2}

    # Override with CLI options
    if enabled is not None:
        result["enabled"] = validator.validate_enabled(enabled)

    if timeout is not None:
        result["timeout_per_stage_seconds"] = validator.validate_timeout(timeout)

    if probe_interval is not None:
        result["probe_interval_seconds"] = validator.validate_probe_interval(probe_interval)

    return result
