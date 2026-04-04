"""Custom exceptions for the data processing and ML pipeline."""


class ProcessorError(Exception):
    """Base exception for failures in loading, cleaning, or validating data."""


class DataLoadError(ProcessorError):
    """Raised when an uploaded file cannot be read, parsed, or is invalid."""


class PreprocessingError(ProcessorError):
    """Raised when a DataFrame fails validation or cannot be cleaned."""


class AIModelError(Exception):
    """Raised when scaling, fitting, or inference in the ML engine fails."""
