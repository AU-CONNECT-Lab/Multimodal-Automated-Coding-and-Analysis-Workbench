"""MACAW -- Multimodal Automated Coding and Analysis Workbench."""

from macaw.config.loader import load_config, resolve_config
from macaw.config.schema import MacawConfig
from macaw.exceptions import MacawError
from macaw.extractors import get_extractor, list_extractors, register_extractor
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.manifest import FeatureSpec, FeatureManifest
from macaw.project.session import MediaFile, Session, Subject
from macaw.store.longtable import read_long_table, write_long_table
from macaw.store.schema import Scale

__version__ = "0.1.0-alpha"

__all__ = [
    "__version__",
    "MacawError",
    # config
    "MacawConfig",
    "load_config",
    "resolve_config",
    # project
    "Session",
    "Subject",
    "MediaFile",
    "FeatureSpec",
    "FeatureManifest",
    # store
    "Scale",
    "read_long_table",
    "write_long_table",
    # extractors
    "Extractor",
    "ExtractorResult",
    "register_extractor",
    "get_extractor",
    "list_extractors",
]
