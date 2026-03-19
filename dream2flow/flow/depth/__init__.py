from importlib import import_module

__all__ = [
    "DepthExtractor",
    "DepthExtractorConfig",
    "FileDepthExtractor",
    "FileDepthExtractorConfig",
    "SpaTrackerDepthExtractor",
    "SpaTrackerDepthExtractorConfig",
]

_NAME_TO_MODULE = {
    "DepthExtractor": "dream2flow.flow.depth.base",
    "DepthExtractorConfig": "dream2flow.flow.depth.base",
    "FileDepthExtractor": "dream2flow.flow.depth.file_depth_extractor",
    "FileDepthExtractorConfig": "dream2flow.flow.depth.file_depth_extractor",
    "SpaTrackerDepthExtractor": "dream2flow.flow.depth.spatracker_depth_extractor",
    "SpaTrackerDepthExtractorConfig": "dream2flow.flow.depth.spatracker_depth_extractor",
}


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
