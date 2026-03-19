from importlib import import_module

__all__ = [
    "ObjectFlowGenerator",
    "PlaybackGenerator",
    "VideoFlowGenerator",
    "VideoFlowGeneratorConfig",
]

_NAME_TO_MODULE = {
    "ObjectFlowGenerator": "dream2flow.flow.generators.base",
    "PlaybackGenerator": "dream2flow.flow.generators.playback_generator",
    "VideoFlowGenerator": "dream2flow.flow.generators.video_flow_generator",
    "VideoFlowGeneratorConfig": "dream2flow.flow.generators.video_flow_generator",
}


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
