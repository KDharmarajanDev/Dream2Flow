from importlib import import_module

__all__ = [
    "DepthExtractor",
    "DepthExtractorConfig",
    "FileDepthExtractor",
    "FileDepthExtractorConfig",
    "SpaTrackerDepthExtractor",
    "SpaTrackerDepthExtractorConfig",
    "VideoFlowGenerator",
    "VideoFlowGeneratorConfig",
    "ObjectFlow",
    "ObjectFlowResult",
    "ParticleTracker",
    "ParticleTrackingResult",
    "OfflineParticleTracker",
    "OfflineParticleTrackerConfig",
    "GroundedRegionSelector",
    "GroundedRegionSelectorConfig",
]

_NAME_TO_MODULE = {
    "DepthExtractor": "dream2flow.flow.depth",
    "DepthExtractorConfig": "dream2flow.flow.depth",
    "FileDepthExtractor": "dream2flow.flow.depth",
    "FileDepthExtractorConfig": "dream2flow.flow.depth",
    "SpaTrackerDepthExtractor": "dream2flow.flow.depth",
    "SpaTrackerDepthExtractorConfig": "dream2flow.flow.depth",
    "VideoFlowGenerator": "dream2flow.flow.generators.video_flow_generator",
    "VideoFlowGeneratorConfig": "dream2flow.flow.generators.video_flow_generator",
    "ObjectFlow": "dream2flow.flow.object_flow",
    "ObjectFlowResult": "dream2flow.flow.object_flow_result",
    "ParticleTracker": "dream2flow.flow.particle_trackers",
    "ParticleTrackingResult": "dream2flow.flow.particle_trackers",
    "OfflineParticleTracker": "dream2flow.flow.particle_trackers",
    "OfflineParticleTrackerConfig": "dream2flow.flow.particle_trackers",
    "GroundedRegionSelector": "dream2flow.flow.region_selectors",
    "GroundedRegionSelectorConfig": "dream2flow.flow.region_selectors",
}


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
