from importlib import import_module

__all__ = [
    "ParticleTracker",
    "ParticleTrackingResult",
    "OfflineParticleTracker",
    "OfflineParticleTrackerConfig",
]

_NAME_TO_MODULE = {
    "ParticleTracker": "dream2flow.flow.particle_trackers.base",
    "ParticleTrackingResult": "dream2flow.flow.particle_trackers.base",
    "OfflineParticleTracker": "dream2flow.flow.particle_trackers.offline_particle_tracker",
    "OfflineParticleTrackerConfig": "dream2flow.flow.particle_trackers.offline_particle_tracker",
}


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
