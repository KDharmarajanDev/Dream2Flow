from importlib import import_module

__all__ = [
    "GroundedRegionSelector",
    "GroundedRegionSelectorConfig",
    "InstructionProcessor",
]

_NAME_TO_MODULE = {
    "GroundedRegionSelector": "dream2flow.flow.region_selectors.grounded_region_selector",
    "GroundedRegionSelectorConfig": "dream2flow.flow.region_selectors.grounded_region_selector",
    "InstructionProcessor": "dream2flow.flow.region_selectors.instruction_processor",
}


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
