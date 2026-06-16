__all__ = [
    "SRGCAUN",
    "build_srg_caun",
    "SRGCAUNHierMatch",
    "build_srg_caun_hier_match",
    "DRTBaseline",
    "build_drt_baseline",
]


def __getattr__(name):
    if name in {"SRGCAUN", "build_srg_caun"}:
        from .srg_caun import SRGCAUN, build_srg_caun
        return {"SRGCAUN": SRGCAUN, "build_srg_caun": build_srg_caun}[name]
    if name in {"SRGCAUNHierMatch", "build_srg_caun_hier_match"}:
        from .srg_caun_hier_match import SRGCAUNHierMatch, build_srg_caun_hier_match
        return {"SRGCAUNHierMatch": SRGCAUNHierMatch, "build_srg_caun_hier_match": build_srg_caun_hier_match}[name]
    if name in {"DRTBaseline", "build_drt_baseline"}:
        from .drt_baseline import DRTBaseline, build_drt_baseline
        return {"DRTBaseline": DRTBaseline, "build_drt_baseline": build_drt_baseline}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
