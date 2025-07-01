def test_basic_imports():
    import importlib

    for mod in [
        "src.features",
        "src.adaptive_k",
        "src.refine_centroid",
        "src.compress",
        "src.metrics",
    ]:
        assert importlib.import_module(mod) 