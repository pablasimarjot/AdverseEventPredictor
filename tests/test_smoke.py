def test_import():
    import predictor
    assert hasattr(predictor, '__version__')
