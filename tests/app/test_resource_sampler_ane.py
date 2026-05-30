def test_parse_powermetrics_ane():
    from app.services.resource_sampler import _parse_ane_pct
    # Mock powermetrics output snippet
    sample = "ANE Energy: 1000 mW\nANE Resampler: 25.5%"
    assert _parse_ane_pct(sample) == 25.5
