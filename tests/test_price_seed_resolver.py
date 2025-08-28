from utils.price_seed_resolver import PriceSeedResolver, DEFAULT_BASE_PRICE


def test_resolve_uses_user_override():
    resolver = PriceSeedResolver()
    seed = resolver.resolve("ES", user_override=12345.0)
    assert seed.base_price == 12345.0


def test_resolve_uses_market_spec_price():
    resolver = PriceSeedResolver()
    seed = resolver.resolve("ES")
    assert seed.base_price == 6340.0


def test_resolve_unknown_symbol_defaults():
    resolver = PriceSeedResolver()
    seed = resolver.resolve("UNKNOWN")
    assert seed.base_price == DEFAULT_BASE_PRICE


def test_get_supported_symbols_sorted_and_contains():
    resolver = PriceSeedResolver()
    supported = resolver.get_supported_symbols()
    assert supported == sorted(supported)
    assert "ES" in supported and "NQ" in supported
