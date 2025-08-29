from .in_sample import test_in_sample
from .out_of_sample import test_out_of_sample
from .in_sample_permutation import test_in_sample_permutation
from .out_of_sample_permutation import test_out_of_sample_permutation
from .noise_injection import test_noise_injection
from .monte_carlo import test_monte_carlo
from .regime_testing import test_regime_testing

TEST_FUNCTIONS = {
    "in_sample": test_in_sample,
    "out_of_sample": test_out_of_sample,
    "in_sample_permutation": test_in_sample_permutation,
    "out_of_sample_permutation": test_out_of_sample_permutation,
    "noise_injection": test_noise_injection,
    "monte_carlo": test_monte_carlo,
    "regime_testing": test_regime_testing,
}

__all__ = ["TEST_FUNCTIONS"]
