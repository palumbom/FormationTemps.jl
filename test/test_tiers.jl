# Which tier of the suite runs. The fast tier is the default; FT_TEST_SLOW=1 in the environment, or
# Pkg.test(test_args=["slow"]), adds the slow tier (full-size references and convergence checks) to it.
run_slow_tests() = get(ENV, "FT_TEST_SLOW", "0") == "1" || "slow" in ARGS
