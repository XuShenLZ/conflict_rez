from confrez.rl import pklot_env
from confrez.rl.pklot_env import parallel_env
from confrez.rl.pklot_env import env as aec_env

from pettingzoo.test import (
    parallel_api_test,
    max_cycles_test,
    render_test,
    performance_benchmark,
    test_save_obs,
)
from pettingzoo.test.seed_test import parallel_seed_test

# parallel_api_test(parallel_env(), num_cycles=300)

# parallel_seed_test(parallel_env, num_cycles=10, test_kept_state=True)

# max_cycles_test(pklot_env)

render_test(aec_env)

# performance_benchmark(aec_env())

# test_save_obs(aec_env())
