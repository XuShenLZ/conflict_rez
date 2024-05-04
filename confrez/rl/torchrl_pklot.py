import copy
import tempfile

import torch

from matplotlib import pyplot as plt
from tensordict import TensorDictBase

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

from torchrl.envs import (
    check_env_specs,
    ExplorationType,
    PettingZooWrapper,
    RewardSum,
    set_exploration_type,
    TransformedEnv,
    VmasEnv,
    PermuteTransform
)

from torchrl.modules import (
    AdditiveGaussianWrapper,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    MultiAgentConvNet
)

from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from tqdm import tqdm

import pklot_env_unicycle_cont as pklot_env_cont

# Seed
seed = 0
torch.manual_seed(seed)

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# Sampling
frames_per_batch = 1_000  # Number of team frames collected per sampling iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# We will stop training the evaders after this many iterations,
# should be 0 <= iteration_when_stop_training_evaders <= n_iters
iteration_when_stop_training_evaders = n_iters // 2

# Replay buffer
memory_size = 1_000_000  # The replay buffer of each group can store this many frames

# Training
n_optimiser_steps = 100  # Number of optimisation steps per training iteration
train_batch_size = 128  # Number of frames trained in each optimiser step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# DDPG
gamma = 0.99  # Discount factor
polyak_tau = 0.005  # Tau for the soft-update of the target network

max_steps = 250  # Environment steps before done

n_agents = 1

env_config = pklot_env_cont.EnvParams(
    reward_stop=-10, reward_dist=-1, reward_heading=-1, reward_time=-1, reward_collision=-10, reward_goal=1000,
    window_size=140
)

base_env = PettingZooWrapper(
    pklot_env_cont.parallel_env(max_cycles=max_steps, n_vehicles=n_agents, params=env_config, random_reset=False),
    seed=seed,
)

print("action_spec:", base_env.full_action_spec)
print("reward_spec:", base_env.full_reward_spec)
print("done_spec:", base_env.full_done_spec)
print("observation_spec:", base_env.observation_spec)

env = TransformedEnv(
    base_env,
    RewardSum(
        in_keys=base_env.reward_keys,
        reset_keys=["_reset"] * len(base_env.group_map.keys()),
    ),
)
env = TransformedEnv(
    env,
    PermuteTransform(
        dims=[-1, -3, -2, -4]
    )
)

######################################################################
# We can see that our rollout has ``batch_size`` of ``(n_rollout_steps)``.
# This means that all the tensors in it will have this leading dimension.
#
# Looking more in depth, we can see that the output tensordict can be divided in the following way:
#
# - *In the root* (accessible by running ``rollout.exclude("next")`` ) we will find all the keys that are available
#   after a reset is called at the first timestep. We can see their evolution through the rollout steps by indexing
#   the ``n_rollout_steps`` dimension. Among these keys, we will find the ones that are different for each agent
#   in the ``rollout[group_name]`` tensordicts, which will have batch size ``(n_rollout_steps, n_agents_in_group)``
#   signifying that it is storing the additional agent dimension. The ones outside the group tensordicts
#   will be the shared ones.
# - *In the next* (accessible by running ``rollout.get("next")`` ). We will find the same structure as the root with some minor differences highlighted below.
#
# In TorchRL the convention is that done and observations will be present in both root and next (as these are
# available both at reset time and after a step). Action will only be available in root (as there is no action
# resulting from a step) and reward will only be available in next (as there is no reward at reset time).
# This structure follows the one in **Reinforcement Learning: An Introduction (Sutton and Barto)** where root represents data at time :math:`t` and
# next represents data at time :math:`t+1` of a world step.
#
#
# Render a random rollout
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# If you are on Google Colab, or on a machine with OpenGL and a GUI, you can actually render a random rollout.
# This will give you an idea of what a random policy will achieve in this task, in order to compare it
# with the policy you will train yourself!
#
# To render a rollout, follow the instructions in the *Render* section at the end of this tutorial
# and just remove the line ``policy=agents_exploration_policy`` from ``env.rollout()``.
#
#
# Policy
# ------
#
# DDPG utilises a deterministic policy. This means that our
# neural network will output the action to take.
# As the action is continuous, we use a Tanh-Delta distribution to respect the
# action space boundaries. The only thing that this class does is apply a Tanh transformation to make sure the action
# is withing the domain bounds.
#
# Another important decision we need to make is whether we want the agents within a team to **share the policy parameters**.
# On the one hand, sharing parameters means that they will all share the same policy, which will allow them to benefit from
# each other's experiences. This will also result in faster training.
# On the other hand, it will make them behaviourally *homogenous*, as they will in fact share the same model.
# For this example, we will enable sharing as we do not mind the homogeneity and can benefit from the computational
# speed, but it is important to always think about this decision in your own problems!
#
# We design the policy in three steps.
#
# **First**: define a neural network ``n_obs_per_agent`` -> ``n_actions_per_agents``
#
# For this we use the ``MultiAgentMLP``, a TorchRL module made exactly for
# multiple agents, with much customisation available.
#
# We will define a different policy for each group and store them in a dictionary.
#
#

policy_modules = {}
for group, agents in env.group_map.items():
    share_parameters_policy = True  # Can change this based on the group

    policy_net = MultiAgentConvNet(
        in_features=env.observation_spec[group, "observation"].shape[
            1::
        ],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[group, "action"].shape[
            -1
        ],  # n_actions_per_agents
        n_agents=len(agents),  # Number of agents in the group
        centralised=False,  # the policies are decentralised (i.e., each agent will act from its local observation)
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
    # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
    # neural networks, and write the
    # outputs in-place at the ``out_keys``.

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "param")],
    )  # We just name the input and output that the network will read and write to the input tensordict
    policy_modules[group] = policy_module


######################################################################
# **Second**: wrap the :class:`~tensodrdict.nn.TensorDictModule` in a :class:`~torchrl.modules.ProbabilisticActor`
#
# We now need to build the TanhDelta distribution.
# We instruct the :class:`~torchrl.modules.ProbabilisticActor`
# class to build a :class:`~torchrl.modules.TanhDelta` out of the policy action
# parameters. We also provide the minimum and maximum values of this
# distribution, which we gather from the environment specs.
#
# The name of the ``in_keys`` (and hence the name of the ``out_keys`` from
# the :class:`TensorDictModule` above) has to end with the
# :class:`~torchrl.modules.TanhDelta` distribution constructor keyword arguments (param).
#

policies = {}
for group, _agents in env.group_map.items():
    policy = ProbabilisticActor(
        module=policy_modules[group],
        spec=env.full_action_spec[group, "action"],
        in_keys=[(group, "param")],
        out_keys=[(group, "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env.full_action_spec[group, "action"].space.low,
            "max": env.full_action_spec[group, "action"].space.high,
        },
        return_log_prob=False,
    )
    policies[group] = policy

######################################################################
# **Third**: Exploration
#
# Since the DDPG policy is deterministic, we need a way to perform exploration during collection.
#
# For this purpose, we need to append an exploration layer to our policies before passing them to the collector.
# In this case we use a :class:`~torchrl.modules.AdditiveGaussianWrapper`, which adds gaussian noise to our action
# (and clamps it if the noise makes the action out of bounds).
#
# This exploration wrapper uses a ``sigma`` parameter which is multiplied by the noise to determine its magnitude.
# Sigma can be annealed throughout training to reduce exploration.
# Sigma will go from ``sigma_init`` to ``sigma_end`` in ``annealing_num_steps``.
#


exploration_policies = {}
for group, _agents in env.group_map.items():
    exploration_policy = AdditiveGaussianWrapper(
        policies[group],
        annealing_num_steps=total_frames
        // 2,  # Number of frames after which sigma is sigma_end
        action_key=(group, "action"),
        sigma_init=0.9,  # Initial value of the sigma
        sigma_end=0.1,  # Final value of the sigma
    )
    exploration_policies[group] = exploration_policy


######################################################################
# Critic network
# --------------
#
# The critic network is a crucial component of the DDPG algorithm, even though it
# isn't used at sampling time. This module will read the observations & actions taken and
# return the corresponding value estimates.
#
# As before, one should think carefully about the decision of **sharing the critic parameters** within an agent group.
# In general, parameter sharing will grant faster training convergence, but there are a few important
# considerations to be made:
#
# - Sharing is not recommended when agents have different reward functions, as the critics will need to learn
#   to assign different values to the same state (e.g., in mixed cooperative-competitive settings).
#   In this case, since the two groups are already using separate networks, the sharing decision only applies
#   for agents within a group, which we already know have the same reward function.
# - In decentralised training settings, sharing cannot be performed without additional infrastructure to
#   synchronise parameters.
#
# In all other cases where the reward function (to be differentiated from the reward) is the same for all agents
# in a group (as in the current scenario),
# sharing can provide improved performance. This can come at the cost of homogeneity in the agent strategies.
# In general, the best way to know which choice is preferable is to quickly experiment both options.
#
# Here is also where we have to choose between **MADDPG and IDDPG**:
#
# - With MADDPG, we will obtain a central critic with full-observability
#   (i.e., it will take all the concatenated global agent observations and actions as input).
#   We can do this because we are in a simulator
#   and training is centralised.
# - With IDDPG, we will have a local decentralised critic, just like the policy.
#
# In any case, the critic output will have shape ``(..., n_agents_in_group, 1)``.
# If the critic is centralised and shared,
# all the values along the ``n_agents_in_group`` dimension will be identical.
#
# As with the policy, we create a critic network for each group and store them in a dictionary.
#

critics = {}
for group, agents in env.group_map.items():
    share_parameters_critic = True  # Can change for each group
    MADDPG = True  # IDDPG if False, can change for each group

    # This module applies the lambda function: reading the action and observation entries for the group
    # and concatenating them in a new ``(group, "obs_action")`` entry
    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[(group, "observation"), (group, "action")],
        out_keys=[(group, "obs_action")],
    )

    critic_module = TensorDictModule(
        module=MultiAgentConvNet(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
            + env.full_action_spec[group, "action"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=len(agents),
            centralised=MADDPG,
            share_params=share_parameters_critic,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        in_keys=[(group, "obs_action")],  # Read ``(group, "obs_action")``
        out_keys=[
            (group, "state_action_value")
        ],  # Write ``(group, "state_action_value")``
    )

    critics[group] = TensorDictSequential(
        cat_module, critic_module
    )  # Run them in sequence


######################################################################
# Let us try our policy and critic modules. As pointed earlier, the usage of
# :class:`~tensordict.nn.TensorDictModule` makes it possible to directly read the output
# of the environment to run these modules, as they know what information to read
# and where to write it.
#
# We can see that after each group's networks are run their output keys are added to the data under the
# group entry.
#
# **From this point on, the multi-agent-specific components have been instantiated, and we will simply use the same
# components as in single-agent learning. Isn't this fantastic?**
#
reset_td = env.reset()
for group, _agents in env.group_map.items():
    print(
        f"Running value and policy for group '{group}':",
        critics[group](policies[group](reset_td)),
    )

######################################################################
# Data collector
# --------------
#
# TorchRL provides a set of data collector classes. Briefly, these
# classes execute three operations: reset an environment, compute an action
# using the policy and the latest observation, execute a step in the environment, and repeat
# the last two steps until the environment signals a stop (or reaches a done
# state).
#
# We will use the simplest possible data collector, which has the same output as an environment rollout,
# with the only difference that it will auto reset done states until the desired frames are collected.
#
# We need to feed it our exploration policies. Furthermore, to run the policies from all groups as if they were one,
# we put them in a sequence. They will not interfere with each other as each group writes and reads keys in different places.
#

# Put exploration policies from each group in a sequence
agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

collector = SyncDataCollector(
    env,
    agents_exploration_policy,
    device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

######################################################################
# Replay buffer
# -------------
#
# Replay buffers are a common building piece of off-policy RL sota-implementations.
# There are many types of buffers, in this tutorial we use a basic buffer to store and sample tensordict
# data randomly.
#

replay_buffers = {}
for group, _agents in env.group_map.items():
    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(
            memory_size, device=device
        ),  # We will store up to memory_size multi-agent transitions
        sampler=RandomSampler(),
        batch_size=train_batch_size,  # We will sample batches of this size
    )
    replay_buffers[group] = replay_buffer

######################################################################
# Loss function
# -------------
#
# The DDPG loss can be directly imported from TorchRL for convenience using the
# :class:`~.objectives.DDPGLoss` class. This is the easiest way of utilising DDPG:
# it hides away the mathematical operations of DDPG and the control flow that
# goes with it.
#
# It is also possible to have different policies for each group.
#
losses = {}
for group, _agents in env.group_map.items():
    loss_module = DDPGLoss(
        actor_network=policies[group],  # Use the non-explorative policies
        value_network=critics[group],
        delay_value=True,  # Whether to use a target network for the value
        loss_function="l2",
    )
    loss_module.set_keys(
        state_action_value=(group, "state_action_value"),
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

    losses[group] = loss_module

target_updaters = {
    group: SoftUpdate(loss, tau=polyak_tau) for group, loss in losses.items()
}

optimisers = {
    group: {
        "loss_actor": torch.optim.Adam(
            loss.actor_network_params.flatten_keys().values(), lr=lr
        ),
        "loss_value": torch.optim.Adam(
            loss.value_network_params.flatten_keys().values(), lr=lr
        ),
    }
    for group, loss in losses.items()
}

######################################################################
# Training utils
# --------------
#
# We do have to define two helper functions that we will use in the training loop.
# They are very simple and do not contain any important logic.
#
#


def process_batch(batch: TensorDictBase) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch


######################################################################
# Training loop
# -------------
# We now have all the pieces needed to code our training loop.
# The steps include:
#
# * Collect data for all groups
#     * Loop over groups
#         * Store group data in group buffer
#         * Loop over epochs
#             * Sample from group buffer
#             * Compute loss on sampled data
#             * Back propagate loss
#             * Optimise
#         * Repeat
#     * Repeat
# * Repeat
#
#

pbar = tqdm(
    total=n_iters,
    desc=", ".join(
        [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
    ),
)
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
train_group_map = copy.deepcopy(env.group_map)

# Training/collection iterations
for iteration, batch in enumerate(collector):
    current_frames = batch.numel()
    batch = process_batch(batch)  # Util to expand done keys if needed
    # Loop over groups
    for group in train_group_map.keys():
        group_batch = batch.exclude(
            *[
                key
                for _group in env.group_map.keys()
                if _group != group
                for key in [_group, ("next", _group)]
            ]
        )  # Exclude data from other groups
        group_batch = group_batch.reshape(
            -1
        )  # This just affects the leading dimensions in batch_size of the tensordict
        replay_buffers[group].extend(group_batch)

        for _ in range(n_optimiser_steps):
            subdata = replay_buffers[group].sample()
            loss_vals = losses[group](subdata)

            for loss_name in ["loss_actor", "loss_value"]:
                loss = loss_vals[loss_name]
                optimiser = optimisers[group][loss_name]

                loss.backward()

                # Optional
                params = optimiser.param_groups[0]["params"]
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

                optimiser.step()
                optimiser.zero_grad()

            # Soft-update the target network
            target_updaters[group].step()

        # Exploration sigma anneal update
        exploration_policies[group].step(current_frames)

    # Stop training a certain group when a condition is met (e.g., number of training iterations)
    if iteration == iteration_when_stop_training_evaders:
        del train_group_map["agent"]

    # Logging
    for group in env.group_map.keys():
        episode_reward_mean = (
            batch.get(("next", group, "episode_reward"))[
                batch.get(("next", group, "done"))
            ]
            .mean()
            .item()
        )
        episode_reward_mean_map[group].append(episode_reward_mean)

    pbar.set_description(
        ", ".join(
            [
                f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                for group in env.group_map.keys()
            ]
        ),
        refresh=False,
    )
    pbar.update()

######################################################################
# Results
# -------
#
# We can plot the mean reward obtained per episode.
#
# To make training last longer, increase the ``n_iters`` hyperparameter.
#
# When running this script locally, you may need to close the opened window to
# proceed with the rest of the screen.
#

fig, axs = plt.subplots(2, 1)
for i, group in enumerate(env.group_map.keys()):
    axs[i].plot(episode_reward_mean_map[group], label=f"Episode reward mean {group}")
    axs[i].set_ylabel("Reward")
    axs[i].axvline(
        x=iteration_when_stop_training_evaders,
        label="Agent (evader) stop training",
        color="orange",
    )
    axs[i].legend()
axs[-1].set_xlabel("Training iterations")
plt.show()

