import datetime
import os
import hydra
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def launch_rlg_hydra(cfg: DictConfig):
    from datetime import datetime
    from os import path as os_path
    import torch
    import numpy as np
    import confrez.rl.envs.pklot_env_unicycle as pklot_env_unicycle
    from tianshou.data import (
        Collector,
        VectorReplayBuffer,
    )
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.trainer import offpolicy_trainer
    from tianshou.utils import WandbLogger
    from torch.utils.tensorboard import SummaryWriter
    from tianshou_experiment import render
    from confrez.rl.utils.train_utils import get_env, get_agents, get_unicycle_env

    import wandb

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    if cfg.wandb_activate:
        # Make sure to install WandB if you actually use this.

        run = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"rainbow{time_str}",
            save_interval=50,
            # monitor_gym=True, #while this could be problematic
        )

    cwd = os_path.dirname(__file__)
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

    if cfg.env_name == "pklot":
        train_env = SubprocVectorEnv([get_env for _ in range(cfg.num_train_envs)])
        test_env = DummyVectorEnv([get_env for _ in range(cfg.num_test_envs)])
        action_shape = 7
    elif cfg.env_name == "pklot_unicycle":
        train_env = SubprocVectorEnv(
            [get_unicycle_env for _ in range(cfg.num_train_envs)]
        )
        test_env = DummyVectorEnv([get_unicycle_env for _ in range(cfg.num_test_envs)])
        temp = pklot_env_unicycle.parallel_env()
        action_shape = len(temp.actions)
        del temp

    # seed
    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents(
        num_agents=cfg.num_agents,
        discount_factor=cfg.discount_factor,
        estimation_step=cfg.estimation_step,
        action_shape=action_shape,
    )

    # ======== Step 3: Collector setup =========

    train_collector = Collector(
        policy,
        train_env,
        VectorReplayBuffer(cfg.buffer_size, len(train_env)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_env)
    for agent in agents:
        policy.policies[agent].set_eps(1)
    train_collector.collect(n_episode=cfg.batch_size * cfg.num_train_envs)

    # ======== Step 4: Callback functions setup =========
    # logger:
    logger = WandbLogger(
        project="confrez-tianshou",
        name=f"rainbow{cfg.task_name}_{timestamp}",
        save_interval=50,
    )

    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "dqn"), exist_ok=True)
        for i, agent in enumerate(agents):
            model_save_path = os.path.join("log", "dqn", f"policy{i}.pth")
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            logger.wandb_run.save(model_save_path)
        render(agents, policy)
        logger.wandb_run.log({"video": wandb.Video("out.gif", fps=4, format="gif")})

    def stop_fn(mean_rewards):
        # currently set to never stop
        return False

    def train_fn(epoch, env_step):
        # print(env_step, policy.policies[agents[0]]._iter)
        for agent in agents:
            policy.policies[agent].set_eps(max(0.99**epoch, 0.1))
            # train_collector.buffer.set_beta(min(0.4 * 1.02**epoch, 1))

    def test_fn(epoch, env_step):
        for agent in agents:
            policy.policies[agent].set_eps(0.0)
        if epoch % 50 == 0:
            render(agents, policy)
            logger.wandb_run.log({"video": wandb.Video("out.gif", fps=4, format="gif")})

    def reward_metric(rews):
        return np.average(rews, axis=1)

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=cfg.max_epoch,
        step_per_epoch=cfg.step_per_epoch,
        step_per_collect=cfg.step_per_collect,
        episode_per_test=cfg.episode_per_test,
        batch_size=cfg.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=cfg.update_per_step,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
    )

    print("result: ", result)


if __name__ == "__main__":
    launch_rlg_hydra()
