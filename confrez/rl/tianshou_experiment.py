import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou_train import *
from tianshou.data import Batch, to_numpy
import PIL
from PIL import Image


def render_human():
    for i, agent in enumerate(agents):
        # shouldn't be in this way when num_agents > 1!
        filename = os.path.join("log", "dqn", f"policy{i}.pth")
        policy.policies[agent].load_state_dict(torch.load(filename, map_location=torch.device('cuda')))
    for agent in agents:
        policy.policies[agent].set_eps(0)
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=1 / 30)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:].mean()}, length: {lens.mean()}")


def render(agents, policy):
    env = get_env()
    frame_list = []
    obs = env.reset()
    obs = np.array([obs['obs']])
    done = False
    total_reward = 0
    while not done:
        for agent in agents:
            logits = policy.policies[agent].model(obs)
            q, _ = policy.policies[agent].compute_q_value(logits, mask=None)
            act = to_numpy(q.max(dim=1)[1])
            obs, reward, done, _, _ = env.step(act)
            obs = np.array([obs['obs']])
            frame_list.append(PIL.Image.fromarray(env.env.env.env.env.env.aec_env.env.env.render('rgb_array')))
            total_reward += np.sum(reward)

    env.close()
    frame_list[0].save('out.gif', save_all=True, append_images=frame_list[1:], duration=100, loop=0)


if __name__ == '__main__':
    # render_human()
    env = DummyVectorEnv([get_env])
    policy, optim, agents = get_agents()

    for i, agent in enumerate(agents):
        # shouldn't be in this way when num_agents > 1!
        filename = os.path.join("log", "dqn", f"policy{i}.pth")
        policy.policies[agent].load_state_dict(torch.load(filename, map_location=torch.device('cuda')))
    render(agents, policy)
