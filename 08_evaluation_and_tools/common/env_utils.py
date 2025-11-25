import gymnasium as gym


def make_cartpole_env(render: bool = False):
    if render:
        return gym.make("CartPole-v1", render_mode="human")
    else:
        return gym.make("CartPole-v1")


def make_pendulum_env(render: bool = False):
    if render:
        return gym.make("Pendulum-v1", render_mode="human")
    else:
        return gym.make("Pendulum-v1")
