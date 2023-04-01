from dataclasses import dataclass, field
import functools
from itertools import product
from typing import Dict, List, Set, Tuple
import random
import casadi as ca
from shapely.geometry import Polygon


from gymnasium.spaces import Box, Discrete
import numpy as np
import pygame
from gymnasium.utils import EzPickle, seeding

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import PythonMsg, VehicleState
from confrez.vehicle_types import VehicleConfig, VehicleBody
from confrez.dynamic_model import unicycle_simulator


@dataclass
class EnvParams(PythonMsg):
    """
    parameters of the environment
    """

    spot_width: float = field(default=2.5)
    region: GeofenceRegion = field(default=None)
    window_size: int = field(default=280)

    dyaw_res: float = field(default=0.1)
    speed_res: float = field(default=0.1)

    dt: float = field(default=0.1)

    goal_r: float = field(default=0.5)
    goal_y: float = field(default=np.pi / 6)

    reward_time: float = field(default=-1)
    reward_stop: float = field(default=-10)
    reward_collision: float = field(default=-1e3)
    reward_dist: float = field(default=-1)
    reward_goal: float = field(default=1e4)

    def __post_init__(self):
        self.region = GeofenceRegion(
            x_max=14 * self.spot_width, x_min=0, y_max=14 * self.spot_width, y_min=0
        )


def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(**kwargs)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "pklot"}

    def __init__(
        self,
        n_vehicles=4,
        max_cycles=500,
        seed=None,
        random_reset=False,
        render_mode="human",
        params=EnvParams(),
    ):
        EzPickle.__init__(self, n_vehicles, max_cycles)
        self.render_mode = render_mode
        self.n_vehicles = n_vehicles

        self.params = params
        self.spot_width = self.params.spot_width
        self.window_size = self.params.window_size

        self.res = self.window_size / (
            self.params.region.x_max - self.params.region.x_min
        )
        self.dt = self.params.dt

        self.possible_agents = ["vehicle_" + str(i) for i in range(self.n_vehicles)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.n_vehicles)))
        )
        self.random_reset = random_reset

        self.vb = VehicleBody()
        self.vehicle_config = VehicleConfig()
        self.simulator = unicycle_simulator(dt=self.dt)

        # Observation space for each vehicle is the entire display window
        self.observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    Box(
                        low=0,
                        high=255,
                        shape=(self.window_size, self.window_size, 3),
                        dtype=np.uint8,
                    )
                ]
                * self.n_vehicles,
            )
        )

        speed_max = self.vehicle_config.v_max
        psidot_max = (
            self.vehicle_config.v_max
            / self.vb.wb
            * np.tan(self.vehicle_config.delta_max)
        )
        num_speed = 2 * int(speed_max / self.params.speed_res) + 1
        num_dpsi = 2 * int(psidot_max / self.params.dyaw_res) + 1

        self.actions: List[Tuple[float, float]] = []
        for v in np.linspace(-speed_max, speed_max, num_speed):
            for w in np.linspace(-psidot_max, psidot_max, num_dpsi):
                self.actions.append((v, w))

        # Action space for each vehicle has num_steer * num_accel discrete choices
        self.action_spaces = dict(
            zip(
                self.possible_agents,
                [Discrete(len(self.actions))] * self.n_vehicles,
            )
        )

        # ======== Initial and final states of all agents
        self.agent_configs = [
            {
                "init_state": [6.5 * self.spot_width, 8 * self.spot_width, np.pi / 2],
                "goal": [12 * self.spot_width, 6.5 * self.spot_width, 0],
            },
            {
                "init_state": [9 * self.spot_width, 7.5 * self.spot_width, np.pi],
                "goal": [6.5 * self.spot_width, 4 * self.spot_width, 3 * np.pi / 2],
            },
            {
                "init_state": [6.5 * self.spot_width, 5 * self.spot_width, np.pi / 2],
                "goal": [2 * self.spot_width, 7.5 * self.spot_width, np.pi],
            },
            {
                "init_state": [5 * self.spot_width, 6.5 * self.spot_width, 0],
                "goal": [6.5 * self.spot_width, 10 * self.spot_width, np.pi / 2],
            },
        ]

        self.max_cycles = max_cycles
        self.frame = 0
        self.cycle_done = False

        self.closed = False
        self.seed(seed)

        pygame.init()
        self.renderOn = False
        self.screen = pygame.Surface(
            (self.window_size, self.window_size)
        )  # Now it is just a normal Surface. When the human render mode is chosen, it will be changed to `display`

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill(color=(0, 0, 0))

        self._colors = [
            {"front": (255, 119, 0), "back": (128, 60, 0)},
            {"front": (0, 255, 212), "back": (0, 140, 117)},
            {"front": (255, 255, 255), "back": (128, 128, 128)},
            {"front": (255, 0, 149), "back": (128, 0, 74)},
            {"front": (200, 255, 0), "back": (100, 128, 0)},
        ]

        self.init_walls()

    def g2i(self, x: float, y: float) -> Tuple[int, int]:
        """
        convert ground coordinates into pixel coordinates
        """
        return x * self.res, (self.params.region.y_max - y) * self.res

    def update_vehicle_polygon(self, agent: str) -> None:
        state = self.states[agent]
        R = state.get_R()[:2, :2]
        car_outline = self.vb.xy @ R.T + np.array([state.x.x, state.x.y])
        self.vehicle_ps[agent] = Polygon(car_outline)

    def init_vehicles(self):
        if self.random_reset:
            n_vehicles = random.choice(range(1, self.n_vehicles + 1))
            self.agents = sorted(random.sample(self.possible_agents, n_vehicles))

            configs = random.sample(self.agent_configs, n_vehicles)
        else:
            self.agents = self.possible_agents[:]
            configs = self.agent_configs[:]

        self.states: Dict[str, VehicleState] = {
            agent: VehicleState() for agent in self.agents
        }
        self.vehicle_ps: Dict[str, Polygon] = {agent: None for agent in self.agents}
        self.goals: Dict[str, VehicleState] = {
            agent: VehicleState() for agent in self.agents
        }

        for agent, config in zip(self.agents, configs):
            # Apply offsets so that the state and goal is about the rear axle center
            init_state = config["init_state"]
            self.states[agent].x.x = init_state[0] - self.vb.wb / 2 * np.cos(
                init_state[2]
            )
            self.states[agent].x.y = init_state[1] - self.vb.wb / 2 * np.sin(
                init_state[2]
            )
            self.states[agent].e.psi = init_state[2]

            self.update_vehicle_polygon(agent)

            goal = config["goal"]

            self.goals[agent].x.x = goal[0] - self.vb.wb / 2 * np.cos(goal[2])
            self.goals[agent].x.y = goal[1] - self.vb.wb / 2 * np.sin(goal[2])
            self.goals[agent].e.psi = goal[2]

    def init_walls(self) -> None:
        """
        initialize the rectangles for walls
        """
        self.walls: List[Polygon] = []

        # =============== Static obstacles
        # Bottom left
        self.walls.append(
            Polygon(
                [
                    [
                        1 * self.spot_width,
                        3 * self.spot_width,
                    ],
                    [
                        1 * self.spot_width,
                        5.5 * self.spot_width,
                    ],
                    [
                        5.5 * self.spot_width + self.vb.w / 2,
                        5.5 * self.spot_width,
                    ],
                    [
                        5.5 * self.spot_width + self.vb.w / 2,
                        3 * self.spot_width,
                    ],
                ]
            )
        )
        # Bottom center
        self.walls.append(
            Polygon(
                [
                    [
                        7.5 * self.spot_width - self.vb.w / 2,
                        3 * self.spot_width,
                    ],
                    [
                        7.5 * self.spot_width - self.vb.w / 2,
                        5.5 * self.spot_width,
                    ],
                    [
                        7.5 * self.spot_width + self.vb.w / 2,
                        5.5 * self.spot_width,
                    ],
                    [
                        7.5 * self.spot_width + self.vb.w / 2,
                        3 * self.spot_width,
                    ],
                ]
            )
        )
        # Bottom right
        self.walls.append(
            Polygon(
                [
                    [
                        9.5 * self.spot_width - self.vb.w / 2,
                        3 * self.spot_width,
                    ],
                    [
                        9.5 * self.spot_width - self.vb.w / 2,
                        5.5 * self.spot_width,
                    ],
                    [
                        13 * self.spot_width,
                        5.5 * self.spot_width,
                    ],
                    [
                        13 * self.spot_width,
                        3 * self.spot_width,
                    ],
                ]
            )
        )
        # Top left
        self.walls.append(
            Polygon(
                [
                    [
                        1 * self.spot_width,
                        8.5 * self.spot_width,
                    ],
                    [
                        1 * self.spot_width,
                        11 * self.spot_width,
                    ],
                    [
                        5.5 * self.spot_width + self.vb.w / 2,
                        11 * self.spot_width,
                    ],
                    [
                        5.5 * self.spot_width + self.vb.w / 2,
                        8.5 * self.spot_width,
                    ],
                ]
            )
        )
        # Top center
        self.walls.append(
            Polygon(
                [
                    [
                        7.5 * self.spot_width - self.vb.w / 2,
                        8.5 * self.spot_width,
                    ],
                    [
                        7.5 * self.spot_width - self.vb.w / 2,
                        11 * self.spot_width,
                    ],
                    [
                        8.5 * self.spot_width + self.vb.w / 2,
                        11 * self.spot_width,
                    ],
                    [
                        8.5 * self.spot_width + self.vb.w / 2,
                        8.5 * self.spot_width,
                    ],
                ]
            )
        )
        # Top right
        self.walls.append(
            Polygon(
                [
                    [
                        10.5 * self.spot_width - self.vb.w / 2,
                        8.5 * self.spot_width,
                    ],
                    [
                        10.5 * self.spot_width - self.vb.w / 2,
                        11 * self.spot_width,
                    ],
                    [
                        13 * self.spot_width,
                        11 * self.spot_width,
                    ],
                    [
                        13 * self.spot_width,
                        8.5 * self.spot_width,
                    ],
                ]
            )
        )

        # =========== Boundaries
        # Up
        self.walls.append(
            Polygon(
                [
                    [0, 11 * self.spot_width],
                    [0, 14 * self.spot_width],
                    [14 * self.spot_width, 14 * self.spot_width],
                    [14 * self.spot_width, 11 * self.spot_width],
                ]
            )
        )

        # Down
        self.walls.append(
            Polygon(
                [
                    [0, 0],
                    [0, 3 * self.spot_width],
                    [14 * self.spot_width, 3 * self.spot_width],
                    [14 * self.spot_width, 0],
                ]
            )
        )

        # Left
        self.walls.append(
            Polygon(
                [
                    [0, 0],
                    [0, 14 * self.spot_width],
                    [1 * self.spot_width, 14 * self.spot_width],
                    [1 * self.spot_width, 0],
                ]
            )
        )

        # Right
        self.walls.append(
            Polygon(
                [
                    [13 * self.spot_width, 0],
                    [13 * self.spot_width, 14 * self.spot_width],
                    [14 * self.spot_width, 14 * self.spot_width],
                    [14 * self.spot_width, 0],
                ]
            )
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Also seeding the action spaces for reproducable results
        if seed is not None:
            for i, agent in enumerate(self.possible_agents):
                self.action_space(agent).seed(seed + i)

    def move(self, agent: str, action: int):
        # speed and yaw rate
        v, w = self.actions[action]
        input = ca.vertcat(v, w)

        state = ca.vertcat(
            self.states[agent].x.x, self.states[agent].x.y, self.states[agent].e.psi
        )
        state_new = self.simulator(state, input)
        self.states[agent].x.x = state_new[0].__float__()
        self.states[agent].x.y = state_new[1].__float__()
        self.states[agent].e.psi = state_new[2].__float__()
        self.update_vehicle_polygon(agent=agent)

    def has_collision(self, agent: str) -> bool:
        """
        check whether this agent collide with other agents or walls
        """
        p = self.vehicle_ps[agent]
        for _p in self.walls:
            if p.intersects(_p):
                return True

        for _agent, _p in self.vehicle_ps.items():
            if agent != _agent and p.intersects(_p):
                return True

        return False

    def dist2goal(self, agent: str) -> float:
        """
        calculate the distance to goal
        """
        state = self.states[agent]
        goal = self.goals[agent]

        return np.linalg.norm([state.x.x - goal.x.x, state.x.y - goal.x.y])

    def reach_goal(self, agent: str) -> bool:
        """
        check whether this agent has reached its goal
        """
        state = self.states[agent]
        goal = self.goals[agent]

        if (
            self.dist2goal(agent=agent) <= self.params.goal_r
            and np.abs(state.e.psi % (2 * np.pi) - goal.e.psi % (2 * np.pi))
            <= self.params.goal_y
        ):
            return True
        else:
            return False

    def draw_walls(self, surf: pygame.Surface = None):
        if surf is None:
            surf = self.canvas

        surf.fill((0, 0, 0))
        for p in self.walls:
            x, y = p.exterior.coords.xy
            px, py = self.g2i(np.array(x), np.array(y))

            pygame.draw.polygon(
                surface=surf, color=(0, 0, 255), points=np.vstack([px, py]).T
            )

    def draw_car(
        self,
        agent: str,
        color: Dict[str, Tuple[int, int, int]],
        surf: pygame.Surface = None,
    ):
        """
        draw the specified agent at current time step
        `color`: the color dictionary to draw the vehicle
        """

        if surf is None:
            surf = self.canvas

        state = self.states[agent]
        R = state.get_R()[:2, :2]
        car_outline = self.vb.xy @ R.T + np.array([state.x.x, state.x.y])
        mid_points = np.array(
            [
                (car_outline[0] + car_outline[1]) / 2,
                (car_outline[2] + car_outline[3]) / 2,
            ]
        )
        front_xy = np.array(
            [car_outline[0], mid_points[0], mid_points[1], car_outline[3]]
        )
        back_xy = np.array(
            [mid_points[0], car_outline[1], car_outline[2], mid_points[1]]
        )

        px, py = self.g2i(front_xy[:, 0], front_xy[:, 1])
        pygame.draw.polygon(
            surface=surf, color=color["front"], points=np.vstack([px, py]).T
        )

        px, py = self.g2i(back_xy[:, 0], back_xy[:, 1])
        pygame.draw.polygon(
            surface=surf, color=color["back"], points=np.vstack([px, py]).T
        )

    def draw_goal(
        self,
        agent: str,
        color: Dict[str, Tuple[int, int, int]],
        surf: pygame.Surface = None,
    ):
        """
        draw the goal of the specified agent
        `surf`: the pygame surface to plot on
        """
        if surf is None:
            surf = self.canvas

        goal = self.goals[agent]

        front_goal_x = goal.x.x + self.vb.wb * np.cos(goal.e.psi)
        front_goal_y = goal.x.y + self.vb.wb * np.sin(goal.e.psi)

        back_goal_x = goal.x.x
        back_goal_y = goal.x.y

        pygame.draw.circle(
            surface=surf,
            color=color["front"],
            center=self.g2i(front_goal_x, front_goal_y),
            radius=self.params.goal_r * self.res,
        )

        pygame.draw.circle(
            surface=surf,
            color=color["back"],
            center=self.g2i(back_goal_x, back_goal_y),
            radius=self.params.goal_r * self.res,
        )

    def observe(self, agent: str) -> np.ndarray:
        """
        get agent-specific observation.
        """
        surf = self.canvas.copy()

        # Draw all goals first
        i = 1  # Index of color
        for _agent in self.agents:
            if _agent == agent:
                continue
            else:
                self.draw_goal(agent=_agent, color=self._colors[i], surf=surf)
                i += 1

        # Draw all cars next to cover goals
        i = 1  # Index of color
        for _agent in self.agents:
            if _agent == agent:
                continue
            else:
                self.draw_car(agent=_agent, color=self._colors[i], surf=surf)
                i += 1

        # plot the car itself with ego color
        self.draw_goal(agent=agent, color=self._colors[0], surf=surf)
        self.draw_car(agent=agent, color=self._colors[0], surf=surf)

        # Return an image
        observation = pygame.surfarray.pixels3d(surf)
        observation = np.rot90(observation, k=3)
        observation = np.fliplr(observation)

        return observation

    def enable_render(self):
        """
        change the `screen` surface to be display
        """
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))

        self.renderOn = True

    def render(self):
        """
        Renders the environment
        """
        if self.render_mode == "human" and not self.renderOn:
            # sets self.renderOn to true and initializes display
            self.enable_render()

        surf = self.canvas.copy()

        # Draw agents
        for agent in self.possible_agents:
            self.draw_goal(
                agent, color=self._colors[self.agent_name_mapping[agent]], surf=surf
            )

        for agent in self.possible_agents:
            self.draw_car(
                agent, color=self._colors[self.agent_name_mapping[agent]], surf=surf
            )

        # Attach canvas on to the screen
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            screenshot = np.array(pygame.surfarray.pixels3d(self.screen))

            return np.transpose(screenshot, axes=(1, 0, 2))

    def close(self):
        if not self.closed:
            self.closed = True
            if self.renderOn:
                self.screen = pygame.Surface((self.window_size, self.window_size))
                self.renderOn = False
                pygame.event.pump()
                pygame.display.quit()

    def reset(
        self, seed=None, return_info=False, options=None
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.seed(seed)

        self.frame = 0
        self.cycle_done = False

        self.init_walls()
        self.init_vehicles()

        self.draw_walls()

        observations = {agent: self.observe(agent) for agent in self.agents}

        if not return_info:
            return observations
        else:
            infos = {
                agent: {"states": self.states[agent].copy()} for agent in self.agents
            }

            return observations, infos

    def is_static_action(self, action: int) -> bool:
        """
        check whether a sampled action is static
        """
        v, w = self.actions[action]
        return np.abs(v) < 1e-2 and np.abs(w) < 1e-2

    def step(self, actions: Dict[str, int]):
        """
        step the entire environment forward
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # Init return values
        observations = {agent: None for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {
            agent: False for agent in self.agents
        }  # All agents that are in self.agents should be active, thus not done
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if not self.cycle_done:
            # Move agents with actions simultaneously
            for agent in self.agents:
                self.move(agent=agent, action=actions[agent])

            # Check collision or goal completion and apply costs
            for agent in self.agents:
                # The time cost for vehicle
                rewards[agent] += self.params.reward_time

                if self.is_static_action(action=actions[agent]):
                    # If the vehicle remain stationary, it will get a small penalty
                    rewards[agent] += self.params.reward_stop

                if self.has_collision(agent):
                    # If collide with other agents or walls, apply huge penalty
                    rewards[agent] += self.params.reward_collision
                elif self.reach_goal(agent):
                    # If reach the goal without collision, the agent will be done and get huge reward
                    terminations[agent] = True
                    rewards[agent] += self.params.reward_goal

            for agent in self.agents:
                # The further the vehicle is away from the goal, the larger the penalty
                rewards[agent] += self.params.reward_dist * self.dist2goal(agent)

                infos[agent]["states"] = self.states[agent].copy()

            observations = {agent: self.observe(agent) for agent in self.agents}

            self.frame += 1

        # Check whether the env reaches its max cycles
        self.cycle_done = self.frame >= self.max_cycles

        # Remove the done agents from the active agent list. This will only affect next iteration
        self.agents = [agent for agent in self.agents if not terminations[agent]]

        # If it reaches the cycle limit, mark all active agents as done
        if self.cycle_done:
            truncations = {agent: True for agent in self.agents}
            self.agents = []

        return observations, rewards, terminations, truncations, infos
