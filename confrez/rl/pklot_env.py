import functools
from itertools import product
from typing import Dict, Set, Tuple

from gym.spaces import Box, Discrete
import numpy as np
import pygame
from gym.utils import EzPickle, seeding

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec

FPS = 5


def get_image(path):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


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
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pklot",
        "is_parallelizable": True,
        "render_fps": FPS,
        # "has_manual_policy": True,
    }

    def __init__(self, n_vehicles=4, max_cycles=100):
        EzPickle.__init__(self, n_vehicles, max_cycles)
        self.n_vehicles = n_vehicles

        self.dt = 1.0 / self.metadata["render_fps"]

        self.n_center_grids = 8  # The center region has 8x8 grids
        self.n_edge_grids = 3  # The edge surrounding the center is 3 grid-wide

        self.n_total_grids = self.n_center_grids + 2 * self.n_edge_grids

        self.grid_size = 20
        self.window_size = self.n_total_grids * self.grid_size

        self.possible_agents = ["vehicle_" + str(i) for i in range(self.n_vehicles)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.n_vehicles)))
        )

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

        # Action space for each vehicle has 7 discrete choices: stop, forward in 3 directions, backward in 3 directions
        self.action_spaces = dict(
            zip(self.possible_agents, [Discrete(7)] * self.n_vehicles)
        )

        # State space for each vehicle is the front and back of the vehicle at different grids
        self.state_spaces = dict(
            zip(
                self.possible_agents,
                [
                    Box(
                        low=0,
                        high=self.n_total_grids - 1,
                        shape=(2, 2),
                        dtype=np.uint8,
                    )
                ]
                * self.n_vehicles,
            )
        )

        # Occupancy
        self.occupancy = {
            xy: set() for xy in product(range(self.n_total_grids), repeat=2)
        }

        self.states = {
            agent: {"front": None, "back": None} for agent in self.possible_agents
        }
        self.goals = {
            agent: {"front": None, "back": None} for agent in self.possible_agents
        }
        self.states_history = {agent: [] for agent in self.possible_agents}

        # Discrete action indices to move step length and turning direction
        self._action_to_inputs = {
            0: [0, 0],
            1: [1, -np.pi / 4],
            2: [1, 0],
            3: [1, np.pi / 4],
            4: [-1, -np.pi / 4],
            5: [-1, 0],
            6: [-1, np.pi / 4],
        }

        # Function to return the index of the reverse action. e.g. the reverse of action #1 [1, -np.pi / 4] is action #6 [-1, np.pi / 4], the reverse of action #0 is still 0
        self.reverse_action = lambda x: (7 - x) % 7

        self.max_cycles = max_cycles
        self.frame = 0
        self.cycle_done = False

        self.closed = False
        self.seed()

        pygame.init()
        self.renderOn = False
        self.screen = pygame.Surface(
            (self.window_size, self.window_size)
        )  # Now it is just a normal Surface. When the human render mode is chosen, it will be changed to `display`

        self.background = get_image("obstacle_map.png")
        self.screen.blit(self.background, (0, 0))

        self._upper_wall_idxs = [3, 4, 5, 7, 8, 10]
        self._lower_wall_idxs = [3, 4, 5, 7, 9, 10]

        self._other_front_color = (255, 85, 0)
        self._other_back_color = (191, 115, 77)
        self._ego_front_color = (64, 255, 0)
        self._ego_back_color = (92, 176, 62)
        self._goal_front_color = (255, 255, 255)
        self._goal_back_color = (179, 179, 179)

        self.init_vehicle_walls()

    def g2i(self, x: int, y: int) -> Tuple[int, int]:
        """
        convert right-hand grid indices to its top-left vertex pixel coordinates
        """
        return x * self.grid_size, (self.n_total_grids - y - 1) * self.grid_size

    def init_vehicle_walls(self):
        """
        initialize the initial & terminal states of the vehicle, and the location of the walls

        For now, it is a deterministic case
        """
        # Reset the occupancy map
        self.occupancy = {
            xy: set() for xy in product(range(self.n_total_grids), repeat=2)
        }

        if "vehicle_0" in self.possible_agents:
            self.states["vehicle_0"] = {"front": (6, 8), "back": (6, 7)}
            self.occupancy[(6, 8)].add("vehicle_0")
            self.occupancy[(6, 7)].add("vehicle_0")
            self.goals["vehicle_0"] = {"front": (12, 6), "back": (11, 6)}

        if "vehicle_1" in self.possible_agents:
            self.states["vehicle_1"] = {"front": (8, 7), "back": (9, 7)}
            self.occupancy[(8, 7)].add("vehicle_1")
            self.occupancy[(9, 7)].add("vehicle_1")
            self.goals["vehicle_1"] = {"front": (6, 3), "back": (6, 4)}

        if "vehicle_2" in self.possible_agents:
            self.states["vehicle_2"] = {"front": (6, 5), "back": (6, 4)}
            self.occupancy[(6, 5)].add("vehicle_2")
            self.occupancy[(6, 4)].add("vehicle_2")
            self.goals["vehicle_2"] = {"front": (1, 7), "back": (2, 7)}

        if "vehicle_3" in self.possible_agents:
            self.states["vehicle_3"] = {"front": (5, 6), "back": (4, 6)}
            self.occupancy[(5, 6)].add("vehicle_3")
            self.occupancy[(4, 6)].add("vehicle_3")
            self.goals["vehicle_3"] = {"front": (6, 10), "back": (6, 9)}

        # ============ Walls
        # ==== Top
        for x, y in product(
            range(self.n_total_grids),
            range(self.n_total_grids - self.n_edge_grids, self.n_total_grids),
        ):
            self.occupancy[(x, y)].add("wall")

        # ==== Down
        for x, y in product(range(self.n_total_grids), range(self.n_edge_grids)):
            self.occupancy[(x, y)].add("wall")

        # ==== Left
        for x, y in product(
            range(self.n_edge_grids),
            range(self.n_edge_grids, self.n_edge_grids + self.n_center_grids),
        ):
            self.occupancy[(x, y)].add("wall")

        # Leave spaces for driving lanes
        for x, y in product(
            range(1, self.n_edge_grids),
            range(self.n_edge_grids + 3, self.n_edge_grids + 5),
        ):
            self.occupancy[(x, y)].discard("wall")

        # ==== Right
        for x, y in product(
            range(self.n_edge_grids + self.n_center_grids, self.n_total_grids),
            range(self.n_edge_grids, self.n_edge_grids + self.n_center_grids),
        ):
            self.occupancy[(x, y)].add("wall")

        # Leave spaces for driving lanes
        for x, y in product(
            range(
                self.n_edge_grids + self.n_center_grids,
                self.n_edge_grids + self.n_center_grids + 2,
            ),
            range(self.n_edge_grids + 3, self.n_edge_grids + 5),
        ):
            self.occupancy[(x, y)].discard("wall")

        # Center operating region

        for i in self._upper_wall_idxs:
            self.occupancy[(i, self.n_edge_grids + self.n_center_grids - 1)].add("wall")
            self.occupancy[(i, self.n_edge_grids + self.n_center_grids - 2)].add("wall")
            self.occupancy[(i, self.n_edge_grids + self.n_center_grids - 3)].add("wall")

        for i in self._lower_wall_idxs:
            self.occupancy[(i, self.n_edge_grids)].add("wall")
            self.occupancy[(i, self.n_edge_grids + 1)].add("wall")
            self.occupancy[(i, self.n_edge_grids + 2)].add("wall")

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
        # Delta moving distance and heading angle
        d, a = self._action_to_inputs[action]

        current_front = self.states[agent]["front"]
        current_back = self.states[agent]["back"]

        # Un-register this agent from the oocupancy
        self.occupancy[current_front].discard(agent)
        self.occupancy[current_back].discard(agent)

        current_angle = np.arctan2(
            current_front[1] - current_back[1], current_front[0] - current_back[0]
        )

        if d == 0:
            # The car stops, then we don't change anything
            new_back = current_back
            new_front = current_front
        elif d > 0:
            # The car moves forward
            new_back = current_front

            new_angle = current_angle + a
            dx = int(d * np.rint(np.cos(new_angle)))
            dy = int(d * np.rint(np.sin(new_angle)))
            new_front = (current_front[0] + dx, current_front[1] + dy)
        else:
            # The car moves backward
            new_front = current_back

            new_angle = current_angle + a
            dx = int(d * np.rint(np.cos(new_angle)))
            dy = int(d * np.rint(np.sin(new_angle)))
            new_back = (current_back[0] + dx, current_back[1] + dy)

        hit_wall = False
        # Check the walls
        if ("wall" in self.occupancy[new_front]) or (
            "wall" in self.occupancy[new_back]
        ):
            # If the new front or back hits the wall, revert to old values
            new_front = current_front
            new_back = current_back
            hit_wall = True
        else:
            # Otherwise the new states are fine, `hit_wall` is still False
            pass

        self.states[agent]["front"] = new_front
        self.states[agent]["back"] = new_back

        # Register the current states of the agent into occupancy dict
        self.occupancy[new_front].add(agent)
        self.occupancy[new_back].add(agent)

        return hit_wall

    def unregister_agent(self, agent: str):
        """
        remove an agent from occupancy dict
        """
        front = self.states[agent]["front"]
        back = self.states[agent]["back"]

        # Un-register this agent from the oocupancy
        self.occupancy[front].discard(agent)
        self.occupancy[back].discard(agent)

    def has_collision(self, agent: str) -> bool:
        """
        check whether this agent collide with other agents. Note: the case of hitting the wall is already considered in `self.move` function
        """
        front = self.states[agent]["front"]
        back = self.states[agent]["back"]

        if len(self.occupancy[front]) > 1 or len(self.occupancy[back]) > 1:
            return True
        else:
            return False

    def reach_goal(self, agent: str) -> bool:
        """
        check whether this agent has reached its goal
        """
        front = self.states[agent]["front"]
        back = self.states[agent]["back"]

        if front == self.goals[agent]["front"] and back == self.goals[agent]["back"]:
            return True
        else:
            return False

    def dist2goal(self, agent: str) -> float:
        """
        calculate the distance to goal
        """
        front = self.states[agent]["front"]
        back = self.states[agent]["back"]
        center = (np.array(front) + np.array(back)) / 2

        goal_front = self.goals[agent]["front"]
        goal_back = self.goals[agent]["back"]
        goal_center = (np.array(goal_front) + np.array(goal_back)) / 2

        return np.linalg.norm(center - goal_center)

    def draw_car(self, agent: str, ego: bool = False, surf: pygame.Surface = None):
        """
        draw the specified agent at current time step
        `ego`: if set to be `True`, use the ego color to draw it.
        """

        if surf is None:
            surf = self.screen

        front = self.states[agent]["front"]
        back = self.states[agent]["back"]

        front_rect = pygame.Rect(
            self.g2i(*front),
            (self.grid_size, self.grid_size),
        )

        back_rect = pygame.Rect(
            self.g2i(*back),
            (self.grid_size, self.grid_size),
        )

        if ego:
            pygame.draw.rect(surface=surf, color=self._ego_front_color, rect=front_rect)
            pygame.draw.rect(surface=surf, color=self._ego_back_color, rect=back_rect)
        else:
            pygame.draw.rect(
                surface=surf, color=self._other_front_color, rect=front_rect
            )
            pygame.draw.rect(surface=surf, color=self._other_back_color, rect=back_rect)

    def draw_goal(self, agent: str, surf: pygame.Surface = None):
        """
        draw the goal of the specified agent, if the goal location is not currently occupied
        `surf`: the pygame surface to plot on
        """
        if surf is None:
            surf = self.screen

        front = self.goals[agent]["front"]
        back = self.goals[agent]["back"]

        if len(self.occupancy[front]) == 0:
            front_rect = pygame.Rect(
                self.g2i(*front),
                (self.grid_size, self.grid_size),
            )
            pygame.draw.rect(
                surface=surf, color=self._goal_front_color, rect=front_rect
            )

        if len(self.occupancy[back]) == 0:
            back_rect = pygame.Rect(
                self.g2i(*back),
                (self.grid_size, self.grid_size),
            )
            pygame.draw.rect(surface=surf, color=self._goal_back_color, rect=back_rect)

    def draw(self):
        """
        draw background, walls, and agents
        """
        self.screen.fill((0, 0, 0))
        # Draw background
        self.screen.blit(self.background, (0, 0))

        # Draw agents
        for agent in self.agents:
            self.draw_car(agent)

    def observe(self, agent: str) -> np.ndarray:
        """
        get agent-specific observation. Here we use the current drawing of the environment, but fill a different color for the agent itself, and also plot the goal positions on the image.

        This function should be called after `self.draw()` is called
        """
        surf = self.screen.copy()

        # Re-plot the car itself with ego color again
        self.draw_car(agent=agent, ego=True, surf=surf)
        self.draw_goal(agent=agent, surf=surf)

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

    def render(self, mode="human"):
        """
        Renders the environment
        """
        if mode == "human" and not self.renderOn:
            # sets self.renderOn to true and initializes display
            self.enable_render()

        self.draw()

        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
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

        self.agents = self.possible_agents[:]

        self.states = {agent: {"front": None, "back": None} for agent in self.agents}
        self.goals = {agent: {"front": None, "back": None} for agent in self.agents}

        self.frame = 0
        self.cycle_done = False

        # Occupancy
        self.occupancy = {
            xy: set() for xy in product(range(self.n_total_grids), repeat=2)
        }

        self.init_vehicle_walls()

        self.draw()

        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations

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
        dones = {
            agent: False for agent in self.agents
        }  # All agents that are in self.agents should be active, thus not done
        infos = {agent: {} for agent in self.agents}

        if not self.cycle_done:
            # Move agents with actions simultaneously
            for agent in self.agents:
                hit_wall = self.move(agent=agent, action=actions[agent])
                if hit_wall:
                    rewards[agent] += -1e3
                self.states_history[agent].append(self.states[agent])

            agents_with_collision = []
            # Check collision or goal completion
            for agent in self.agents:
                if self.has_collision(agent):
                    # If collide with other agents, mark the agent name and apply huge penalty
                    agents_with_collision.append(agent)
                    rewards[agent] += -1e3
                elif self.reach_goal(agent):
                    # If reach the goal, the agent will be done and get huge reward
                    dones[agent] = True
                    rewards[agent] += 1e4
                elif actions[agent] == 0:
                    # If the vehicle remain stationary, it will get a small penalty
                    rewards[agent] += -10
                else:
                    # The time cost for vehicle
                    rewards[agent] += -1

            for agent in agents_with_collision:
                self.move(agent, self.reverse_action(actions[agent]))

            for agent in self.agents:
                # The further the vehicle is away from the goal, the larger the penalty
                rewards[agent] += -self.dist2goal(agent)

            self.draw()

            observations = {agent: self.observe(agent) for agent in self.agents}

            self.frame += 1

        # Check whether the env reaches its max cycles
        self.cycle_done = self.frame >= self.max_cycles

        # If it reaches, mark all active agents as done
        if self.cycle_done:
            dones = {agent: True for agent in self.agents}

        # For agents that are done, un-register them from the occupancy dict
        for agent in self.agents:
            if dones[agent]:
                self.unregister_agent(agent)

        # Remove the done agents from the active agent list. This will only affect next iteration
        self.agents = [agent for agent in self.agents if not dones[agent]]

        return observations, rewards, dones, infos
