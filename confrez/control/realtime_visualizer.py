import time
from typing import Tuple
import pygame
import numpy as np
from confrez.control.compute_sets import compute_obstacles

from confrez.pytypes import VehiclePrediction, VehicleState
from confrez.vehicle_types import VehicleBody


class RealtimeVisualizer(object):
    """
    real tinme visualizer using pygame
    """

    def __init__(
        self,
        vehicle_body: VehicleBody,
        width: int = 35,
        height: int = 28,
        res: int = 30,
    ) -> None:
        self.vehicle_body = vehicle_body

        self.res = res
        self.width = width
        self.height = height

        self.top_offset = 3

        self.obstacles = compute_obstacles()

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width * self.res, self.height * self.res)
        )
        self.draw_background()

    def __del__(self):
        self.close()

    def g2i(self, x: int, y: int) -> Tuple[int, int]:
        """
        convert right-hand grid indices to its top-left vertex pixel coordinates
        """
        return x * self.res, (self.height - y + self.top_offset) * self.res

    def draw_background(self):
        """
        draw background
        """
        self.screen.fill((255, 255, 255))

    def draw_obstacles(self):
        """
        draw static obstacles
        """
        for obs in self.obstacles:
            points = []
            for v in obs.V:
                points.append(list(self.g2i(*v)))

            pygame.draw.polygon(self.screen, (0, 125, 255), points)

    def draw_traj(
        self, traj: VehiclePrediction, color: Tuple[int, int, int] = (0, 0, 0)
    ):
        px, py = self.g2i(np.array(traj.x), np.array(traj.y))

        pygame.draw.lines(self.screen, color, False, np.vstack([px, py]).T, width=5)
        # pygame.draw.aalines(self.screen, color, False, np.vstack([px, py]).T)

    def draw_car(self, state: VehicleState, color: Tuple[int, int, int] = (0, 0, 0)):
        """
        draw a moving car
        """
        R = state.get_R()[:2, :2]
        car_outline = self.vehicle_body.xy @ R.T + np.array([state.x.x, state.x.y])

        px, py = self.g2i(car_outline[:, 0], car_outline[:, 1])

        pygame.draw.polygon(self.screen, color, np.vstack([px, py]).T)

    def render(self):
        """
        render current frame
        """
        pygame.display.flip()

    def test_draw(self):
        """
        draw update
        """
        state = VehicleState()
        state.x.x = 5
        state.x.y = 10
        state.e.psi = np.pi / 6
        for i in range(25):
            self.draw_background()
            self.draw_obstacles()

            state.x.x += 1
            self.draw_car(state)

            self.render()

            time.sleep(0.05)

    def close(self):
        """
        close the visualizer
        """
        pygame.event.pump()
        pygame.display.quit()


def main():
    """
    main
    """
    vis = RealtimeVisualizer(vehicle_body=VehicleBody())

    vis.test_draw()

    time.sleep(2)

    # vis.close()


if __name__ == "__main__":
    main()
