from collections import deque


DOWN_THRESHOLD = 160
UP_THRESHOLD = 60
WINDOW_SIZE = 5


class BicepCurlCounter:

    def __init__(self):
        self.state = "DOWN"
        self.reps = 0
        self.angle_history = deque(maxlen=WINDOW_SIZE)

    def update(self, angle):

        if angle is None:
            return self.reps

        self.angle_history.append(angle)

        smoothed_angle = sum(self.angle_history) / len(self.angle_history)

        if smoothed_angle <= UP_THRESHOLD:
            self.state = "UP"

        elif smoothed_angle >= DOWN_THRESHOLD:

            if self.state == "UP":
                self.reps += 1

            self.state = "DOWN"

        return self.reps, self.state, smoothed_angle