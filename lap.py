"""Lap detection via line-segment intersection with directionality."""
from dataclasses import dataclass


@dataclass
class LapDetector:
    """Detects laps when the car's path crosses a finish line in the correct direction.

    line_p1, line_p2: endpoints of the finish line, in image pixel coords.
    forward_dir: (dx, dy) — the direction the car should be moving when it crosses.
                 e.g. (1, 0) means "must be moving rightward to count".
    """
    line_p1: tuple
    line_p2: tuple
    forward_dir: tuple = (1, 0)

    def __post_init__(self):
        self.lap_count = 0
        self.prev_pos = None

    def reset(self, init_pos):
        self.lap_count = 0
        self.prev_pos = tuple(init_pos)

    @staticmethod
    def _segments_intersect(a, b, c, d) -> bool:
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    def update(self, curr_pos) -> bool:
        """Call once per env step. Returns True iff a lap was just completed."""
        curr_pos = tuple(curr_pos)
        completed = False
        if self.prev_pos is not None:
            crossed = self._segments_intersect(
                self.prev_pos, curr_pos, self.line_p1, self.line_p2
            )
            if crossed:
                dx = curr_pos[0] - self.prev_pos[0]
                dy = curr_pos[1] - self.prev_pos[1]
                dot = dx * self.forward_dir[0] + dy * self.forward_dir[1]
                if dot > 0:
                    self.lap_count += 1
                    completed = True
        self.prev_pos = curr_pos
        return completed