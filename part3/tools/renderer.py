import pygame

class Renderer:
    """Encapsulate pygame drawing to keep env logic clean."""

    def __init__(self, grid_size: int, cell_size: int, fps: int):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.window = None
        self.clock = None

    def draw(self, snake, food, obstacles=None):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((0, 0, 0))

        # Draw food
        fx, fy = food.position
        food_color = getattr(food, "color", (255, 0, 0))
        pygame.draw.rect(
            self.window,
            food_color,
            (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size),
        )

        # Draw obstacles (colored or plain)
        obstacles = obstacles or []
        for item in obstacles:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                (ox, oy), col = item
            else:
                ox, oy = item
                col = (100, 100, 100)
            pygame.draw.rect(
                self.window,
                col,
                (ox * self.cell_size, oy * self.cell_size, self.cell_size, self.cell_size),
            )

        # Draw snake
        for i, (bx, by) in enumerate(snake.body):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                self.window,
                color,
                (bx * self.cell_size, by * self.cell_size, self.cell_size, self.cell_size),
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.window:
            pygame.quit()
