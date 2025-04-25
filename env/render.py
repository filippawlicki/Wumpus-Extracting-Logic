import pygame


class Renderer:
    def __init__(self, env, tile_size=100):
        pygame.init()
        self.env = env
        self.tile_size = tile_size
        self.grid_size = env.grid_size
        self.width = tile_size * self.grid_size
        self.height = tile_size * self.grid_size + 60
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Wumpus World")

        self.font = pygame.font.SysFont("Arial", 16)

        self.colors = {
            "agent": (0, 0, 255),
            "wumpus": (150, 0, 0),
            "pit": (50, 50, 50),
            "gold": (255, 215, 0),
            "background": (240, 240, 240),
            "grid": (180, 180, 180),
            "text": (0, 0, 0),
        }

    def draw_tile(self, x, y, contents):
        rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, self.colors["background"], rect)
        pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

        symbols = []
        if contents.get("gold"):
            symbols.append(("G", self.colors["gold"]))
        if contents.get("pit"):
            symbols.append(("P", self.colors["pit"]))
        if contents.get("wumpus") and self.env.wumpus_alive:
            symbols.append(("W", self.colors["wumpus"]))
        if contents.get("agent"):
            arrows = ["↑", "→", "↓", "←"]
            dir_arrow = arrows[self.env.agent_dir]
            symbols.append((dir_arrow, self.colors["agent"]))

        for i, (char, color) in enumerate(symbols):
            text = self.font.render(char, True, color)
            self.screen.blit(text, (x * self.tile_size + 10, y * self.tile_size + 10 + i * 20))

    def draw_observation_bar(self, obs):
        stench, breeze, glitter, bump, scream = obs
        msg = f"Stench: {'Yes' if stench else 'No'} | Breeze: {'Yes' if breeze else 'No'} | Glitter: {'Yes' if glitter else 'No'} | Bump: {'Yes' if bump else 'No'} | Scream: {'Yes' if scream else 'No'}"
        text = self.font.render(msg, True, self.colors["text"])
        pygame.draw.rect(self.screen, (200, 200, 200), (0, self.grid_size * self.tile_size, self.width, 60))
        self.screen.blit(text, (10, self.grid_size * self.tile_size + 20))

    def render(self, observation):
        self.screen.fill(self.colors["background"])

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                contents = {}
                if (x, y) == self.env.agent_pos:
                    contents["agent"] = True
                if (x, y) == self.env.gold_pos and not self.env.agent_has_gold:
                    contents["gold"] = True
                if (x, y) in self.env.pit_pos:
                    contents["pit"] = True
                if (x, y) == self.env.wumpus_pos and self.env.wumpus_alive:
                    contents["wumpus"] = True
                self.draw_tile(x, y, contents)

        self.draw_observation_bar(observation)
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
