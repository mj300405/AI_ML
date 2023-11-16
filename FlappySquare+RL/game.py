import pygame
import random
import numpy as np 

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
PIPE_WIDTH, PIPE_HEIGHT = 80, 400
FPS = 60

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (135, 206, 250)
YELLOW = (255, 255, 0)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')
clock = pygame.time.Clock()

# Bird class
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 30))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect(center=(100, SCREEN_HEIGHT // 2))
        self.velocity = 0
        self.gravity = 0.7  # Decreased gravity for smoother fall
        self.time_since_last_flap = 0
        self.alive = True

    def update(self):
        if self.alive:
            self.velocity += self.gravity
            self.rect.y += int(self.velocity)
            self.time_since_last_flap += 1

            # Check if bird hits the ground
            if self.rect.bottom >= SCREEN_HEIGHT:
                self.alive = False
                self.rect.bottom = SCREEN_HEIGHT
                self.velocity = 0

            # Check if bird flies above the top of the screen
            if self.rect.top <= 0:
                self.rect.top = 0
                self.velocity = 0  # Stop the bird from moving further up  

    def jump(self):
        self.velocity = -12  # Increased jump velocity for quicker lift
        self.time_since_last_flap = 0

# Pipe class
class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, ypos, gap_size, pipe_height):
        super().__init__()
        self.image = pygame.Surface((80, pipe_height))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.passed = False
        self.inverted = inverted
        self.GAP_SIZE = gap_size
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = (SCREEN_WIDTH + 80, ypos - gap_size / 2)
            self.gap_center_y = ypos - gap_size / 2 - pipe_height / 2
        else:
            self.rect.topleft = (SCREEN_WIDTH + 80, ypos + gap_size / 2)
            self.gap_center_y = ypos + gap_size / 2 + pipe_height / 2

    def update(self):
        self.rect.x -= 5
        if self.rect.right < 0:
            self.kill()
    
    # Additional helper properties to get the top and bottom of the gap
    @property
    def top(self):
        if self.inverted:
            return self.rect.bottom - self.GAP_SIZE
        else:
            return self.rect.top

    @property
    def bottom(self):
        if self.inverted:
            return self.rect.bottom
        else:
            return self.rect.top + self.GAP_SIZE


class FlappyBirdGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Square')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        
        # Initialize sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.pipes = pygame.sprite.Group()
        
        # Create a bird instance
        self.bird = Bird()
        self.all_sprites.add(self.bird)
        
        # Create initial pipes
        self._create_pipe_pair(SCREEN_WIDTH + 100)
        
        # Initialize score
        self.score = 0
        self.running = True

    def _create_pipe_pair(self, xpos):
        gap_size = 200  # Set your desired gap size
        ypos = random.randint(100 + gap_size // 2, SCREEN_HEIGHT - 100 - gap_size // 2)  # Gap's vertical position

        # Create the bottom pipe
        bottom_pipe = Pipe(False, ypos, gap_size, PIPE_HEIGHT)
        self.pipes.add(bottom_pipe)
        self.all_sprites.add(bottom_pipe)

        # Create the top pipe
        top_pipe = Pipe(True, ypos, gap_size, PIPE_HEIGHT)
        self.pipes.add(top_pipe)
        self.all_sprites.add(top_pipe)


    def _get_next_pipe(self):
        # Find the next pipe that is to the right of the bird
        for pipe in self.pipes:
            if pipe.rect.left > self.bird.rect.right and not pipe.inverted:
                return pipe
        return None


    def get_state(self):
        # Find the next pipe that the bird is approaching
        pipe_ind = 0
        if len(self.pipes) > 1 and self.bird.rect.left > self.pipes.sprites()[0].rect.right:
            pipe_ind = 1  # The bird is between two pipes
        
        next_pipe = self.pipes.sprites()[pipe_ind]
        horizontal_distance = next_pipe.rect.left - self.bird.rect.right  # Distance to the next pipe
        vertical_distance_top = next_pipe.rect.top - self.bird.rect.centery  # Distance from bird to top of the gap
        vertical_distance_bottom = next_pipe.rect.bottom - self.bird.rect.centery  # Distance from bird to bottom of the gap

        # Get bird's vertical velocity
        bird_velocity_y = self.bird.velocity
        # Time since last flap could be maintained as a counter in the Bird class
        time_since_last_flap = self.bird.time_since_last_flap

        # Additional state information such as bird's height could also be included if it affects the gameplay
        bird_height = self.bird.rect.centery

        # Normalize the state before returning it
        state = np.array([
            horizontal_distance / SCREEN_WIDTH,
            vertical_distance_top / SCREEN_HEIGHT,
            vertical_distance_bottom / SCREEN_HEIGHT,
            bird_height / SCREEN_HEIGHT,
            bird_velocity_y / 10,  # Assuming the bird's velocity ranges around [-10, 10]
            time_since_last_flap / FPS  # Normalized assuming FPS is the maximum time between flaps
        ], dtype=np.float32)

        return state

    def _check_passed_pipe(self):
        pipe_passed = False
        pipe_passed_recently = False
        
        # Logic to determine if a pipe was passed
        # This could be based on the bird's position relative to the pipes
        # You need to store the state of the pipes and whether they've been passed or not
        for pipe in self.pipes:
            if not pipe.passed and pipe.rect.right < self.bird.rect.left:
                pipe.passed = True
                pipe_passed = True
                pipe_passed_recently = True
        
        return pipe_passed, pipe_passed_recently

    def step(self, action):
        # Update game state
        self.all_sprites.update()
        self._handle_pipes()

        reward = 0.01  # Reward for living

        # Make one step in the game given the agent's action.
        if action == 1:
            self.bird.jump()
        
        # Increase reward for passing a pipe
        for pipe in self.pipes:
            if not pipe.passed and pipe.rect.right < self.bird.rect.left:
                pipe.passed = True
                reward += 10.0  # Reward for passing a pipe

        # Calculate distances to the next pipe gap
        next_pipe = self._get_next_pipe()
        if next_pipe:
            gap_center_y = next_pipe.gap_center_y
            vertical_distance_to_gap_center = abs(self.bird.rect.centery - gap_center_y)
            horizontal_distance_to_pipe = next_pipe.rect.left - self.bird.rect.right

            # Adjust rewards and penalties based on proximity to the pipe
            reward -= vertical_distance_to_gap_center * 0.005  # Penalty based on vertical distance
            if horizontal_distance_to_pipe < 200:
                reward += (200 - horizontal_distance_to_pipe) * 0.01  # Reward for approaching the pipe

        # Penalties for risky behavior
        if action == 1:  # Penalty for flapping
            reward -= 0.1
        if abs(self.bird.velocity) > 5:  # Penalty for high vertical speed
            reward -= 0.05
        if self.bird.rect.top <= 0 or self.bird.rect.bottom >= SCREEN_HEIGHT:  # Penalty for flying too high or low
            reward -= 5

        # Penalty for dying
        if pygame.sprite.spritecollideany(self.bird, self.pipes) or not self.bird.alive:
            reward = -15
            self.running = False

        return self.get_state(), reward, not self.running


    def _handle_pipes(self):
        # Create new pipes and update existing pipes.
        for pipe in self.pipes:
            if pipe.rect.right < 0:
                pipe.kill()
        if len(self.pipes) == 0 or all(pipe.rect.right < SCREEN_WIDTH for pipe in self.pipes):
            self._create_pipe_pair(SCREEN_WIDTH)

    def render(self):
        # Clear the screen
        self.screen.fill(BLUE)
        # Draw all sprites
        self.all_sprites.draw(self.screen)
        # Draw the score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        # Update the full display Surface to the screen
        pygame.display.flip()
        # Control the frame rate to make it visible
        self.clock.tick(FPS)  # Ensure FPS is set to a reasonable number



    def reset(self):
        # Reset the game to the initial state
        self.__init__()
        return self.get_state()

    def run_game_loop(self):
        # Main game loop
        running = True
        while running:
            self.screen.fill(BLUE)
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.bird.jump()
            
            # Update game state
            self.all_sprites.update()
            
            # Check for collisions
            if pygame.sprite.spritecollideany(self.bird, self.pipes) or not self.bird.alive:
                running = False  # Game over
            
            # Check if the bird has passed the pipe
            for pipe in self.pipes:
                if not pipe.passed and pipe.rect.right < self.bird.rect.left:
                    pipe.passed = True
                    self.score += 1
            
            # Create new pipes
            # Inside the game loop
            if not self.pipes or all(pipe.rect.right < SCREEN_WIDTH // 2 for pipe in self.pipes):
                self._create_pipe_pair(SCREEN_WIDTH + PIPE_WIDTH)

            
            # Draw everything
            self.all_sprites.draw(self.screen)
            
            # Score display
            score_text = self.font.render(f'Score: {self.score}', True, WHITE)
            self.screen.blit(score_text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

# If you run this script directly, it will play the game with keyboard input.
if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run_game_loop()