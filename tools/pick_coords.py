"""Click on the map to print pixel coordinates. Use to pick finish-line endpoints."""
import sys
import pygame

pygame.init()
img = pygame.image.load(sys.argv[1])
w, h = img.get_size()
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption(f"Click to print coords — {sys.argv[1]}")

points = []
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            points.append((x, y))
            print(f"({x}, {y})")
    screen.blit(img, (0, 0))
    for p in points:
        pygame.draw.circle(screen, (255, 0, 0), p, 5)
    for i in range(0, len(points) - 1, 2):
        pygame.draw.line(screen, (255, 0, 0), points[i], points[i + 1], 2)
    pygame.display.flip()
pygame.quit()