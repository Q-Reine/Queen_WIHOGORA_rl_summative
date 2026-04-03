"""
Pygame Visualization for the Assistive Technology Rehabilitation Center.
Renders a top-down view of the rehab center with patient queue, device
inventory, therapy status, patient outcomes, action log, and reward panels.
"""

import math
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ---- Colors ----
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BG = (30, 35, 45)
PANEL_BG = (40, 45, 55)
BORDER = (70, 80, 100)

# Center colors
FLOOR_LIGHT = (200, 210, 220)
FLOOR_DARK = (160, 170, 185)
WALL_COLOR = (100, 110, 130)

# Status colors
HEALTH_HIGH = (50, 200, 80)
HEALTH_MID = (220, 200, 40)
HEALTH_LOW = (220, 60, 50)
WATER_BLUE = (60, 140, 220)
TEAL = (0, 180, 170)
WHEELCHAIR_BLUE = (70, 130, 200)
PROSTHETIC_ORANGE = (220, 140, 50)
HEARING_GREEN = (80, 190, 100)
THERAPY_PURPLE = (150, 100, 200)
GOLD = (220, 200, 60)
SATISFACTION_PINK = (220, 100, 140)
DEVICE_GRAY = (140, 150, 165)
IMPACT_CYAN = (60, 200, 220)

# Layout
WINDOW_W = 900
WINDOW_H = 650
CENTER_X, CENTER_Y = 20, 60
CENTER_W, CENTER_H = 420, 360
PANEL_X = 460
PANEL_W = 420


def _lerp_color(c1, c2, t):
    """Linear interpolation between two colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _status_color(value):
    """Return green/yellow/red based on value [0,1]."""
    if value > 0.6:
        return HEALTH_HIGH
    elif value > 0.3:
        return HEALTH_MID
    return HEALTH_LOW


def _draw_bar(surface, x, y, w, h, value, color, label, font, max_label=None):
    """Draw a labeled progress bar."""
    pygame.draw.rect(surface, (60, 60, 70), (x, y, w, h), border_radius=3)
    fill_w = max(0, int(w * min(value, 1.0)))
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=3)
    pygame.draw.rect(surface, BORDER, (x, y, w, h), 1, border_radius=3)
    lbl = font.render(label, True, WHITE)
    surface.blit(lbl, (x, y - 16))
    if max_label:
        val_text = font.render(max_label, True, WHITE)
    else:
        val_text = font.render(f"{value:.0%}", True, WHITE)
    surface.blit(val_text, (x + w + 5, y + 1))


def _init_pygame(env):
    """Initialize pygame if needed."""
    if not PYGAME_AVAILABLE:
        raise ImportError("pygame is required for rendering. pip install pygame")

    if env._screen is None:
        pygame.init()
        if env.render_mode == "human":
            env._screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption(
                "Assistive Tech Rehab Center - Kigali, Rwanda")
        else:
            env._screen = pygame.Surface((WINDOW_W, WINDOW_H))
        env._clock = pygame.time.Clock()
        env._font = pygame.font.SysFont("consolas", 13)
        env._font_med = pygame.font.SysFont("consolas", 15, bold=True)
        env._font_large = pygame.font.SysFont("consolas", 18, bold=True)
        env._font_title = pygame.font.SysFont("consolas", 20, bold=True)


def _draw_header(screen, env):
    """Draw the top header bar."""
    pygame.draw.rect(screen, (25, 30, 40), (0, 0, WINDOW_W, 50))
    pygame.draw.line(screen, BORDER, (0, 50), (WINDOW_W, 50), 2)

    title = env._font_title.render(
        "ASSISTIVE TECH REHAB CENTER - Kigali", True, TEAL)
    screen.blit(title, (15, 5))

    center = "Urban (Kigali)" if env.center_type == 0 else "Rural Outreach"
    info = env._font_med.render(
        f"Day {env.day}/{env.max_days}  |  {center}  |  "
        f"Budget: {env.budget:.0f}/{env.initial_budget:.0f}", True, WHITE)
    screen.blit(info, (15, 28))


def _draw_center_view(screen, env):
    """Draw the main rehabilitation center view."""
    # Floor
    pygame.draw.rect(screen, FLOOR_LIGHT, (CENTER_X, CENTER_Y, CENTER_W, CENTER_H))
    pygame.draw.rect(screen, BORDER, (CENTER_X, CENTER_Y, CENTER_W, CENTER_H), 2)

    # Grid floor pattern
    for i in range(0, CENTER_W, 40):
        pygame.draw.line(screen, FLOOR_DARK,
                         (CENTER_X + i, CENTER_Y), (CENTER_X + i, CENTER_Y + CENTER_H), 1)
    for j in range(0, CENTER_H, 40):
        pygame.draw.line(screen, FLOOR_DARK,
                         (CENTER_X, CENTER_Y + j), (CENTER_X + CENTER_W, CENTER_Y + j), 1)

    # ---- Waiting Area (top-left) ----
    wx, wy = CENTER_X + 10, CENTER_Y + 10
    pygame.draw.rect(screen, (220, 225, 235), (wx, wy, 190, 100), border_radius=5)
    pygame.draw.rect(screen, WALL_COLOR, (wx, wy, 190, 100), 2, border_radius=5)
    label = env._font_med.render("WAITING AREA", True, WALL_COLOR)
    screen.blit(label, (wx + 5, wy + 3))

    # Draw patient icons in waiting area
    for i in range(min(env.patients_waiting, 12)):
        px = wx + 15 + (i % 6) * 28
        py = wy + 30 + (i // 6) * 30
        # Person icon (circle head + body)
        pygame.draw.circle(screen, (100, 120, 160), (px, py), 7)
        pygame.draw.line(screen, (100, 120, 160), (px, py + 7), (px, py + 18), 2)

    wait_text = env._font.render(f"{env.patients_waiting} waiting", True, HEALTH_LOW
                                  if env.patients_waiting > 5 else WHITE)
    screen.blit(wait_text, (wx + 5, wy + 82))

    # ---- Rehab Zone (bottom-left) ----
    rx, ry = CENTER_X + 10, CENTER_Y + 125
    pygame.draw.rect(screen, (230, 245, 230), (rx, ry, 190, 120), border_radius=5)
    pygame.draw.rect(screen, HEALTH_HIGH, (rx, ry, 190, 120), 2, border_radius=5)
    label = env._font_med.render("REHAB ZONE", True, (40, 120, 60))
    screen.blit(label, (rx + 5, ry + 3))

    # Draw active patients with progress
    active = getattr(env, '_active_patients', [])
    for i, p in enumerate(active[:6]):
        px = rx + 20 + (i % 3) * 60
        py = ry + 30 + (i // 3) * 40

        # Progress color
        prog_color = _status_color(p["progress"])
        pygame.draw.circle(screen, prog_color, (px, py), 9)

        # Device icon
        device = p.get("device", "")
        if device == "wheelchair":
            pygame.draw.rect(screen, WHEELCHAIR_BLUE, (px - 5, py + 10, 10, 6), border_radius=2)
        elif device == "prosthetic":
            pygame.draw.line(screen, PROSTHETIC_ORANGE, (px, py + 10), (px, py + 20), 3)
        elif device == "hearing_aid":
            pygame.draw.circle(screen, HEARING_GREEN, (px + 8, py - 4), 4, 1)

        prog_text = env._font.render(f"{p['progress']:.0%}", True, BLACK)
        screen.blit(prog_text, (px - 12, py + 22))

    rehab_text = env._font.render(f"{env.patients_in_rehab} in rehab", True, (40, 120, 60))
    screen.blit(rehab_text, (rx + 5, ry + 102))

    # ---- Device Storage (top-right) ----
    dx, dy = CENTER_X + 215, CENTER_Y + 10
    pygame.draw.rect(screen, (235, 230, 220), (dx, dy, 190, 100), border_radius=5)
    pygame.draw.rect(screen, DEVICE_GRAY, (dx, dy, 190, 100), 2, border_radius=5)
    label = env._font_med.render("DEVICE STORAGE", True, (100, 90, 70))
    screen.blit(label, (dx + 5, dy + 3))

    # Device icons with counts
    items = [
        (WHEELCHAIR_BLUE, "Wheelchairs", env.wheelchair_stock),
        (PROSTHETIC_ORANGE, "Prosthetics", env.prosthetic_stock),
        (HEARING_GREEN, "Hearing Aids", env.hearing_aid_stock),
    ]
    for j, (color, name, count) in enumerate(items):
        iy = dy + 25 + j * 22
        pygame.draw.rect(screen, color, (dx + 10, iy, 12, 12), border_radius=2)
        text = env._font.render(f"{name}: {count}", True, BLACK)
        screen.blit(text, (dx + 28, iy - 1))

    # ---- Assessment Room (bottom-right) ----
    ax, ay = CENTER_X + 215, CENTER_Y + 125
    pygame.draw.rect(screen, (235, 230, 245), (ax, ay, 190, 120), border_radius=5)
    pygame.draw.rect(screen, THERAPY_PURPLE, (ax, ay, 190, 120), 2, border_radius=5)
    label = env._font_med.render("ASSESSMENT", True, (100, 70, 140))
    screen.blit(label, (ax + 5, ay + 3))

    if env.current_patient_assessed:
        # Show assessed patient
        disability_names = {0.0: "None", 0.33: "Mobility", 0.5: "Amputation",
                            0.67: "Hearing", 1.0: "Multiple"}
        d_name = disability_names.get(env.current_patient_disability, "Unknown")
        pygame.draw.circle(screen, GOLD, (ax + 95, ay + 55), 15)
        d_text = env._font.render(f"Disability: {d_name}", True, (60, 40, 80))
        screen.blit(d_text, (ax + 10, ay + 80))
        u_text = env._font.render(f"Urgency: {env.current_patient_urgency:.0%}", True,
                                   HEALTH_LOW if env.current_patient_urgency > 0.7 else (60, 40, 80))
        screen.blit(u_text, (ax + 10, ay + 98))
    else:
        empty = env._font.render("Ready for assessment", True, (120, 100, 150))
        screen.blit(empty, (ax + 15, ay + 55))

    # ---- Bottom info bar ----
    served_text = env._font_med.render(
        f"Served: {env.patients_served}  |  Impact: {env.community_impact:.0f}",
        True, GOLD)
    screen.blit(served_text, (CENTER_X + 5, CENTER_Y + CENTER_H - 30))

    # Therapist availability
    therapy_text = env._font.render(
        f"Therapists: {env.therapist_availability}/{env.max_therapists}",
        True, THERAPY_PURPLE)
    screen.blit(therapy_text, (CENTER_X + 260, CENTER_Y + CENTER_H - 30))


def _draw_inventory_panel(screen, env):
    """Draw device inventory panel."""
    y = CENTER_Y
    x = PANEL_X
    w = PANEL_W

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, 90), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, 90), 1, border_radius=5)

    title = env._font_med.render("DEVICE INVENTORY", True, WHEELCHAIR_BLUE)
    screen.blit(title, (x + 10, y + 5))

    bar_w = 180
    _draw_bar(screen, x + 15, y + 35, bar_w, 12, env.wheelchair_stock / 15.0,
              WHEELCHAIR_BLUE, "Wheelchairs", env._font, f"{env.wheelchair_stock}")
    _draw_bar(screen, x + 15, y + 55, bar_w, 12, env.prosthetic_stock / 10.0,
              PROSTHETIC_ORANGE, "Prosthetics", env._font, f"{env.prosthetic_stock}")
    _draw_bar(screen, x + 15, y + 75, bar_w, 12, env.hearing_aid_stock / 12.0,
              HEARING_GREEN, "Hearing Aids", env._font, f"{env.hearing_aid_stock}")


def _draw_patient_panel(screen, env):
    """Draw patient status panel."""
    y = CENTER_Y + 100
    x = PANEL_X
    w = PANEL_W

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, 110), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, 110), 1, border_radius=5)

    title = env._font_med.render("PATIENT STATUS", True, HEALTH_HIGH)
    screen.blit(title, (x + 10, y + 5))

    bar_w = 200
    _draw_bar(screen, x + 15, y + 35, bar_w, 14, env.avg_patient_progress,
              _status_color(env.avg_patient_progress), "Avg Progress", env._font)
    _draw_bar(screen, x + 15, y + 62, bar_w, 14, env.patient_satisfaction,
              SATISFACTION_PINK, "Satisfaction", env._font)
    _draw_bar(screen, x + 15, y + 89, bar_w, 14, env.device_condition,
              DEVICE_GRAY, "Device Condition", env._font)


def _draw_impact_panel(screen, env):
    """Draw community impact panel."""
    y = CENTER_Y + 220
    x = PANEL_X
    w = PANEL_W

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, 60), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, 60), 1, border_radius=5)

    title = env._font_med.render("COMMUNITY IMPACT", True, IMPACT_CYAN)
    screen.blit(title, (x + 10, y + 5))

    bar_w = 200
    _draw_bar(screen, x + 15, y + 35, bar_w, 14, env.community_impact / 100.0,
              IMPACT_CYAN, "Impact Score", env._font,
              f"{env.community_impact:.0f}/100")


def _draw_budget_panel(screen, env):
    """Draw budget panel."""
    y = CENTER_Y + 290
    x = PANEL_X
    w = PANEL_W

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, 55), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, 55), 1, border_radius=5)

    title = env._font_med.render("BUDGET", True, GOLD)
    screen.blit(title, (x + 10, y + 5))

    bar_w = 200
    budget_ratio = env.budget / env.initial_budget
    _draw_bar(screen, x + 15, y + 32, bar_w, 14, budget_ratio,
              _status_color(budget_ratio), "Remaining", env._font,
              f"{env.budget:.0f}/{env.initial_budget:.0f}")


def _draw_action_log(screen, env):
    """Draw recent action log."""
    y = CENTER_Y + CENTER_H + 30
    x = CENTER_X
    w = CENTER_W
    h = 190

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, h), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, h), 1, border_radius=5)

    title = env._font_med.render("ACTION LOG", True, WHITE)
    screen.blit(title, (x + 10, y + 5))

    for i, entry in enumerate(env._action_log[-8:]):
        color = HEALTH_HIGH if entry["reward"] > 0 else (
            HEALTH_LOW if entry["reward"] < -1 else WHITE)
        text = env._font.render(
            f"Day {entry['day']:3d}: {entry['action']:<18s} R={entry['reward']:+.1f}",
            True, color)
        screen.blit(text, (x + 10, y + 25 + i * 20))


def _draw_reward_panel(screen, env):
    """Draw cumulative reward display."""
    y = CENTER_Y + 355
    x = PANEL_X
    w = PANEL_W

    pygame.draw.rect(screen, PANEL_BG, (x, y, w, 55), border_radius=5)
    pygame.draw.rect(screen, BORDER, (x, y, w, 55), 1, border_radius=5)

    title = env._font_med.render("TOTAL REWARD", True, WHITE)
    screen.blit(title, (x + 10, y + 5))

    rew_color = HEALTH_HIGH if env.total_reward >= 0 else HEALTH_LOW
    rew_text = env._font_large.render(
        f"{env.total_reward:+.1f}", True, rew_color)
    screen.blit(rew_text, (x + 15, y + 28))

    # Status
    status_parts = []
    if env.patients_served > 0:
        status_parts.append(f"SERVED:{env.patients_served}")
    if env.patients_in_rehab > 0:
        status_parts.append(f"REHAB:{env.patients_in_rehab}")
    if env.patients_waiting > 0:
        status_parts.append(f"QUEUE:{env.patients_waiting}")

    status = env._font.render(" | ".join(status_parts) if status_parts else "STARTING",
                               True, GOLD)
    screen.blit(status, (x + 130, y + 32))


def _draw_footer(screen, env):
    """Draw footer with legend."""
    y = WINDOW_H - 25
    pygame.draw.line(screen, BORDER, (0, y - 5), (WINDOW_W, y - 5), 1)

    legends = [
        (WHEELCHAIR_BLUE, "Wheelchair"),
        (PROSTHETIC_ORANGE, "Prosthetic"),
        (HEARING_GREEN, "Hearing Aid"),
        (THERAPY_PURPLE, "Therapy"),
        (GOLD, "Discharged"),
    ]
    x = 15
    for color, text in legends:
        pygame.draw.circle(screen, color, (x + 5, y + 7), 5)
        lbl = env._font.render(text, True, (180, 180, 180))
        screen.blit(lbl, (x + 15, y))
        x += len(text) * 8 + 40


def render_frame(env):
    """Render a frame to the pygame display (human mode)."""
    _init_pygame(env)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env._screen = None
            return

    env._screen.fill(DARK_BG)

    _draw_header(env._screen, env)
    _draw_center_view(env._screen, env)
    _draw_inventory_panel(env._screen, env)
    _draw_patient_panel(env._screen, env)
    _draw_impact_panel(env._screen, env)
    _draw_budget_panel(env._screen, env)
    _draw_action_log(env._screen, env)
    _draw_reward_panel(env._screen, env)
    _draw_footer(env._screen, env)

    pygame.display.flip()
    env._clock.tick(env.metadata["render_fps"])


def render_to_array(env):
    """Render a frame and return as numpy array (rgb_array mode)."""
    _init_pygame(env)

    env._screen.fill(DARK_BG)

    _draw_header(env._screen, env)
    _draw_center_view(env._screen, env)
    _draw_inventory_panel(env._screen, env)
    _draw_patient_panel(env._screen, env)
    _draw_impact_panel(env._screen, env)
    _draw_budget_panel(env._screen, env)
    _draw_action_log(env._screen, env)
    _draw_reward_panel(env._screen, env)
    _draw_footer(env._screen, env)

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(env._screen)), axes=(1, 0, 2))
