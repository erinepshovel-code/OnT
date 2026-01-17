#!/usr/bin/env python3
"""
light_blocked_unit_circles.py

Generates a "light blocked" visualization:
- A point light at the center.
- For each integer n, place a "blocker" at angle n degrees and radius (n/max_n)*R.
- Draw multiple concentric circles (default radii: 1, 4, 8) with degree tick marks.

Designed to run in Pydroid 3.
Outputs a PNG by default.

Example (Pydroid terminal):
  python light_blocked_unit_circles.py --outfile blocked.png
  python light_blocked_unit_circles.py --radii 1,4 --max_n 360 --zoom 1.2 --tick_step 15 --label_step 45
"""

import math
import argparse
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt


def is_prime(n: int) -> bool:
    """Simple deterministic primality test good for small n (<= a few million)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


@dataclass
class Style:
    bg: str
    circle_color: str
    tick_color: str
    label_color: str
    light_color: str
    prime_color: str
    composite_color: str
    one_color: str


def parse_radii(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    radii = []
    for p in parts:
        try:
            radii.append(float(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad radius value: {p!r}")
    if not radii:
        raise argparse.ArgumentTypeError("No radii provided.")
    return radii


def polar_to_xy(r: float, deg: float):
    th = math.radians(deg)
    return r * math.cos(th), r * math.sin(th)


def draw_circle(ax, R, style: Style, circle_lw=1.5):
    circle = plt.Circle((0, 0), R, fill=False, linewidth=circle_lw, color=style.circle_color)
    ax.add_patch(circle)


def draw_ticks_and_labels(
    ax,
    R,
    tick_step: int,
    label_step: int,
    tick_len_frac: float,
    label_offset_frac: float,
    style: Style,
    font_size: int,
):
    # ticks
    for deg in range(0, 360, tick_step):
        x1, y1 = polar_to_xy(R, deg)
        x2, y2 = polar_to_xy(R * (1.0 + tick_len_frac), deg)
        ax.plot([x1, x2], [y1, y2], linewidth=1.0, color=style.tick_color)

    # labels
    if label_step > 0:
        for deg in range(0, 360, label_step):
            x, y = polar_to_xy(R * (1.0 + label_offset_frac), deg)
            ax.text(
                x,
                y,
                str(deg),
                fontsize=font_size,
                color=style.label_color,
                ha="center",
                va="center",
            )


def plot_blockers(
    ax,
    R,
    max_n: int,
    point_size: float,
    alpha: float,
    style: Style,
    show_numbers: bool,
    number_step: int,
    number_font: int,
):
    # place blockers at r = (n/max_n)*R, angle = n degrees
    for n in range(1, max_n + 1):
        r = (n / max_n) * R
        deg = n % 360  # keep angle within [0, 359]
        x, y = polar_to_xy(r, deg)

        if n == 1:
            c = style.one_color
        else:
            c = style.prime_color if is_prime(n) else style.composite_color

        ax.scatter([x], [y], s=point_size, c=[c], alpha=alpha, edgecolors="none")

        if show_numbers and (n == 1 or n % number_step == 0 or is_prime(n)):
            ax.text(
                x,
                y,
                str(n),
                fontsize=number_font,
                color=c,
                ha="center",
                va="center",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=parse_radii, default=parse_radii("1,4,8"),
                    help="Comma-separated radii to draw, e.g. 1,4,8")
    ap.add_argument("--max_n", type=int, default=360,
                    help="Largest integer to plot (controls radial scaling). Default 360.")
    ap.add_argument("--tick_step", type=int, default=30,
                    help="Degree tick step around each circle. Default 30.")
    ap.add_argument("--label_step", type=int, default=90,
                    help="Degree label step. 0 to disable. Default 90.")
    ap.add_argument("--zoom", type=float, default=1.15,
                    help="Zoom factor around the largest radius. Default 1.15.")
    ap.add_argument("--point_size", type=float, default=14.0,
                    help="Scatter point size. Default 14.")
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="Point alpha. Default 0.9.")
    ap.add_argument("--show_numbers", action="store_true",
                    help="Annotate points with numbers (can get busy).")
    ap.add_argument("--number_step", type=int, default=30,
                    help="If --show_numbers: label every k plus all primes and 1. Default 30.")
    ap.add_argument("--outfile", type=str, default="light_blocked.png",
                    help="Output PNG filename.")
    ap.add_argument("--dpi", type=int, default=200,
                    help="PNG DPI. Default 200.")
    ap.add_argument("--figsize", type=str, default="10,10",
                    help="Figure size in inches, e.g. 10,10")

    # Color selectors
    ap.add_argument("--bg", type=str, default="black")
    ap.add_argument("--circle_color", type=str, default="white")
    ap.add_argument("--tick_color", type=str, default="white")
    ap.add_argument("--label_color", type=str, default="white")
    ap.add_argument("--light_color", type=str, default="gold")
    ap.add_argument("--prime_color", type=str, default="cyan")
    ap.add_argument("--composite_color", type=str, default="magenta")
    ap.add_argument("--one_color", type=str, default="orange")

    args = ap.parse_args()

    try:
        w, h = [float(x.strip()) for x in args.figsize.split(",")]
    except Exception:
        w, h = 10.0, 10.0

    style = Style(
        bg=args.bg,
        circle_color=args.circle_color,
        tick_color=args.tick_color,
        label_color=args.label_color,
        light_color=args.light_color,
        prime_color=args.prime_color,
        composite_color=args.composite_color,
        one_color=args.one_color,
    )

    radii = sorted(args.radii)
    Rmax = max(radii)

    fig, ax = plt.subplots(figsize=(w, h), dpi=args.dpi)
    fig.patch.set_facecolor(style.bg)
    ax.set_facecolor(style.bg)

    # Draw circles + ticks + blockers
    for R in radii:
        draw_circle(ax, R, style=style, circle_lw=2.0 if R == Rmax else 1.5)
        draw_ticks_and_labels(
            ax,
            R,
            tick_step=max(1, args.tick_step),
            label_step=max(0, args.label_step),
            tick_len_frac=0.025,
            label_offset_frac=0.08,
            style=style,
            font_size=10 if R == Rmax else 9,
        )
        plot_blockers(
            ax,
            R,
            max_n=max(1, args.max_n),
            point_size=args.point_size,
            alpha=args.alpha,
            style=style,
            show_numbers=args.show_numbers,
            number_step=max(1, args.number_step),
            number_font=7 if R == Rmax else 6,
        )

    # Light source at center
    ax.scatter([0], [0], s=140, c=[style.light_color], alpha=1.0, edgecolors="none")

    # Cosmetic / framing
    lim = Rmax * args.zoom
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Title (optional): comment out if you want it clean
    ax.text(0, lim * 0.92, f"Light-Blocked Lattice (max_n={args.max_n})",
            color=style.label_color, ha="center", va="center", fontsize=14)

    plt.tight_layout(pad=0.2)
    plt.savefig(args.outfile, facecolor=style.bg, bbox_inches="tight")
    print(f"Saved: {args.outfile}")


if __name__ == "__main__":
    main()