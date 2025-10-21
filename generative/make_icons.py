# generative/make_icons.py
# Make a tiny synthetic dataset of automotive-style icons (battery, warning, engine, door, tyre).
# Saves grayscale 64x64 PNGs into data/custom_gan/.

import os, random, math
from pathlib import Path
from PIL import Image, ImageDraw

OUT_DIR = "data/custom_gan"
SIZE = 64
BG = 0          # black background
FG = 230        # light gray foreground (will be normalized by the GAN pipeline)

random.seed(42)

def _canvas():
    img = Image.new("L", (SIZE, SIZE), BG)
    drw = ImageDraw.Draw(img)
    return img, drw

def icon_battery():
    img, d = _canvas()
    x0, y0 = 10, 20; x1, y1 = SIZE-10, SIZE-20
    d.rectangle([x0, y0, x1, y1], outline=FG, width=3)
    # terminals
    d.rectangle([x0+6, y0-6, x0+14, y0], fill=FG)
    d.rectangle([x1-14, y0-6, x1-6, y0], fill=FG)
    # plus/minus
    d.line([x0+18, y0+10, x0+30, y0+10], fill=FG, width=3) # minus
    d.line([x1-30, y0+6, x1-18, y0+6], fill=FG, width=3)
    d.line([x1-24, y0+0, x1-24, y0+12], fill=FG, width=3) # plus vertical
    return img

def icon_warning():
    img, d = _canvas()
    tri = [(SIZE//2, 8), (8, SIZE-8), (SIZE-8, SIZE-8)]
    d.polygon(tri, outline=FG, fill=None)
    d.line([SIZE//2, 18, SIZE//2, SIZE-22], fill=FG, width=4)
    d.ellipse([SIZE//2-2, SIZE-18, SIZE//2+2, SIZE-14], fill=FG)
    return img

def icon_engine():
    img, d = _canvas()
    d.rectangle([12, 20, SIZE-12, SIZE-24], outline=FG, width=3)
    d.rectangle([20, 12, 36, 20], outline=FG, width=3)  # intake
    d.rectangle([SIZE-36, 12, SIZE-20, 20], outline=FG, width=3)  # exhaust
    # pipes
    d.line([36, 20, 36, SIZE-24], fill=FG, width=2)
    d.line([SIZE-36, 20, SIZE-36, SIZE-24], fill=FG, width=2)
    return img

def icon_door():
    img, d = _canvas()
    d.rounded_rectangle([14, 10, SIZE-14, SIZE-10], radius=10, outline=FG, width=3)
    d.ellipse([SIZE-28, SIZE//2-2, SIZE-22, SIZE//2+4], fill=FG)  # handle
    return img

def icon_tyre():
    img, d = _canvas()
    d.ellipse([10, 10, SIZE-10, SIZE-10], outline=FG, width=3)
    d.ellipse([22, 22, SIZE-22, SIZE-22], outline=FG, width=3)
    for a in range(0, 360, 30):
        rad = math.radians(a)
        cx, cy = SIZE//2, SIZE//2
        r1, r2 = 12, 20
        x1, y1 = cx + r1*math.cos(rad), cy + r1*math.sin(rad)
        x2, y2 = cx + r2*math.cos(rad), cy + r2*math.sin(rad)
        d.line([x1, y1, x2, y2], fill=FG, width=2)
    return img

ICONS = [icon_battery, icon_warning, icon_engine, icon_door, icon_tyre]

def make_set(n_per_class=20):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    count = 0
    for fn in ICONS:
        for i in range(n_per_class):
            img = fn()
            # random small translation/scale jitter
            img = img.transform(img.size, Image.AFFINE,
                                (1, 0, random.randint(-2,2), 0, 1, random.randint(-2,2)))
            img.save(os.path.join(OUT_DIR, f"{fn.__name__}_{i:03d}.png"))
            count += 1
    print(f"Saved {count} images to {OUT_DIR}")

if __name__ == "__main__":
    make_set(n_per_class=20)