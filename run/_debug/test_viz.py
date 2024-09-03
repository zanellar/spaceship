
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_slits(ax, slit):
    # Dimensions for the larger rectangle
    outer_width = slit["width"] * 3  # Example: 3 times the slit width
    outer_height = slit["height"] * 3  # Example: 3 times the slit height
    outer_x = slit["position"][0] - outer_width / 2
    outer_y = slit["position"][1] - outer_height / 2

    # Create the larger green rectangle
    outer_rect = patches.Rectangle((outer_x, outer_y), outer_width, outer_height, linewidth=1, edgecolor='r', facecolor='green')
    ax.add_patch(outer_rect)

    # Create the transparent hole using a PathPatch with a Path that outlines both the outer rectangle and the inner rectangle (hole)
    from matplotlib.path import Path
    path_data = [
        (Path.MOVETO, (outer_x, outer_y)),
        (Path.LINETO, (outer_x + outer_width, outer_y)),
        (Path.LINETO, (outer_x + outer_width, outer_y + outer_height)),
        (Path.LINETO, (outer_x, outer_y + outer_height)),
        (Path.CLOSEPOLY, (outer_x, outer_y)),
        (Path.MOVETO, (slit["position"][0] - slit["width"]/2, slit["position"][1] - slit["height"]/2)),
        (Path.LINETO, (slit["position"][0] + slit["width"]/2, slit["position"][1] - slit["height"]/2)),
        (Path.LINETO, (slit["position"][0] + slit["width"]/2, slit["position"][1] + slit["height"]/2)),
        (Path.LINETO, (slit["position"][0] - slit["width"]/2, slit["position"][1] + slit["height"]/2)),
        (Path.CLOSEPOLY, (slit["position"][0] - slit["width"]/2, slit["position"][1] - slit["height"]/2)),
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='green', lw=2, edgecolor='r', alpha=0.5)
    ax.add_patch(patch)

    # Adjust the axes limits
    ax.set_xlim(outer_x - 1, outer_x + outer_width + 1)
    ax.set_ylim(outer_y - 1, outer_y + outer_height + 1)

# Example usage
fig, ax = plt.subplots()
slit = {
    "position": (0, 0),  # Center position of the slit
    "width": 1,
    "height": 2
}
add_slits(ax, slit)
plt.show()