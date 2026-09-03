import matplotlib as mpl


def get_rgba_color(num):
    """
    Returns a unique RGBA color tuple for a given number less than 10.
    """
    assert num >= 0 and num < 10, "Number should be between 0 and 9"
    colors = mpl.colormaps["tab10"]
    rgba = colors(num)  # get the RGBA color tuple for the given number
    rgba = tuple([int(255 * x) for x in rgba])
    return rgba
