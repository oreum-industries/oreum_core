# eda.utils.py
# copyright 2022 Oreum Industries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def display_image_file(fqn):
    """Hacky way to display pre-created image file in a Notebook
    such that nbconvert can see it and render to PDF
    Force to a max width 16 inches, so you can see live it in Notebook
    and it also renders full width in PDF

    NOTE:
    Alternatives are bad
        1. This one is entirely missed by nbconvert at render to PDF
        # <img src="img.jpg" style="float:center; width:900px" />

        2. This one causes following markdown to render monospace in PDF
        # from IPython.display import Image
        # Image("./assets/img/oreum_eloss_blueprint3.jpg", retina=True)
    """
    img = mpimg.imread(fqn)
    f, axs = plt.subplots(1, 1, figsize=(16, 10))
    im = axs.imshow(img)
    ax = plt.gca()
    ax.grid(False)
    ax.set_frame_on(False)
    plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
