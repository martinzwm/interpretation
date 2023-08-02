import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import cv2
from typing import Optional


def embedding_overlay(
      image: np.ndarray,
      embedding: np.ndarray,
      downsize_to: Optional[int] = None,
      overlay_strength: float = 0.4,
):
    """Overlays the similarity matrix on the original image.

    Args:
      image: np.ndarray of shape (H, W) for the original image.
      embedding: np.ndarray of shape (1, C, N, M) for cosine similarity.
      overlay_strength: float to control the opacity of the overlay.

    Returns:
      overlay: np.ndarray of shape (H, W) for the overlay image.
    """
    # Sanity checks
    try:
      image = np.array(image)
    except Exception as exc:
      raise ValueError(
          "Unable to convert {} type to numpy.ndarray type.".format(type(image))
      ) from exc

    if len(image.shape) != 2:
      raise ValueError("Expect image to be of shape (H, W) (i.e., grayscale).")

    # Normalize the image to [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Global average pooling on the embedding
    embedding = np.mean(embedding, axis=(0, 1))

    # Downsize embedding if needed
    if downsize_to is not None:
        embedding = scipy.ndimage.zoom(embedding, (downsize_to / embedding.shape[0], downsize_to / embedding.shape[1]))

    h, w = image.shape
    n, m = embedding.shape

    # Resize the similarity grid to the size of the original image.
    similarity_resized = scipy.ndimage.zoom(embedding, (h / n, w / m))

    # Create a color overlay using a colormap
    jet_cmap = matplotlib.cm.get_cmap("jet")
    overlay = jet_cmap(similarity_resized)

    # Set the transparency of the overlay based on the similarity matrix
    overlay[..., 3] = similarity_resized * overlay_strength

    # Create an RGBA image from the grayscale image
    image_rgba = np.repeat(image[..., np.newaxis], 4, axis=2)
    image_rgba[..., 3] = 1  # fully opaque

    # Apply the overlay over the grayscale image
    image_with_overlay = overlay * overlay_strength + image_rgba * (
        1 - overlay_strength
    )

    # Plot the original image and the overlay
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")

    ax[1].imshow(image_with_overlay)
    ax[1].set_title("Image with similarity overlay")

    # add colormap to the plot
    sm = plt.cm.ScalarMappable(cmap=jet_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)

    return fig


def test_case():
    image = cv2.imread('x_ray.jpeg', cv2.IMREAD_GRAYSCALE)
    embedding = np.random.rand(1, 256, 320, 320)
    fig = embedding_overlay(image, embedding, downsize_to=10)
    fig.savefig('results.png')


if __name__ == "__main__":
   test_case()