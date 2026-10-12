import base64
import io

from PIL import Image
from sentence_transformers import SentenceTransformer


class ClipEmbedder:
    """CLIP image encoder. Ingestion and search must use the same model."""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        vectors = self._model.encode(images, batch_size=32, normalize_embeddings=True)
        return vectors.tolist()


def decode_data_url(url: str) -> Image.Image:
    """'data:image/png;base64,...' -> RGB PIL image."""
    _, b64 = url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
