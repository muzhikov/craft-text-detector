import PIL.Image
import unittest

from craft_text_detector import read_image
from pathlib import Path


class TestCraftTextDetectorHelpers(unittest.TestCase):
    image_path = "figures/idcard.png"

    def test_read_image_accepts_string(self):
        image = read_image(self.image_path)
        self.assertEqual(image.shape, (500, 786, 3))

    def test_read_image_accepts_path(self):
        image = read_image(Path(self.image_path))
        self.assertTrue(image.shape, (500, 786, 3))

    def test_read_image_accepts_pillow_image(self):
        image = read_image(PIL.Image.open(self.image_path))
        self.assertTrue(image.shape, (500, 786, 3))


if __name__ == "__main__":
    unittest.main()
