import unittest

from craft_text_detector import read_image


class TestCraftTextDetectorHelpers(unittest.TestCase):
    image_path = "figures/idcard.png"

    def test_read_image(self):
        image = read_image(self.image_path)
        self.assertTrue(image.shape, (500, 786, 3))


if __name__ == "__main__":
    unittest.main()
