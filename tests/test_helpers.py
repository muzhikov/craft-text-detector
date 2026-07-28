import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from craft_text_detector import (
    export_detected_regions,
    export_extra_results,
    get_prediction,
    load_craftnet_model,
    load_refinenet_model,
    read_image,
)


class TestCraftTextDetectorHelpers(unittest.TestCase):
    image_path = "figures/idcard.png"

    def test_load_craftnet_model(self):
        craft_net = load_craftnet_model(cuda=False)
        self.assertTrue(craft_net)

        with TemporaryDirectory() as dir_name:
            weight_path = Path(dir_name, "weights.pth")
            self.assertFalse(weight_path.is_file())
            load_craftnet_model(cuda=False, weight_path=weight_path)
            self.assertTrue(weight_path.is_file())

    def test_load_refinenet_model(self):
        refine_net = load_refinenet_model(cuda=False)
        self.assertTrue(refine_net)

        with TemporaryDirectory() as dir_name:
            weight_path = Path(dir_name, "weights.pth")
            self.assertFalse(weight_path.is_file())
            load_refinenet_model(cuda=False, weight_path=weight_path)
            self.assertTrue(weight_path.is_file())

    def test_read_image(self):
        image = read_image(self.image_path)
        self.assertTrue(image.shape, (500, 786, 3))


if __name__ == "__main__":
    unittest.main()
