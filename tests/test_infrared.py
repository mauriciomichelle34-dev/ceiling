"""Spot exclusion must preserve visible neighbours and flow into saved masks."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infrared import detect_infrared,exclude_infrared,InfraredDetectionError
from local_app import Predictor


def scene(spots=((510,480),),red=(255,35,30),core=True):
    image=np.full((683,1024,3),150,np.uint8)
    for cx,cy in spots:
        cv2.circle(image,(cx,cy),25,red,cv2.FILLED)
        if core:
            cv2.circle(image,(cx,cy),16,(255,245,220),cv2.FILLED)
    return Image.fromarray(image)


BOUNDARY={'center_x_px':512.,'center_y_px':341.,'radius_px':320.}


class InfraredTests(unittest.TestCase):
    def test_automatic_localization_tracks_spot_movement(self):
        for position in [(450,420),(590,470),(512,510)]:
            source=scene((position,))
            ignored,info=detect_infrared(source,BOUNDARY)
            self.assertEqual(info['status'],'excluded')
            self.assertTrue(ignored[position[1],position[0]])
            spot=info['spots'][0]
            self.assertAlmostEqual(spot['center_x_px'],position[0],delta=3)
            self.assertAlmostEqual(spot['center_y_px'],position[1],delta=3)

    def test_white_glare_without_red_halo_is_not_erased(self):
        ignored,info=detect_infrared(scene(red=(225,225,225)),BOUNDARY)
        self.assertEqual(info['status'],'not_detected')
        self.assertFalse(ignored.any())

    def test_red_mark_without_bright_core_is_not_erased(self):
        ignored,info=detect_infrared(scene(core=False),BOUNDARY)
        self.assertEqual(info['status'],'not_detected')
        self.assertFalse(ignored.any())

    def test_blue_mark_with_white_center_is_not_erased(self):
        ignored,_=detect_infrared(scene(red=(20,50,255)),BOUNDARY)
        self.assertFalse(ignored.any())

    def test_outside_dish_spot_does_not_change_interior(self):
        ignored,_=detect_infrared(scene(((80,80),)),BOUNDARY)
        self.assertFalse(ignored.any())

    def test_two_equally_strong_spots_stop_instead_of_guessing(self):
        with self.assertRaises(InfraredDetectionError):
            detect_infrared(scene(((450,450),(590,450))),BOUNDARY)

    def test_exclusion_preserves_every_pixel_outside_local_mask(self):
        ignored,_=detect_infrared(scene(),BOUNDARY)
        original=np.ones(ignored.shape,np.uint8)
        filtered=exclude_infrared(original,ignored)
        self.assertTrue(original.all())
        self.assertTrue(np.array_equal(filtered[~ignored],original[~ignored]))
        self.assertFalse(filtered[ignored].any())
        # Surrounding visible hyphae are not removed by a rectangular blackout.
        self.assertEqual(filtered[480,545],1)
        self.assertEqual(filtered[445,510],1)
        self.assertEqual(filtered[480,475],1)

    def test_predictor_saves_filtered_masks_and_previews(self):
        source=scene(); raw=io.BytesIO(); source.save(raw,format='PNG')
        predictor=Predictor.__new__(Predictor)
        predictor.network=None; predictor.config={'threshold':.5}
        predictor.device='cpu'; predictor.model_name='fixture'
        with tempfile.TemporaryDirectory() as folder:
            predictor.result_root=Path(folder)
            suggestion={'center_x_px':512.,'center_y_px':341.,'dish_diameter_px':640.}
            with patch('local_app.detect_dish',return_value=(BOUNDARY,suggestion)),patch('local_app.predict_probability',return_value=np.ones((683,1024),np.float32)):
                meta=predictor.predict(raw.getvalue(),'fixture.png')
            saved=predictor.result_root/meta['id']
            with Image.open(saved/'mask.png') as im: mask=np.asarray(im)
            with Image.open(saved/'infrared_ignore.png') as im: ignored=np.asarray(im)>0
            self.assertEqual(meta['infrared']['status'],'excluded')
            self.assertGreater(meta['infrared']['removed_foreground_pixels'],0)
            self.assertFalse(mask[ignored].any())
            self.assertEqual(mask[480,545],1)
            with Image.open(saved/'region.png') as im: region=np.asarray(im)
            self.assertEqual(region[480,510,3],0)
            self.assertGreater(region[480,545,3],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
