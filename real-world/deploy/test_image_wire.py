import base64,unittest,zlib
import numpy as np
from image_wire import encode_image_bytes,decode_image_bytes

class ImageWireTests(unittest.TestCase):
    def test_exact_roundtrip_and_legacy(self):
        a=np.random.default_rng(4).integers(0,256,(48,64,3),dtype=np.uint8)[:,::-1]
        self.assertEqual(decode_image_bytes(dict(compression='zlib',data_b64=encode_image_bytes(a)),a.nbytes),a.tobytes())
        self.assertEqual(decode_image_bytes(dict(data_b64=base64.b64encode(a.tobytes()).decode()),a.nbytes),a.tobytes())
    def test_rejects_truncation_and_expansion(self):
        b=base64.b64encode(zlib.compress(b'x'*100)).decode()
        with self.assertRaises(ValueError):decode_image_bytes(dict(compression='zlib',data_b64=b),3)
        b=base64.b64encode(zlib.compress(b'x'*100)[:-2]).decode()
        with self.assertRaises(ValueError):decode_image_bytes(dict(compression='zlib',data_b64=b),100)

if __name__=='__main__':unittest.main()
