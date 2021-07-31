import cog
import cv2
import tempfile
from pathlib import Path
from test import Photo2Cartoon
from test_onnx import Photo2Cartoon as Onnx_Photo2Cartoon


class Photo2CartoonModel(cog.Model):
    def setup(self):
        pass

    @cog.input("photo", type=Path, help="portrait photo (size < 1M)")
    def predict(self, photo):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        img = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB)
        c2p = Photo2Cartoon()
        cartoon = c2p.inference(img)
        if cartoon is not None:
            cv2.imwrite(str(out_path), cartoon)
        return out_path
