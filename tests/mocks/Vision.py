"""Stand-in for Apple's Vision framework (PyObjC surface used by stampede), so the pipeline runs
on Linux. Text: RapidOCR, or fixed answers from env vars. Faces: OpenCV's DNN face detector at four
rotations, reported once with a roll like Vision does. It is much noisier than real Vision.

Env vars:
  MOCK_VISION_LINE  text Vision "reads" on every stamp-line crop
  MOCK_VISION_FULL  text Vision "reads" in the full-photo pass (placed bottom right, like a stamp)
  MOCK_CRASH        kill the worker process while it analyses a file whose path contains this"""
import io, math, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
VNRequestTextRecognitionLevelAccurate = 0
CALLS = {"text": 0, "face": 0}

class _P:  # CGPoint/CGSize/CGRect-ish
    def __init__(self, **k): self.__dict__.update(k)

class _Alloc:
    @classmethod
    def alloc(cls): return cls()
    def init(self): return self

class _Cand:
    def __init__(self, s, c): self._s, self._c = s, c
    def string(self): return self._s
    def confidence(self): return self._c

class _TextObs:
    def __init__(self, s, c, bb): self._cand, self._bb = _Cand(s, c), bb
    def topCandidates_(self, n): return [self._cand][:n]
    def boundingBox(self): return self._bb

class _FaceObs:
    def __init__(self, c, roll): self._c, self._r = c, roll
    def confidence(self): return self._c
    def roll(self): return self._r

class VNRecognizeTextRequest(_Alloc):
    def init(self):
        self.level, self.langcorr, self.langs, self.minh, self._res = None, True, None, None, []
        return self
    def setRecognitionLevel_(self, v): self.level = v
    def setUsesLanguageCorrection_(self, v): self.langcorr = v
    def setRecognitionLanguages_(self, v): self.langs = list(v)
    def setMinimumTextHeight_(self, v): self.minh = v
    def results(self): return self._res
    def _run(self, img):
        CALLS["text"] += 1
        assert self.langcorr is False and self.level == 0, "stampede must disable language correction"
        a = np.asarray(img)
        H, W = a.shape[:2]
        if os.environ.get("MOCK_VISION_LINE") and H < 200:  # simulate Vision reading the stamp line correctly
            bb = _P(origin=_P(x=0.1, y=0.1), size=_P(width=0.8, height=0.8))
            self._res.append(_TextObs(os.environ["MOCK_VISION_LINE"], 0.9, bb)); return
        if os.environ.get("MOCK_VISION_FULL") and H >= 200:  # simulate the full-photo band read
            bb = _P(origin=_P(x=0.62, y=0.2), size=_P(width=0.3, height=0.15))  # bottom right, like a stamp
            self._res.append(_TextObs(os.environ["MOCK_VISION_FULL"], 0.3, bb)); return
        global _ENG
        if "_ENG" not in globals():
            from rapidocr_onnxruntime import RapidOCR
            _ENG = RapidOCR()
        res, _ = _ENG(a)
        if res is None and H < 200:  # short line image: recognise as a whole
            r = _ENG.text_recognizer([a])[0]
            res = [[[[0, 0], [W, 0], [W, H], [0, H]], r[0][0], r[0][1]]] if r and r[0][0] else None
        for box, txt, sc in res or []:
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            bb = _P(origin=_P(x=min(xs) / W, y=1 - max(ys) / H), size=_P(width=(max(xs) - min(xs)) / W, height=(max(ys) - min(ys)) / H))
            self._res.append(_TextObs(txt, float(sc), bb))

class VNDetectFaceRectanglesRequest(_Alloc):
    def init(self): self._res, self.rev = [], None; return self
    def setRevision_(self, v): self.rev = v
    def results(self): return self._res
    def _run(self, img):
        CALLS["face"] += 1
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
        import stampede as sd
        a = np.ascontiguousarray(np.asarray(img))
        # like the real detector: each face reported once, with the roll it has in this image
        scores = {k: sd.faces_dnn(a, k) for k in (0, 90, 180, 270)}
        k = max(scores, key=scores.get)
        if scores[k] >= 0.6:  # upright after rotating CW by k => rolled by -k in the input
            self._res.append(_FaceObs(min(scores[k], 1.0), math.radians(-k if k <= 180 else 360 - k)))

class VNImageRequestHandler(_Alloc):
    def initWithData_options_(self, data, opts):
        self.img = Image.open(io.BytesIO(bytes(data))).convert("RGB"); return self
    def performRequests_error_(self, reqs, err):
        import sys as _s
        bad = os.environ.get("MOCK_CRASH")
        if bad:  # test hook: kill the worker process while it analyses a file whose name contains MOCK_CRASH
            f = _s._getframe()
            while f:
                if f.f_code.co_name == "analyse" and bad in str(f.f_locals.get("p", "")):
                    os._exit(139)
                f = f.f_back
        for r in reqs: r._run(self.img)
        return (True, None)
