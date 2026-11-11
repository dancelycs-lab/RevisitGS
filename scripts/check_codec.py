"""Small deterministic regression checks for correspondence order and edge cases."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from revisitgs import GaussianMap,CodecConfig,encode_revision,decode_revision,load_ply,save_ply
from revisitgs.codec import _allocate
def scene(x):
    n=len(x);return GaussianMap(np.asarray(x,np.float32),np.tile(np.array([[1,0,0,0]],np.float32),(n,1)),np.zeros((n,3),np.float32),np.zeros(n,np.float32),np.zeros((n,16,3),np.float32))
a=scene(np.arange(6)[:,None]*np.array([[10,0,0]]));b=scene(a.means+np.array([[.6,0,0],[.5,0,0],[.4,0,0],[.3,0,0],[.2,0,0],[.1,0,0]]))
keep,match,_,_=_allocate(a,b,np.ones(6,bool),np.ones(6,bool),np.arange(6),.5)
assert np.array_equal(np.flatnonzero(keep),match)
# One retained predictor must decode without fitting a degenerate PCA.
a=scene([[0,0,0],[10,0,0]]);b=scene([[.01,0,0],[10.02,0,0]])
payload,birth,meta=encode_revision(a,b);decoded=decode_revision(payload,a)
assert len(decoded)+len(birth)==len(b)
assert np.min(np.linalg.norm(b.means-decoded.means[0],axis=1))<.001
# No support: complete intra input is retained and empty revision decodes safely.
b=scene([[100,0,0],[110,0,0]])
payload,birth,meta=encode_revision(a,b,CodecConfig(radius_fraction=.00001))
assert len(decode_revision(payload,a))==0 and len(birth)==len(b)
assert meta['recommended_mode']=='intra'
# Empty birth maps are valid when every target primitive is revised.
# Check disk I/O as well as in-memory decoding for that case.
from tempfile import TemporaryDirectory
with TemporaryDirectory(prefix="revisitgs_check_") as tmp:
    path = Path(tmp) / "map.ply"
    for count in (0, 1, 6):
        value = scene(np.arange(count)[:, None] * np.array([[10, 0, 0]]))
        save_ply(value, path)
        restored = load_ply(path)
        assert restored.sh.shape == value.sh.shape
        for field in ("means", "quats", "scales", "opacities", "sh"):
            assert np.array_equal(getattr(value, field), getattr(restored, field))
    source = scene([[0, 0, 0], [10, 0, 0]])
    target = scene([[.01, 0, 0], [10.02, 0, 0]])
    target.sh[:] = np.arange(target.sh.size, dtype=np.float32).reshape(target.sh.shape) * .0001
    payload, birth, _ = encode_revision(source, target, CodecConfig(transport_fraction=1.0))
    assert len(birth) == 0
    save_ply(birth, path)
    from revisitgs import concatenate
    restored = concatenate(decode_revision(payload, source), load_ply(path))
    assert len(restored) == len(target)
print('PASS: predictor order, singleton revision, zero-support fallback, PLY round trips and empty births')

# Selection rejects incomplete or incomparable measurements before writing a mode.
import json
import subprocess
with TemporaryDirectory(prefix="revisitgs_selection_") as tmp:
    root = Path(tmp)
    conditional = {"total_bytes": 100, "mean_psnr_db": 40., "target_gaussians": 10,
                   "cardinality_match": True, "views": [{"index": 0, "camera": "test"}]}
    intra = {**conditional, "total_bytes": 200}
    for changes, valid in (({}, True), ({"mean_psnr_db": float('nan')}, False),
                           ({"total_bytes": -1}, False), ({"cardinality_match": False}, False),
                           ({"views": [{"index": 1, "camera": "other"}]}, False)):
        (root / 'conditional.json').write_text(json.dumps({**conditional, **changes}))
        (root / 'intra.json').write_text(json.dumps(intra))
        mode = root / 'mode.bin'
        mode.unlink(missing_ok=True)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('select_mode.py')),
                                 '--conditional', str(root / 'conditional.json'),
                                 '--intra', str(root / 'intra.json'), '--target-gaussians', '10',
                                 '--mode-output', str(mode)], capture_output=True, text=True)
        assert (result.returncode == 0) == valid, result.stderr
        assert mode.exists() == valid
        if valid:
            assert mode.read_bytes() == b'\x01'
print('PASS: mode selection and rejection of invalid/incomparable records')
