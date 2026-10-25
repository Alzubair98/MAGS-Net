"""Check frozen inference and UI with synthetic input, without a dataset or training."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from magsnet_demo.inference import load_ensemble, predict
from magsnet_demo.preprocessing import prepare_image
from magsnet_demo.utils import DemoError
from magsnet_demo.visualization import make_downloads, render_outputs


def main():
    y, x = np.mgrid[:256, :256]
    inside = ((x - 128) / 85) ** 2 + ((y - 128) / 105) ** 2 < 1
    pixels = np.where(inside, 70 + (x + 2 * y) % 150, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    data = buffer.getvalue()
    prepared = prepare_image(data)
    print('Loading and checking the five CPU models...', flush=True)
    ensemble = load_ensemble('cpu')
    assert len(ensemble.models) == 5
    assert all(not m.training and not any(p.requires_grad for p in m.parameters()) for m in ensemble.models)
    assert all(sum(p.numel() for p in m.parameters()) == 688270 for m in ensemble.models)
    with patch.object(torch.optim.Optimizer, 'step', side_effect=AssertionError('Training forbidden')):
        result = predict(ensemble, prepared.tensor)
    print('CPU inference passed; checking exports...', flush=True)
    assert result['mask'].shape == (224, 224)
    assert np.isfinite(result['probabilities']).all() and np.isclose(result['probabilities'].sum(), 1)
    assert np.allclose(result['gates'].sum(axis=1), 1, atol=1e-5)
    visuals = render_outputs(prepared, result)
    downloads = make_downloads(prepared, result, visuals, 'cpu')
    assert set(downloads) == {'mask', 'overlay', 'analyzed_mask', 'summary'}
    summary = json.loads(downloads['summary'])
    assert summary['models'] == 5 and 'synthetic.png' not in downloads['summary'].decode()
    print('Checking Streamlit initial page...', flush=True)
    with patch('magsnet_demo.config.DEVICE', 'cpu'):
        app = AppTest.from_file(ROOT / 'magsnet_demo/app.py', default_timeout=60).run()
        print('Checking upload and Analyze action...', flush=True)
        assert not app.exception and app.title[0].value == 'MAGS-Net'
        if any(option == 'CPU' for option in app.selectbox[0].options):
            app.selectbox[0].set_value('CPU').run()
        app.file_uploader[0].set_value(('synthetic.png', data, 'image/png')).run()
        assert not app.exception and 'result' not in app.session_state
        next(b for b in app.button if b.label == 'Analyze MRI').click().run(timeout=60)
        print('Checking downloads, invalid input and session clear...', flush=True)
        assert not app.exception and not app.error
        assert len(app.download_button) == 3 and len(app.expander) == 2
        app.run()
        assert not app.exception and 'result' in app.session_state
        app.file_uploader[0].set_value(('invalid.png', b'invalid', 'image/png')).run()
        assert not app.exception and app.error and 'result' not in app.session_state
        next(b for b in app.button if b.label == 'Clear uploaded image & results').click().run()
        assert not app.exception and 'prepared' not in app.session_state
    try:
        prepare_image(b'invalid')
    except DemoError:
        pass
    else:
        raise AssertionError('Invalid image was accepted')
    print(json.dumps({'status': 'passed', 'frozen_models': 5, 'parameters_per_model': 688270,
                      'cpu_inference': True, 'preprocessing_and_downloads': True,
                      'streamlit_upload_analyze_clear': True, 'training_run': False,
                      'input': 'synthetic; not a scientific performance result'}, indent=2))


if __name__ == '__main__':
    main()
