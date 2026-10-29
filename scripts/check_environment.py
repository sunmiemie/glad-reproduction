"""Validate the selected CPU/CUDA backend and optional real GLAD models.

Run from any directory. No downloads, installation, or automatic CPU fallback.
CUDA is the default so an accidentally installed CPU build fails explicitly.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='cuda:0 (default) or explicit cpu')
    parser.add_argument('--model-smoke-test', action='store_true',
                        help='Also load and execute the three published GLAD models')
    args = parser.parse_args(argv)
    report = {
        'status': 'error',
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'requested_device': args.device,
        'checks': {},
    }
    stage = 'imports'
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from torchvision.ops import nms

        report.update(torch=torch.__version__, torchvision=torchvision.__version__,
                      opencv=cv2.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()))
        stage = 'device'
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Supported checks are --device cpu or --device cuda:0')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and install matching '
                                   'CUDA builds of torch and torchvision in this environment. '
                                   'This check will not fall back to CPU.')
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError('Requested CUDA device index is not present')
            device = torch.device('cuda', index)
            properties = torch.cuda.get_device_properties(device)
            report.update(selected_device=str(device), device_name=properties.name,
                          device_memory_bytes=properties.total_memory,
                          compute_capability='%d.%d' % (properties.major, properties.minor))
            torch.cuda.set_device(device)
        else:
            report.update(selected_device='cpu', device_name=platform.processor() or 'CPU',
                          device_memory_bytes=None, compute_capability=None)
        torch.set_num_threads(4)
        cv2.setNumThreads(4)

        def synchronize():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        stage = 'matrix_multiplication'
        with torch.inference_mode():
            a = torch.ones((32, 32), dtype=torch.float32, device=device)
            product = a @ a
            synchronize()
            if product.device != device or not torch.allclose(product, torch.full_like(product, 32)):
                raise RuntimeError('Matrix multiplication produced an unexpected result or device')
            report['checks'][stage] = {'passed': True, 'device': str(product.device)}

            stage = 'convolution'
            conv = torch.nn.Conv2d(3, 4, 3).to(device).eval()
            output = conv(torch.randn((1, 3, 16, 16), device=device))
            synchronize()
            if output.device != device or tuple(output.shape) != (1, 4, 14, 14) or not torch.isfinite(output).all().item():
                raise RuntimeError('Convolution produced an unexpected result or device')
            report['checks'][stage] = {'passed': True, 'device': str(output.device)}

            stage = 'torchvision_nms'
            boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],
                                  [30., 30., 40., 40.]], device=device)
            scores = torch.tensor([.9, .8, .7], device=device)
            selected = nms(boxes, scores, .5)
            synchronize()
            if selected.device != device or selected.cpu().tolist() != [0, 2]:
                raise RuntimeError('Torchvision NMS produced an unexpected result or device')
            report['checks'][stage] = {'passed': True, 'device': str(selected.device)}

        if args.model_smoke_test:
            stage = 'load_glad_models'
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from glad.appearance import AppearanceDetector
            detector = AppearanceDetector(
                ROOT / 'weights' / 'yolov5s_GLAD.pt',
                ROOT / 'weights' / 'yolov5s_GLAD-crop.pt',
                ROOT / 'weights' / 'Net_best.pth', ROOT / 'external' / 'yolov5',
                device=str(device), imgsz=640, threads=4)
            for name, model in [('global', detector.global_model),
                                ('local', detector.local_model), ('classifier', detector.classifier)]:
                devices = {str(parameter.device) for parameter in model.parameters()}
                if devices != {str(device)}:
                    raise RuntimeError('%s parameters are on %s, expected %s' % (name, devices, device))
            report['checks'][stage] = {'passed': True, 'device': str(device),
                                      'metadata': detector.metadata}

            stage = 'glad_model_forward'
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            started = perf_counter()
            global_result = detector.detect(image, local=False)
            local_result = detector.detect(image, local=True)
            probability = detector.classify(image[:32, :32])
            synchronize()
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError('Candidate classifier produced an invalid probability')
            report['checks'][stage] = {
                'passed': True, 'global_output_count': len(global_result),
                'local_output_count': len(local_result), 'classifier_probability': probability,
                'smoke_test_seconds': perf_counter() - started,
                'note': 'Blank-image execution check only; not accuracy or warmed-up FPS.',
            }
        report['status'] = 'ok'
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report.update(failed_stage=stage, error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            report['hint'] = ('Use this project virtual environment and install its requirements. '
                              'Install torch 2.8.0 and torchvision 0.23.0 from the same CUDA 12.6 '
                              'wheel index before installing the remaining requirements.')
        elif stage == 'imports':
            report['hint'] = ('A runtime package could not initialize. Check the error above and '
                              'install matching torch/torchvision CUDA builds in the project '
                              'virtual environment, then install requirements.txt.')
        elif stage == 'torchvision_nms':
            report['hint'] = ('Install torch and torchvision from the same CUDA wheel index; '
                              'a CPU-only torchvision package cannot execute CUDA NMS.')
        elif stage in ('load_glad_models', 'glad_model_forward'):
            report['hint'] = ('Check weights/ and external/yolov5/. Run scripts/bootstrap.py '
                              '--verify-only to validate the bundled assets.')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
