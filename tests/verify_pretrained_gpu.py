"""Verify genuine initialization and GPU feasibility using synthetic images only.

Run explicitly: python tests/verify_pretrained_gpu.py --weights-dir outputs/pretrained_weights
The report and disposable state dictionaries are written to a fresh scratch folder.
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch import nn
from cnn_assignment.model_specs import PRETRAINED_MODELS, PRETRAINED_SPECS
from cnn_assignment.models import build_model, load_saved_state, model_cost
from cnn_assignment.utils import atomic_torch_save, seed_everything, sha256


def verify(name, weights, output):
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(name, pretrained=True, pretrained_weights_file=weights, progress=False)
    official = torch.load(weights, map_location='cpu', weights_only=True)
    prefix = 'classifier.' if name == 'mobilenet_v2' else 'fc.'
    keys = [key for key in model.state_dict() if not key.startswith(prefix) and key in official]
    assert keys and all(torch.equal(model.state_dict()[key], official[key]) for key in keys)
    missing_legacy = [key for key in model.state_dict() if not key.startswith(prefix) and key not in official]
    assert all(key.endswith('num_batches_tracked') and model.state_dict()[key].item() == 0 for key in missing_legacy)
    head = model.classifier[1] if name == 'mobilenet_v2' else model.fc
    head_key = prefix + ('1.weight' if name == 'mobilenet_v2' else 'weight')
    assert head.out_features == 10 and not torch.equal(head.weight, official[head_key][:10])
    del official
    model.to(device)
    cost = model_cost(model)
    assert cost['trainable_parameters'] == cost['total_parameters']
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    batches = []
    for batch_size in (1, 20, 64):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free_before, total = torch.cuda.mem_get_info()
        images = torch.randn(batch_size, 3, 64, 64, device=device)
        labels = torch.arange(batch_size, device=device) % 10
        with torch.inference_mode():
            model.eval()
            logits = model(images)
            assert logits.shape == (batch_size, 10) and torch.isfinite(logits).all().item()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = next(model.parameters()).detach().clone()
        train_logits = model(images)
        assert train_logits.shape == (batch_size, 10) and torch.isfinite(train_logits).all().item()
        loss = nn.CrossEntropyLoss()(train_logits, labels)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        optimizer.step()
        assert not torch.equal(before, next(model.parameters()))
        row = {'batch_size': batch_size, 'output_shape': [batch_size, 10],
               'finite_eval_and_train_logits': True, 'finite_loss': float(loss.detach()),
               'all_gradients_finite': True, 'backbone_updated': True}
        if device.type == 'cuda':
            torch.cuda.synchronize()
            row.update(free_before_bytes=free_before, total_gpu_bytes=total,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        batches.append(row)
        del images, labels, logits, train_logits, loss, before
    model.eval()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = output / 'best_weights.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    atomic_torch_save(state, checkpoint)
    cost = model_cost(model, checkpoint)
    restored = build_model(name, pretrained=False).to(device).eval()
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    load_saved_state(restored, loaded)
    assert all(torch.equal(loaded[key], restored.state_dict()[key].cpu()) for key in loaded)
    buffers = [key for key in loaded if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    assert buffers and all(torch.equal(loaded[key], restored.state_dict()[key].cpu()) for key in buffers)
    probe = torch.randn(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        torch.testing.assert_close(model(probe), restored(probe), rtol=0, atol=0)
    result = {'model': name, 'device': str(device), 'source_sha256': sha256(weights),
              'pretrained_weights': PRETRAINED_SPECS[name]['pretrained_weights'],
              'source': PRETRAINED_SPECS[name]['pretrained_source'],
              'backbone_tensors_matched_before_update': len(keys),
              'legacy_bn_counters_initialized_to_zero': len(missing_legacy),
              'classifier_replaced_after_loading': True, 'cost': cost, 'synthetic_batches': batches,
              'strict_saved_restoration': True, 'batchnorm_buffers_restored': len(buffers),
              'restored_outputs_bitwise_equal': True,
              'checkpoint_scope': 'Synthetic verification state, not a EuroSAT-trained checkpoint'}
    del model, restored, optimizer, state, loaded, probe
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    output = Path(args.output_dir) if args.output_dir else ROOT / 'tmp' / (
        'pretrained_gpu_check_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'data_scope': 'Synthetic tensors only; no dataset loader, training run or test evaluation',
              'models': []}
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)
        report['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    for name in PRETRAINED_MODELS:
        filename = PRETRAINED_SPECS[name]['pretrained_source'].rsplit('/', 1)[1]
        result = verify(name, Path(args.weights_dir) / filename, output / name)
        report['models'].append(result)
        print(json.dumps(result), flush=True)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Verification report:', output / 'verification.json', flush=True)


if __name__ == '__main__':
    main()
