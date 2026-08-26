# FeatherTalk Pretrained Weights

This source release includes the lightweight preprocessing models required for face detection and
landmark detection:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
```

Wenet feature extraction still needs the larger encoder file below. It is about 110 MB, so it is
not tracked in git. Download it only if you want to use `--asr wenet`:

```text
data_utils/encoder.onnx
```

Download link:

[encoder.onnx](https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link)

HuBERT extraction uses Hugging Face Transformers and the model id:

```text
facebook/hubert-large-ls960-ft
```

FeatherHuBERT checkpoints are regular PyTorch `.pth` files. They are not included in the source
tree and are ignored by git by default. You can use the checkpoint from the demo training asset
pack or place your own externally trained checkpoint wherever you prefer.

To convert an explicitly supplied FeatherHuBERT checkpoint into the standard auditable package,
provide a separately reviewed `LICENSES.json`, an RFC 3339 creation time, and a new destination:

```powershell
cargo run -p feathertalk-model-package -- feather-hubert `
  --source <path-to-feather_hubert.pth> `
  --licenses <path-to-reviewed-LICENSES.json> `
  --destination <new-feather-hubert-package-directory> `
  --created-at 2026-08-27T00:00:00Z `
  --minimum-app-version 0.1.0
```

The converter snapshots and audits the source before importing it, writes only the standard
three-file package, and never accepts a license-free shortcut. A demo checkpoint may be used as
the source, but a local synthetic test license is not permission to redistribute model weights.

## VGG19 perceptual weights

The training perceptual extractor uses the torchvision VGG19 `features.14` convolution output
(`VGG19_Weights.IMAGENET1K_V1`). The fixed source URL is:

```text
https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
```

The official file used for numerical parity has SHA-256
`dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44`.

The source checkpoint is not bundled in this repository. Build a runtime package only from an
explicitly supplied source file and an independently reviewed license bundle:

```powershell
cargo run -p feathertalk-vgg19-package -- --source <path-to-vgg19-dcbb9e9d.pth> --licenses <path-to-reviewed-LICENSES.json> --destination <new-vgg19-package-directory>
```

The runtime directory is intentionally limited to these three regular files:

```text
manifest.json
model.safetensors
LICENSES.json
```

The loader verifies the manifest schema, source/model/license hashes, byte lengths, tensor names,
shapes, dtypes, and license entries before returning frozen parameters. It never downloads
weights, searches a cache, parses a `.pth` file, or falls back to random parameters. The input
contract remains BGR float32 values in `[0,1]` with no ImageNet normalization, matching the
existing training semantics.

`rust/tests/fixtures/vgg19/LICENSES.local-parity.json` is an honest local numerical-parity record
only; it is not approval to redistribute the pretrained weights. A release package must provide
an independently reviewed `LICENSES.json` and record the source, model, and license hashes in its
manifest.
