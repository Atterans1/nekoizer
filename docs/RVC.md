# RVC voices in Nekoizer

An RVC voice is a slot in the same voice bank as a reference clip. Slot 0 can hold a wav, mp3, flac or ogg clip (Seed-VC engine) and slot 1 an RVC model (RVC engine). Selecting a slot, in the UI or with Ctrl+Alt+1 to 4, switches the engine. The input chain, VAD, balancer, SOLA and output chain are shared.

## Importing

- A `.pth` (and optional `.index`), or a `.zip` with them, dropped on a slot or chosen from disk.
- An https link to a direct file or zip (Hugging Face resolve links work). Web pages and Google Drive share links are refused: download the file first and import it from disk.
- voice-models.com lists RVC models. The "Browse voice-models.com" button opens the site in your browser. Nekoizer never scrapes it.
- Only use voices you have the right to use.

## Safety

RVC `.pth` files are Python pickles, which can run code. Nekoizer loads them only with `torch.load(weights_only=True)` and refuses a model that does not load that way. Zip imports extract only `.pth` and `.index` entries by name, reject path traversal, and cap sizes (model 400 MB, zip 600 MB). Links must be https to a public address (checked again at connect time against DNS rebinding), with a size cap.

## First use

HuBERT (190 MB) and RMVPE (181 MB) are downloaded the first time an RVC voice is used, into `%LOCALAPPDATA%\Nekoizer\rvc_weights`, with the same resumable SHA256-checked downloader as the Seed-VC models. They are not part of the installer. Imported voices live in `%LOCALAPPDATA%\Nekoizer\rvc_models`.

## Measured on the RTX 3050 4 GB laptop GPU (shared with other processes)

One real v2 40 kHz voice, 220 ms blocks, RMVPE pitch, no index:

| Run | median per block | p95 per block | p95 / block |
|---|---|---|---|
| 1 (GPU clocks started idle) | 188 ms | 253 ms | 1.15 |
| 2 | 165 ms | about 205 ms (0.93 x 220, ratio measured) | 0.93 |

The rule for calling an engine real time is p95 / block <= 0.9, so RVC does **not** meet it on this GPU in either run. Use a larger block (the block size is a control) or accept occasional dropouts. A converted clip scored UTMOS 4.1 and DNSMOS 3.38 (predictors, not listening). Retrieval over the `.index` file is not used.
