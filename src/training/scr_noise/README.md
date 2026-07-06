# scr_noise BiGRU baselines

This folder is separate from the existing clean precomputed-embedding BiGRU
pipeline in `src/training/train_bigru.py`.

The flow here is:

```text
raw audio sequence
-> DataLoader
-> optional random AWGN during training only
-> Wav2Vec2 extracts embeddings on the fly
-> BiGRU
```
Run dynamic-AWGN baselines:

```powershell
python src/training/scr_noise/train_bigru.py --mode flat8 --baseline dynamic-awgn
python src/training/scr_noise/train_bigru.py --mode stage1 --baseline dynamic-awgn
python src/training/scr_noise/train_bigru.py --mode stage2 --baseline dynamic-awgn
```

Run all six:

```powershell
python src/training/scr_noise/train_bigru.py --run-all-baselines
```

Useful options:

```powershell
python src/training/scr_noise/train_bigru.py --mode stage1 --baseline dynamic-awgn --awgn-prob 0.5 --awgn-snr-choices 10,15,20
```

Dynamic AWGN is applied only to training batches. Validation/test batches stay
clean.
