# CNNCap baseline

Cleaned, behavior-compatible copy of the original CNNCap 1D-ResNet34 baseline.
It directly reads the prepared ASAP7 and real65 splits under this repository's
`data/` directory.

```bash
source .venv/bin/activate
python baselines/cnncap/run_train.py
python baselines/cnncap/run_eval.py
```

Individual entries can also be run directly:

```bash
python baselines/cnncap/train.py data/asap7_50K \
  --goal=total --window_width=2.736 --batch_size=448 \
  --device=cuda:0 --out_dir=training_output/cnncap_baseline/asap7-total

python baselines/cnncap/eval.py data/asap7_50K \
  training_output/cnncap_baseline/asap7-total/best.model.pth \
  --split=test --goal=total --window_width=2.736 \
  --device=cuda:0 --logfile=training_output/cnncap_baseline/asap7-total/test.log
```
