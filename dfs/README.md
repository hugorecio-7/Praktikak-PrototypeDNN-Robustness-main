# DFS / DFL experiments

This folder contains isolated DFS/DFL prototype-distance experiments. It does
not modify the original training or evaluation pipeline.

Estimate a useful DFL distance range with:

```bash
python dfs/estimate_dmax.py --split test --max-batches 1
```
