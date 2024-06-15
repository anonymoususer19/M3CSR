# A Multi-modal Modeling Framework for Cold-start Short-video Recommendation
## Dependencies
- python >= 3.8.5
- torch >= 2.0.1
- cudatoolkit >= 11.4.1
- scipy >= 1.10.1
- numpy >= 1.22.4
- tqdm >= 4.66.1

## Datasets
All datasets are publicly available at https://github.com/HKUDS/MMSSL/tree/main

Supported datasets: ```tiktok```

## Usage
Start training and inference as:

```
cd M3CSR
python main.py --dataset {DATASET}
```

## Notes
- Please modify the paths in ```utility/logging.py``` and ```utility/parser.py``` before running the code, specifically line 5 in ```logging.py``` and line 50 in ```parser.py```
