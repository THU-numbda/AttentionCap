# AttentionCap: Transformer Model for 2D Interconnect Capacitance Extraction
## Full experiment scripts will be released soon.

## 1\. Prepare Dataset

The first step is to process the raw JSONL data into a PyTorch formatj. The `prepare.py` handles this by reading the data, grouping it by length, splitting it into training, validation, and test sets, and saving the results as compressed tensor files.

### Input Data Format

`prepare.py` expects a `.jsonl` file where each line is a JSON object containing at least the following keys:

  * `"size"`: An integer representing the number of conductors ($L$).
  * `"conductors"`: A list of conductor properties `[x,y,w,h]`, which will become the model's features ($L, 4$).
  * `"capacitances"`: A flattened list of capacitance values, which will be the model's target outputs ($L^2$).

**Example `data.jsonl` line:**

```json
{"size": 2, "conductors": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], "capacitances": [1.0, 2.0, 3.0, 4.0]}
```

### Usage

```bash
python prepare.py path/to/your/data.jsonl --plot # Recommended
python prepare.py path/to/your/data.jsonl --output_dir ./processed_dataset --train_ratio 0.7 --val_ratio 0.15 --shuffle --plot
```

### Script Arguments

  * `input_file`: (Required) Path to the input `.jsonl` file.
  * `--output_dir`: Directory to save the output files. If not provided, it defaults to a new directory named after the input file (e.g., `data.jsonl` -\> `data/`).
  * `--train_ratio`: (Default: `0.8`) The proportion of data to use for the training set.
  * `--val_ratio`: (Default: `0.1`) The proportion of data to use for the validation set. The remaining data becomes the test set.
  * `--shuffle`: A flag to enable shuffling of the data before splitting. (Not totally necessary because shuffling is handled in training loaders)
  * `--plot`: A flag to generate and save a bar chart showing the data distribution across different sizes for each split.

### Output Structure

After running, the output directory will contain the following files:

  * `train_data.pt`: A PyTorch file containing the training data.
  * `val_data.pt`: A PyTorch file containing the validation data.
  * `test_data.pt`: A PyTorch file containing the testing data.
  * `dataset_distribution.png`: (Optional, if `--plot` is used) An image visualizing the distribution over sample length.

Each `.pt` file is a dictionary where keys are the integer sizes ($L$) from the dataset. The values are another dictionary containing two tensors:

  * `'features'`: A `torch.Tensor` of shape ($N, L, 4$), where $N$ is the number of samples in that split for that size.
  * `'outputs'`: A `torch.Tensor` of shape ($N, L, S$).



## 2. Create Dataloader

The `dataset.py` script takes the processed `.pt` files from the previous step and creates efficient PyTorch `DataLoader` objects, ready for model training and evaluation.

### Overview

The script is designed for efficiency when dealing with variable-length sequences.

* **Input**: It loads a data split file (e.g., `train_data.pt`) created by `prepare.py`.
* **Augmentation**: It doubles the size of the dataset by applying a simple **horizontal flip** augmentation to the conductor coordinates.
* **Bucket Sampling**: To minimize computational waste, it uses a custom `BucketBatchSampler`. This component groups samples of **similar lengths** (specified via `bucket_width`) into the same batches.
* **Dynamic Padding**: Within each batch, sequences are padded with `0.0` to the maximum length present in that batch.
* **Visualization**: When loading the `test` set, the script automatically generates and saves a visualization of the first sample from each size group (e.g., `sample0_16.png`).

### Usage

* For visualization, import `visualize_sample(rects: torch.Tensor, colors_mat: torch.Tensor, fname: str)` to plot any sample.

* For creating dataloaders,
```python
from dataset import get_dataloader
dataloaders = {
    k : get_dataloader(
        split=k,
        data_dir=data_dir,
        batch_size=batch_size,
        bucket_width=bucket_width,
        num_workers=4, # adjust based on your system
        pin_memory=(device_type == 'cuda'), # pin memory for faster transfers to GPU
    ) for k in ['train', 'val', 'test']
}
for split, loader in dataloaders.items():
    for X, Y, mask in loader:
        # Your training/evaluation code here
```

### Output Batch

When you iterate over a `DataLoader` instance created by this script, each batch is a tuple containing three tensors, dynamically padded to the maximum length within that specific batch:

1.  **Padded Features** (`torch.Tensor`): The input conductor data for the model.
    * **Shape**: ($B, L_{max}, 4$), where $B$ is the batch size and $L_{max}$ is the maximum sequence length in the current batch.
2.  **Padded Targets** (`torch.Tensor`): The corresponding ground-truth capacitance matrices.
    * **Shape**: ($B, L_{max}, L_{max}$).
3.  **Attention Mask** (`torch.Tensor`): A binary mask used to inform the model which elements are real data versus padding.
    * **Shape**: ($B, L_{max}$), with a value of `1` for real data points and `0` for padding.
