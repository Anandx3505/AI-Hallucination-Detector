

### Usage
To start training and evaluation, use the following command:
```bash
python train.py [arguments]
```

### Arguments
```bash
--use-cuda: Enable GPU acceleration if available (default: True)
--path <path_to_data>: Set the path to the data folder (default: "../data/")
--output_dir <output_directory>: Specify the path to save the model weights (default: "../weights/")
--epochs <num_epochs>: Define the number of epochs to train the model (default: 1000)
--batch-size <batch_size>: Set the batch size used during training and evaluation (default: 256)
--lr <lr>: Set the training learning rate for the optimizer (default: 0.01)
--optimizer <optimizer>: Choose the optimizer to train the model between SGD and Adam (default: Adam)
--model <model>: Which model to train and evaluate between MLP, CE, and PCA
--pretrained <pretrained>: What model to use for computing embeddings, if a local pretrained model us used, then provide the path to the location of the model (default: bert-base-uncased)
```

