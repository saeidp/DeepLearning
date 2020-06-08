# Pytorch Lightning

Lightning is a very lightweight wrapper on PyTorch. It defers the core training and validation logic to you and automates the rest.

Lightning is a way to organize your PyTorch code to decouple the science code from the engineering. It's more of a PyTorch style-guide than a framework.

#### The Essential chunks of a Neural Network in Lightning are the following

1. Model architecture
2. Data
3. Forward pass
4. Optimizer
5. Training Step
6. Training and Validation Loops (Lightning Trainer)

#### Lightning provides us with the following methods:

forward — It remains exactly the same in Lightning.  
training_step — This contains the commands that are to be executed when we begin training. it is called for training data.  
testing_step  
validation_step

training_epoch_end — This callback determines what will be done with the results (the outcome of a forward pass) at the end of an epoch.  
testing_epoch_end  
Validation_epoch_end

train_dataloader— This method allows us to set-up the dataset for training and returns a Dataloader object from torch.utils.data module.  
test_dataloader  
val_dataloader

configure_optimizers — It sets up the optimizers such as Adam, SGD, etc. We can even return 2 optimizers (in case of a GAN)

training_end — It contains the piece of code that will be executed when training ends

and many more

### Installing Lightning

`!pip install pytorch_lightning`

By usiing Trainer You get the following tools:

1. Training and validation loop
2. Tensorboard logging
3. Early-stopping
4. Model checkpointing
5. The ability to resume training from wherever you left

### Loading tensorboard

`%load_ext tensorboard`  
`%tensorboard --logdir lightning_logs/`

#### Highlights

Without Lightning, the PyTorch code is allowed to be in arbitrary parts. With Lightning, this is structured.

It is the same exact code for both except that it’s structured in Lightning.

As the project grows in complexity, your code won’t because Lightning abstracts out most of it.

You retain the flexibility of PyTorch because you have full control over the key points in training. For instance, you could have an arbitrarily complex training_step such as a seq2seq

#### Sample Code:

```python
class LightningMNISTClassifier(pl.LightningModule):
    def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)

      logs = {'train_loss': loss}
      return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      tensorboard_logs = {'val_loss': avg_loss}
      return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

trainer = pl.Trainer(max_epochs=2, gpus=[0])
trainer.fit(LightningMNISTClassifier())
```

#### Using Lightning you don't need to:

1. The batch iteration
2. The epoch iteration
3. loss.backward()
4. optimizer.step()
5. optimizer.zero_grad()
6. The validation loop

and more

#### Lightning Trainer Flags

You can assign values to these flags to configure our classifier’s behavior.

1. gpus — Number of GPUs you want to train on
2. max_epochs — Stop training once this number of epochs is reached
3. min_epoch — Force training for at least these many epochs
4. weights_save_path — Directory of where to save weights if specified.
5. precision — Full(32 bit) or half(16 bit)

and many more

#### For example

In the case of GPUs, You don’t have to worry about converting tensors to tensor.to(device=cuda). It automatically figures out the details. You just have to set a few flags. With this, You can even enable 16-bit precision, auto-cluster saving, auto-learning-rate-finder, Tensorboard visualization, etc.
