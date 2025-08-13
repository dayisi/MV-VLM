# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Xinlong Hou and collaborators

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from pprint import pprint
from configs.config import parser
from dataset.data_module_textEmb import DataModule
from lightning_tools.callbacks import add_callbacks
from models.MVVLM_TextEncoder_screen import MVVLM
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from datetime import datetime

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = MVVLM.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = MVVLM(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)
    # Capture the end time
    end_time = datetime.now()
    print(f"End Time: {end_time}")

    # Calculate the duration
    duration = end_time - start_time
    print(f"Duration: {duration}")

if __name__ == '__main__':
    main()