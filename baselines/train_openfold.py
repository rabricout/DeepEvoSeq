import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint
)

from openfold.utils.logger import PerformanceLoggingCallback

# RAPH: logger for pytorch lightning so we can visualize in tensorboard and json for the custom logger
from pytorch_lightning import loggers as pl_loggers
import json
import datetime
import shutil


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        #self.set_current_epoch(42)
        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1
        
    #@current_epoch.setter
    #def set_current_epoch(self, value):
    #    self._current_epoch = value
        
    def forward(self, batch):
        return self.model(batch)
    
    def _log_metrics(self, metrics, batch_idx, train=True):    # RAPH: function to log raw in a file so we can use it in a separate program
        metrics['batch_idx'] = batch_idx
        metrics['epoch_nb'] = self.current_epoch
        metrics['global_step'] = self.global_step+1    # log the global step for the logs to be consistent when loading model

        target_file = self.log_id+'/metrics.json' if train else self.log_id+'/metrics_eval.json'
        with open(target_file, 'a') as f_log:
            json.dump(metrics, f_log)
        return
    
    def set_log_id(self, m_id):    # RAPH: for our custom log thing
        self.log_id = m_id
        print('self.log_id', self.log_id)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k,v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True
            )

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)
        # Run the model

        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        # Compute loss
        loss, loss_breakdown, metrics = self.loss(
            outputs, batch, _return_breakdown=True
        )
        # Log it
        self._log(loss_breakdown, batch, outputs)
        for i, m in enumerate(metrics):
            self._log_metrics(m, batch_idx)    # RAPH: log raw metrics in a separate file
        #except Exception as error:
        #    print('ERROR', error)
        #    loss = torch.zeros(1)
        #except:
        #    print('ERROR (not a catchable exception)')
        #    loss = torch.zeros(1)
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        #print(torch.cuda.memory_summary())
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
    
        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.
        _, loss_breakdown, metrics = self.loss(
            outputs, batch, _return_breakdown=True
        )
        #print(torch.cuda.memory_summary())
        self._log(loss_breakdown, batch, outputs, train=False)
        
        for i, m in enumerate(metrics):
            self._log_metrics(m, batch_idx, train=False)    # RAPH: log raw metrics in a separate file
        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        return metrics    # RAPH: osef the metrics
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,    # 1e-3
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
#        return torch.optim.Adam(
#            self.model.parameters(),
#            lr=learning_rate,
#            eps=eps
#        )
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        if(not self.model.template_config.enabled):
            ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_from_jax(self, jax_path):
        model_basename = os.path.splitext(
                os.path.basename(
                    os.path.normpath(jax_path)
                )
        )[0]
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(
                self.model, jax_path, version=model_version
        )


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=(str(args.precision) == "16"),
        nature_only=args.nature_only,    # RAPH
        hyper_config=args.hyper_config,    # RAPH
        strat = args.strat
    ) 
    
    model_module = OpenFoldWrapper(config)
    if(args.resume_from_ckpt):
        if(os.path.isdir(args.resume_from_ckpt)):
            last_global_step = get_global_step_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt)
            last_global_step = int(sd['global_step'])
        model_module.resume_last_lr_step(last_global_step)
        logging.info("Successfully loaded last lr step...")
    #if(args.resume_from_ckpt and args.resume_model_weights_only):
    #    if(os.path.isdir(args.resume_from_ckpt)):
    #        sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
    #    else:
    #        sd = torch.load(args.resume_from_ckpt)
    #    sd = {k[len("module."):]:v for k,v in sd.items()}
    #    model_module.load_state_dict(sd)
    #    logging.info("Successfully loaded model weights...")
    if(args.resume_from_jax_params):
        model_module.load_from_jax(args.resume_from_jax_params)
        logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")

    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    #data_module = DummyDataLoader("new_batch.pickle")
    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    #if(args.checkpoint_every_epoch):
    #   mc = ModelCheckpoint(
    #       every_n_epochs=1,
    #       auto_insert_metric_name=False,
    #       save_top_k=-1,
    #   )
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            #dirpath='checkpoints',
            filename='m_model',
            verbose=True,
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=True,
        )
        callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)

    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        #strategy = 'ddp'
        if(args.wandb):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
        strategy = DDPPlugin(find_unused_parameters=False)
        #strategy = 'ddp'
    else:
        strategy = None
    print('strategy', strategy)
        
    if(args.wandb):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")

    # RAPH: add logger, both for tensorboard and for or custom logs
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=str(args.nature_only)+'_deep/'+str(args.strat)+'/', version='version_'+args.my_model_id)
    name = tb_logger.save_dir+tb_logger.name+str(tb_logger.version)
    #name = tb_logger.save_dir+tb_logger.name+'version_'+str(args.my_model_id)
    model_module.set_log_id(name)
    #trainer = pl.Trainer.from_argparse_args(
    #    args,
    #    default_root_dir=args.output_dir,
    #    strategy=strategy,
    #    callbacks=callbacks,
    #    log_every_n_steps=1,    # so that logs are each epoch
    #    logger=tb_logger, #loggers
    #    #profiler='simple',    # RAPH
    #    max_epochs=args.starting_epoch+1,    # RAPH train only on 1 epoch
    #)

    #print(args.num_nodes)
    #input()
    
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=strategy,
        devices='auto',
        num_nodes=args.num_nodes,
        precision=args.precision,
        logger=tb_logger, #loggers
        callbacks=callbacks,
        max_epochs=args.starting_epoch+args.nb_epochs_to_do,    # RAPH
        log_every_n_steps=1,    # so that logs are each epoch
        default_root_dir=args.output_dir,
        num_sanity_val_steps=0,
        #profiler='simple',    # RAPH
        resume_from_checkpoint=args.resume_from_ckpt,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit_loop.epoch_loop.global_step = args.global_step
    trainer.fit_loop.epoch_progress.current.completed = args.starting_epoch  # manually change the epoch in a case of a re-training
    trainer.fit_loop.epoch_progress.current.processed = args.starting_epoch  # manually change the epoch in a case of a re-training
    
    if not os.path.isdir(name):
        os.makedirs(name)    # create dir before the trainer does to write the info file
    with open(name+'/info.txt', 'w') as m_f:
        m_f.write('nature_only: '+str(args.nature_only)+'\n')
        m_f.write('strat: '+'deep'+'\n')
    with open(name+'/config.txt', 'w') as m_f:
        m_f.write(str(config.model.evoformer_mini_stack))
        with open(args.hyper_config, 'r') as hyper_config_file:
            hyper_config_dict = hyper_config_file.read()
        m_f.write(hyper_config_dict)
    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "train_data_dir", type=str,
    #    help="Directory containing training mmCIF files"
    #)
    #parser.add_argument(
    #    "train_alignment_dir", type=str,
    #    help="Directory containing precomputed training alignments"
    #)
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "precomputed_data_dir_train", type=str,
        help='''directory with precomputed dataset'''
    )
    parser.add_argument(
        "precomputed_data_dir_eval", type=str,
        help='''directory with precomputed dataset'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    #parser.add_argument(    # RAPH: add labels we need to predict
    #    "label_dir", type=str, 
    #    help="Directory with labels, meaning the target sequences."
    #)
    #parser.add_argument(    # RAPH: add anchor to have a reference to evaluate the predictions
    #    "anchor_dir", type=str, 
    #    help="Directory with anchor, meaning the sequences to compare with."
    #)
    #parser.add_argument(
    #    "--max_epochs", type=str, default=None,
    #    help='''Max number of epochs, set to 1 for the flexible checkpoint'n'train strategy'''
    #)
    parser.add_argument(
        "--my_model_id", type=str, default='default',
        help="The id to save the model in"
    )
    parser.add_argument(
        "--strat", type=str, default='default',
        help="The baseline strategy to use"
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_jax_params", type=str, default=None,
        help="""Path to an .npz JAX parameter file with which to initialize the model"""
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(    # RAPH: adding eval_chain_data_cache to control eval epoch_len
        "--eval_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(    # RAPH: control the number of samples of eval so we are not obliged to go through all at each epoch
        "--eval_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each eval epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser.add_argument(    # RAPH: set the number of the used GPU
        "--gpu", type=str, default=None,
        help="Gpu number."
    )
    parser.add_argument(    # RAPH: set the number of Gpus we wanna use
        "--nb_gpus", type=int, default=1,
        help="Number of Gpus."
    )
    parser.add_argument(    # RAPH: set if the positions of substitutions are given or not
        "--nature_only", type=bool_type, default=None,
        help="Set to True means that positions of substitutionare given as input."
    )
    parser.add_argument(    # RAPH: set if the positions of substitutions are given or not
        "--precision", type=int, default=32,
        help="Set to True means that positions of substitutionare given as input."
    )
    parser.add_argument(    # RAPH: set if the positions of substitutions are given or not
        "--hyper_config", type=str, default=None,
        help="The config file for hyperparameters"
    )
    #parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    #parser.set_defaults(
    #    num_sanity_val_steps=0,    # RAPH: some validation steps first to spot the COOM of doom (Cuda Out Of Memory)
    #)

    # Remove some buggy/redundant arguments introduced by the Trainer
    #remove_arguments(
    #    parser, 
    #    [
    #        "--accelerator", 
    #        "--resume_from_checkpoint",
    #        "--reload_dataloaders_every_epoch",
    #        "--reload_dataloaders_every_n_epochs",
    #    ]
    #) 

    args = parser.parse_args()
    args.gpus = args.nb_gpus
    args.num_nodes = 1    # I don't exactly know what's the difference between gpus and nodes but as long as the add_argparse_args is not in the newest version of pytorch lightning let's do like that.
    
    #args.nature_only = args.nature_only == 'True'    # RAPH: cast it from str to bool, it's ugly but the parser is bad at dealing with bools)

    args.seed = random.randint(0,424242)
    args.nb_epochs_to_do = 100
    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if(args.resume_from_jax_params is not None and args.resume_from_ckpt is not None):
        raise ValueError("Choose between loading pretrained Jax-weights and a checkpoint-path")

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    if args.gpu != ',' or True:    # RAPH: GPU things
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        #os.environ['NCCL_P2P_DISABLE'] = '1'
        #os.environ['NCCL_P2P_LEVEL'] = 'NVL'
        os.environ['NCCL_DEBUG']='INFO'
        os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
        #torch.backends.cudnn.benchmark=False
        #torch.backends.cudnn.deterministic=True
        #torch.distributed.init_process_group(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=60), world_size=- 1, rank=-1, store=None, group_name='', pg_options=None)
        #os.environ['NCCL_IB_DISABLE'] = '1'
        #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        #os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
        #torch.cuda.set_device(int(args.gpu))
        print('GPU:'+args.gpu)
        
    #checkpoint loading dir if exists, if so manage the checkpoint
    m_dir = 'lightning_logs/'+str(args.nature_only)+'_deep/'+'version_'+args.my_model_id+'/checkpoints/'
    if os.path.isdir(m_dir):
        m_checkpoints = os.listdir(m_dir)
        if len(m_checkpoints) == 0:
            args.resume_from_ckpt=None
        elif len(m_checkpoints) == 1:
            args.resume_from_ckpt = m_dir+m_checkpoints[0]
        else:
            v_version = [c for c in m_checkpoints if '-v' in c]
            v_notVersion = [c for c in m_checkpoints if '-v' not in c]
            if len(v_version) > 1 and len(v_notVersion) > 1:
                print('ERROR in loading checkpoint')
                print(v_version, v_notVersion)
                exit()
            else:
                #print('Remove', m_dir+v_notVersion[0])
                #print('Move', m_dir+v_version[0], m_dir+v_notVersion[0])
                #input()
                os.remove(m_dir+v_notVersion[0])
                os.rename(m_dir+v_version[0], m_dir+v_notVersion[0])
                args.resume_from_ckpt = m_dir+v_notVersion[0]
                
    #Kinda dirty stuff, we use the log file to find the id of the last epoch to be able to split the training by epoch (for Jean Zay)
    log_file='lightning_logs/'+str(args.nature_only)+'_deep/'+'version_'+args.my_model_id+'/metrics.json'
    args.starting_epoch = 0
    args.global_step = 0
    if os.path.isfile(log_file):    # if the pretrained model exists, get the number of the epoch and the step
        with open(log_file) as f_json:
            logs = f_json.read()
            split_logs = logs.split('}')[:-1]
            last_log = json.loads(split_logs[-1]+'}') # add } because it's removed by the split
        args.starting_epoch = last_log['epoch_nb'] + 1
        args.global_step = last_log['global_step'] #len(split_logs)
    main(args)
