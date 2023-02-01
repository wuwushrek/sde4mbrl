# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import itertools
import warnings
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
# from torch import optim as optim

from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import TransitionIterator

from mbrl.models import _NO_META_WARNING_MSG, Model, ModelTrainer

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]


class SGDModelTrainer(ModelTrainer):
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        logger (:class:`mbrl.util.Logger`, optional): the logger to use.
    """

    _LOG_GROUP_NAME = "model_train"

    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        logger: Optional[Logger] = None,
        minibatch_size: int = 128,
    ):
        super().__init__(model, optim_lr, weight_decay, logger)
        self.minibatch_size = minibatch_size

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        updates the model.

        If a validation dataset is provided in the constructor, this method will also evaluate
        the model over the validation data once per training epoch. The method will keep track
        of the weights with the best validation score, and after training the weights of the
        model will be set to the best weights. If no validation dataset is provided, the method
        will keep the model with the best loss over training data.

        Args:
            dataset_train (:class:`mbrl.util.TransitionIterator`): the iterator to
                use for the training data.
            dataset_val (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the validation data.
            num_epochs (int, optional): if provided, the maximum number of epochs to train for.
                Default is ``None``, which indicates there is no limit.
            patience (int, optional): if provided, the patience to use for training. That is,
                training will stop after ``patience`` number of epochs without improvement.
                Ignored if ``evaluate=False`.
            improvement_threshold (float): The threshold in relative decrease of the evaluation
                score at which the model is seen as having improved.
                Ignored if ``evaluate=False`.
            callback (callable, optional): if provided, this function will be called after
                every training epoch with the following positional arguments::

                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - validation score (for ensembles, factored per member)
                    - best validation score so far

            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.update()`` (during training),
                and ``model.eval_score()`` (during evaluation). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is one of ``"train"`` or ``"eval"``, indicating if the callback
                was called during training or evaluation.

            evaluate (bool, optional): if ``True``, the trainer will use ``model.eval_score()``
                to keep track of the best model. If ``False`` the model will not compute
                an evaluation score, and simply train for some number of epochs. Defaults to
                ``True``.

        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        eval_dataset = dataset_train if dataset_val is None else dataset_val

        training_losses, val_scores = [], []
        best_weights: Optional[Dict] = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        best_val_score = self.evaluate(eval_dataset) if evaluate else None
        for epoch in epoch_iter:
            if batch_callback:
                batch_callback_epoch = functools.partial(batch_callback, epoch)
            else:
                batch_callback_epoch = None
            batch_losses: List[float] = []
            for batch in dataset_train:
                loss_and_maybe_meta = self.model.update(batch, self.optimizer)
                if isinstance(loss_and_maybe_meta, tuple):
                    loss = cast(float, loss_and_maybe_meta[0])
                    meta = cast(Dict, loss_and_maybe_meta[1])
                else:
                    # TODO remove this if in v0.2.0
                    warnings.warn(_NO_META_WARNING_MSG)
                    loss = cast(float, loss_and_maybe_meta)
                    meta = None
                batch_losses.append(loss)
                if batch_callback_epoch:
                    batch_callback_epoch(loss, meta, "train")
            total_avg_loss = np.mean(batch_losses).mean().item()
            training_losses.append(total_avg_loss)

            eval_score = None
            model_val_score = 0
            if evaluate:
                eval_score = self.evaluate(
                    eval_dataset, batch_callback=batch_callback_epoch
                )
                val_scores.append(eval_score.mean().item())

                maybe_best_weights = self.maybe_get_best_weights(
                    best_val_score, eval_score, improvement_threshold
                )
                if maybe_best_weights:
                    best_val_score = torch.minimum(best_val_score, eval_score)
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1
                model_val_score = eval_score.mean()

            if self.logger:
                self.logger.log_data(
                    self._LOG_GROUP_NAME,
                    {
                        "iteration": self._train_iteration,
                        "epoch": epoch,
                        "train_dataset_size": dataset_train.num_stored,
                        "val_dataset_size": dataset_val.num_stored
                        if dataset_val is not None
                        else 0,
                        "model_loss": total_avg_loss,
                        "model_val_score": model_val_score,
                        "model_best_val_score": best_val_score.mean()
                        if best_val_score is not None
                        else 0,
                    },
                )
            if callback:
                callback(
                    self.model,
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )

            if patience and epochs_since_update >= patience:
                break

        # saving the best models:
        if evaluate:
            self._maybe_set_best_weights_and_elite(best_weights, best_val_score)

        self._train_iteration += 1
        return training_losses, val_scores
