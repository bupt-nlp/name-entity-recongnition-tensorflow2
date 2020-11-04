import re
from typing import Callable, List, Optional

import tensorflow as tf


class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Apply warmup scheduler on a given learning rate decay schedule"""

    def __init__(self, learning_rate: float, warmup_steps: int, decay_schedule_fn: Callable = None,
                 power: float = 1., name: str = None):
        self.learning_rate: float = learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def get_config(self) -> dict:
        """return the configuration of warmup scheduler"""
        return {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_schedule_fn": self.decay_schedule_fn,
            "power": self.power,
            "name": self.name,
        }

    def __call__(self, step):
        """schedule the learning rate by steps, usually this is made by training graph"""
        with tf.name_scope(self.name or 'Warmup') as name:
            # 1. cast the
            global_step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)

            warmup_percent = warmup_steps / global_step
            warmup_learning_rate = self.learning_rate * tf.math.pow(warmup_percent, self.power)

            return tf.cond(
                warmup_steps < global_step,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name
            )

    def create_optimization(self, init_learning_rate: float, num_train_steps: int, warmup_step: int = None):
        """create optimization with learning rate scheduler"""
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_learning_rate,
            decay_steps=num_train_steps,
            end_learning_rate=0.0001
        )
        if warmup_step:
            learning_rate_fn = Warmup(
                learning_rate=init_learning_rate,
                warmup_steps=warmup_step,
                decay_schedule_fn=learning_rate_fn,
            )
        optimization = None


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Enable weight l2 weight decay and clip_by_global_norm on gradients

    Just adding the square of the weights to the loss function is *not* the
    correct way of using l2 regularization/weight decay with Adam, since that
    will interactive"""

    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-7, ams_grad: bool = False, weight_decay_rate: float = 0.,
                 include_in_weight_decay: Optional[List[str]] = None,
                 exclude_from_weight_decay: Optional[List[str]] = None):
        super().__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                         epsilon=epsilon, amsgrad=ams_grad, name="AdamWeightDecay")

        self.weight_decay_rate = weight_decay_rate
        self.include_in_weight_decay = include_in_weight_decay
        self.exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """create optimizer from config with warmup custom object"""
        if not custom_objects:
            custom_objects = {}
        custom_objects["Warmup"] = Warmup
        return super(AdamWeightDecay, cls).from_config(config=config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """"""
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state['weight_decay_rate'] = tf.constant(self.weight_decay_rate, name='adam_weight_decay_rate')

    def _do_use_weight_decay(self, param_name: str) -> bool:
        """whether to user l2 weight decay for `param_name`"""
        if self.weight_decay_rate == 0.:
            return False

        if self.include_in_weight_decay:
            for weight_name in self.include_in_weight_decay:
                if re.search(weight_name, param_name):
                    return True
        if self.exclude_from_weight_decay:
            for weight_name in self.exclude_from_weight_decay:
                if re.search(weight_name, param_name):
                    return False
        return True

    def _decay_weight_op(self, variable: tf.Variable, learning_rate: float, apply_state: dict):
        """apply the weight operation on variables"""
        do_decay = self._do_use_weight_decay(variable.name)
        if do_decay:
            return variable.assign_add(
                learning_rate * apply_state['weight_decay_rate'] * variable,
                use_locking=self._use_locking
            )
        return tf.no_op()

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        """schedule weight decay gradients"""
        grads, variables = list(zip(*grads_and_vars))
        # clip the global gradient by normalization
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super().apply_gradients(zip(grads, variables))

    def _get_learning_rate(self, variable_device: str, variable_type: str, apply_state: dict):
        """retrieves the learning rate with given state"""
        if apply_state is None:
            return self._decayed_lr()


