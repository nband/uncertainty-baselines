import pdb
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import tree
from jax.experimental import optimizers
from jax import jit

from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils_linearization
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import (
    partition_params,
    predicate_batchnorm)
from baselines.diabetic_retinopathy_detection.utils import get_diabetic_retinopathy_class_balance_weights

dtype_default = jnp.float32
eps = 1e-6


class Objectives_hk:
    def __init__(
        self,
        architecture,
        apply_fn,
        predict_f,
        predict_f_deterministic,
        predict_y,
        predict_y_multisample,
        predict_y_multisample_jitted,
        output_dim,
        kl_scale: str,
        predict_f_multisample,
        predict_f_multisample_jitted,
        noise_std,
        regularization,
        n_samples,
        full_cov,
        prior_type,
        stochastic_linearization,
        linear_model,
        full_ntk=False,
        kl_type=0,
    ):
        self.architecture = architecture
        self.apply_fn = apply_fn
        self.predict_f = predict_f
        self.predict_f_deterministic = predict_f_deterministic
        self.predict_y = predict_y
        self.predict_f_multisample = predict_f_multisample
        self.predict_f_multisample_jitted = predict_f_multisample_jitted
        self.predict_y_multisample = predict_y_multisample
        self.predict_y_multisample_jitted = predict_y_multisample_jitted
        self.output_dim = output_dim
        self.kl_scale = kl_scale
        self.regularization = regularization
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.stochastic_linearization = stochastic_linearization
        self.linear_model = linear_model
        self.full_ntk = full_ntk
        self.kl_type = kl_type

    @partial(jit, static_argnums=(0, 10, 11))
    def objective_and_state(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        class_weight,
        objective_fn,
    ):
        is_training = True
        params = hk.data_structures.merge(trainable_params, non_trainable_params,)

        objective = objective_fn(
            params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            is_training,
            class_weight
        )

        state = self.apply_fn(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
            stochastic=True,
            is_training=is_training,
        )[1]

        return objective, state

    @partial(jit, static_argnums=(0,))
    def accuracy(self, preds, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)
        return jnp.mean(predicted_class == target_class)

    def _crossentropy_log_likelihood(self, preds_f_samples, targets):
        log_likelihood = jnp.mean(
            jnp.sum(
                jnp.sum(
                    targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1
                ),
                axis=-1,
            ),
            axis=0,
        )
        return log_likelihood

    def _crossentropy_log_likelihood_with_class_weights(self, preds_f_samples, targets):
        # get_positive_empirical_prob
        # TODO: remove the hardcoded 1
        minibatch_positive_empirical_prob = targets[:, 1].sum() / targets.shape[0]
        minibatch_class_weights = (
            get_diabetic_retinopathy_class_balance_weights(
                positive_empirical_prob=minibatch_positive_empirical_prob))

        log_likelihoods = jnp.mean(jnp.sum(
                    targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1
                ),
            axis=0
        )
        weights = jnp.where(
            targets[:, 1] == 1,
            minibatch_class_weights[1],
            minibatch_class_weights[0]
        )
        reduced_value = jnp.sum(jnp.multiply(log_likelihoods, weights))
        return reduced_value

    def _function_kl(
        self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Evaluate the multi-output KL between the function distribution obtained by linearising BNN around
        params, and the prior function distribution represented by (`prior_mean`, `prior_cov`)

        @param inputs: used for computing scale, only the shape is used
        @param inducing_inputs: used for computing scale and function distribution used in KL

        @return:
            kl: scalar value of function KL
            scale: scale to multiple KL with
        """
        # TODO: Maybe change "params_deterministic" to "params_model"
        params_mean, params_log_var, params_deterministic = partition_params(params)
        scale = compute_scale(self.kl_scale, inputs, inducing_inputs.shape[0])

        # mean, cov = self.linearize_fn(
        #     params_mean=params_mean,
        #     params_log_var=params_log_var,
        #     params_batchnorm=params_batchnorm,
        #     state=state,
        #     inducing_inputs=inducing_inputs,
        #     rng_key=rng_key,
        # )

        if self.kl_type == 0:
            mean, cov = utils_linearization.bnn_linearized_predictive(
                self.apply_fn,
                params_mean,
                params_log_var,
                params_deterministic,
                state,
                inducing_inputs,
                rng_key,
                self.stochastic_linearization,
                self.full_ntk,
            )
        elif self.kl_type == 1:
            print('*' * 100)
            print(f"new kl_type is used! The inducing inputs has shape {inducing_inputs}")
            mean = jnp.mean(inducing_inputs, 0)
            cov = jnp.square(jnp.std(inducing_inputs, 0))
        else:
            raise NotImplementedError(self.kl_type)


        kl = utils.kl_divergence(
            mean,
            prior_mean,
            cov,
            prior_cov,
            self.output_dim,
            self.full_cov,
            self.prior_type,
        )

        return kl, scale

    def _nll_loss_classification(
        self, params, state, inputs, targets, rng_key, is_training, class_weight
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = (
            self.crossentropy_log_likelihood(preds_f_samples, targets, class_weight)
            / targets.shape[0]
        )
        loss = -log_likelihood
        return loss

    def _elbo_fsvi_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
        class_weight,
        loss_type,
        l2_strength,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        if self.kl_type == 1:
            permutation = jax.random.permutation(key=rng_key, x=inputs.shape[0])
            inducing_inputs_output_vals = preds_f_samples[:, permutation[:inducing_inputs.shape[0]], :]
            kl, scale = self.function_kl(
                params, state, prior_mean, prior_cov, inputs, inducing_inputs_output_vals, rng_key,
            )
        else:
            kl, scale = self.function_kl(
                params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
            )

        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets,
                                                          class_weight)
        if loss_type == 1:
            elbo = log_likelihood - scale * kl
        elif loss_type == 2:
            elbo = log_likelihood / inputs.shape[0] - scale * kl / inducing_inputs.shape[0]
        elif loss_type == 3:
            elbo = (log_likelihood - scale * kl) / inputs.shape[0]
        elif loss_type == 4:
            elbo = log_likelihood / inputs.shape[0]
        elif loss_type == 5:
            batch_norm_params = hk.data_structures.filter(predicate_batchnorm, params)
            l2_loss = jnp.sum(jnp.stack([jnp.sum(x * x) for x in tree.flatten(batch_norm_params)]))
            elbo = (log_likelihood - scale * kl) / inputs.shape[0] - l2_loss * l2_strength
        else:
            raise NotImplementedError(loss_type)

        return elbo, log_likelihood, kl, scale

    @partial(jit, static_argnums=(0, 10, 11))
    def map_loss_classification(
            self,
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            class_weight,
            loss_type,
    ):
        (loss, negative_log_likelihood), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            class_weight,
            self._map_loss_classification,
        )
        return loss, {
            "state": state,
            "loss": loss,
            "log_likelihood": -negative_log_likelihood * targets.shape[0]
        }

    def _map_loss_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
        class_weight,
    ):
        negative_log_likelihood = self._nll_loss_classification(
            params, state, inputs, targets, rng_key, is_training, class_weight
        )
        reg_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]
        loss = (
            negative_log_likelihood
            + self.regularization * jnp.square(optimizers.l2_norm(reg_params))
        )
        return loss, negative_log_likelihood


    @partial(jit, static_argnums=(0, 3))
    def crossentropy_log_likelihood(self, preds_f_samples, targets, class_weight):
        if class_weight:
            return self._crossentropy_log_likelihood_with_class_weights(preds_f_samples, targets)
        else:
            return self._crossentropy_log_likelihood(preds_f_samples, targets)

    @partial(jit, static_argnums=(0,))
    def function_kl(
        self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ):
        return self._function_kl(
            params=params,
            state=state,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            inputs=inputs,
            inducing_inputs=inducing_inputs,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0,))
    def nll_loss_classification(
        self, trainable_params, non_trainable_params, state, inputs, targets, rng_key
    ):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            inputs,
            targets,
            rng_key,
            self._nll_loss_classification,
        )

    @partial(jit, static_argnums=(0, 10, 11, 12))
    def nelbo_fsvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        class_weight: bool,
        loss_type: int,
        l2_strength: float,
    ):
        params = hk.data_structures.merge(trainable_params, non_trainable_params, )
        is_training = True
        elbo, log_likelihood, kl, scale = self._elbo_fsvi_classification(
            params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            is_training,
            class_weight,
            loss_type,
            l2_strength,
        )

        state = self.apply_fn(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
            stochastic=True,
            is_training=is_training,
        )[1]

        return -elbo, {"state": state, "elbo": elbo, "log_likelihood": log_likelihood,
                       "kl": kl, "scale": scale, "loss": -elbo}


@partial(jit, static_argnums=(0,))
def compute_scale(kl_scale: str, inputs: jnp.ndarray, n_inducing_inputs: int) -> float:
    if kl_scale == "none":
        scale = 1.0
    elif kl_scale == "equal":
        scale = inputs.shape[0] / n_inducing_inputs
    elif kl_scale == "normalized":
        scale = 1.0 / n_inducing_inputs
    else:
        scale = dtype_default(kl_scale)
    return scale
