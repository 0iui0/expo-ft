"""GR00T-specific batch processor for mixing online and offline replay buffer data."""

import logging

import jax

from expo_ft.agents import restore_replay_buffer
from expo_ft.utils.train_utils import clear_batch, combine_batches


logger = logging.getLogger(__name__)


class Gr00tBatchProcessor:
    """Builds critic, actor, and optional offline batches for EXPOLearnerGR00T.

    Simplified version of ``BatchProcessor`` — no OpenPI format conversion
    since ``Gr00tReplayBuffer`` already outputs raw data with the correct keys.
    """

    def __init__(
        self,
        replay_buffer,
        offline_replay_buffer=None,
        batch_size: int = 64,
        utd_ratio: int = 20,
        offline_ratio: float = 0.0,
        actor_success_only: bool = True,
        dataset=None,
    ):
        if dataset is not None:
            if offline_ratio == 0:
                replay_buffer.insert_dataset(dataset)
                logger.info("Inserted dataset into online replay buffer")
            else:
                offline_replay_buffer.insert_dataset(dataset)
                logger.info("Inserted dataset into offline replay buffer")

        self.replay_buffer = replay_buffer
        self.offline_replay_buffer = offline_replay_buffer
        self.batch_size = batch_size
        self.offline_ratio = offline_ratio
        self.actor_success_only = actor_success_only

        self._ep_buffer_start = replay_buffer._insert_index

    def insert_transition(self, transition_dict):
        self.replay_buffer.insert(transition_dict)

    def on_episode_start(self):
        self._ep_buffer_start = self.replay_buffer._insert_index

    def on_episode_done(self, success):
        if success:
            self.replay_buffer.mark_episode_success(
                self._ep_buffer_start, self.replay_buffer._insert_index
            )
        self._ep_buffer_start = self.replay_buffer._insert_index

    def restore(self, checkpoint_dir, up_to_step=None):
        restore_replay_buffer(checkpoint_dir, self.replay_buffer, up_to_step=up_to_step)
        self.replay_buffer.restore_success_marks()

    def next_batch(self, combine_rng):
        """Return (critic_batch, actor_batch, new_rng) for one update step.

        Both batches are raw dicts as returned by ``Gr00tReplayBuffer.sample_jax``.
        """
        if self.offline_ratio == 0:
            batch = self.replay_buffer.sample_jax(self.batch_size * 20)  # UTD multiplier
            new_rng = combine_rng
        else:
            online = self.replay_buffer.sample_jax(int(self.batch_size * 20 * (1 - self.offline_ratio)))
            offline = self.offline_replay_buffer.sample_jax(int(self.batch_size * 20 * self.offline_ratio))
            shuffle_key, new_rng = jax.random.split(combine_rng)
            batch = combine_batches(online, offline, rng=shuffle_key)
            clear_batch(online)
            clear_batch(offline)

        actor_batch = None
        if self.actor_success_only:
            actor_batch = self._sample_success_batch(new_rng)
            if actor_batch is not None:
                new_rng_parts = jax.random.split(new_rng)
                new_rng = new_rng_parts[0]

        return batch, actor_batch, new_rng

    def _sample_success_batch(self, rng):
        if self.offline_ratio == 0:
            return self.replay_buffer.sample_jax(self.batch_size, success_only=True)

        online = self.replay_buffer.sample_jax(int(self.batch_size * (1 - self.offline_ratio)), success_only=True)
        if online is not None:
            offline = self.offline_replay_buffer.sample_jax(int(self.batch_size * self.offline_ratio), success_only=True)
            if offline is not None:
                shuffle_key, _ = jax.random.split(rng)
                return combine_batches(online, offline, rng=shuffle_key)

        return self.offline_replay_buffer.sample_jax(self.batch_size, success_only=True)
