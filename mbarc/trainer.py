import os
from math import ceil

import torch
import torch.nn as nn
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from atari_utils.logger import WandBLogger
from mbarc.adafactor import Adafactor
from mbarc.focal_loss import get_cbf_loss
from mbarc.utils import count_rewards


class Trainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = None
        if self.config.use_wandb:
            self.logger = WandBLogger()

        self.optimizer = Adafactor(self.model.parameters())
        self.reward_optimizer = Adafactor(self.model.reward_estimator.parameters())

        self.model_step = 1
        self.reward_step = 1

        # self.eval_buffer = torch.load(os.path.join('data', 'buffer_eval.pt'), map_location=self.config.device)

    def train(self, epoch, real_env):
        self.train_world_model(epoch, real_env)
        self.train_reward_model(real_env)

    def train_world_model(self, epoch, env, steps=15000):
        # evaluations = 30

        if epoch == 0:
            steps *= 3

        c, h, w = self.config.frame_shape
        rollout_len = self.config.rollout_length
        states, actions, rewards, new_states, dones, values = env.buffer[0]

        if env.buffer[0][5] is None:
            raise BufferError('Can\'t train the world model, the buffer does not contain one full episode.')

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        assert new_states.dtype == torch.uint8
        assert values.dtype == torch.float32

        def get_index():
            index = -1
            while index == -1:
                index = int(torch.randint(len(env.buffer) - rollout_len, size=(1,)))
                for i in range(rollout_len):
                    done, value = env.buffer[index + i][4:6]
                    if done or value is None:
                        index = -1
                        break
            return index

        def get_indices():
            return [get_index() for _ in range(self.config.batch_size)]

        def preprocess_state(state):
            state = state.float() / 255
            noise_prob = torch.tensor([[self.config.input_noise, 1 - self.config.input_noise]])
            noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
            noise_mask = torch.multinomial(noise_prob, state.numel(), replacement=True).view(state.shape)
            noise_mask = noise_mask.to(state)
            state = state * noise_mask + torch.median(state) * (1 - noise_mask)
            return state

        if self.config.use_cbf_loss:
            counts = count_rewards(env.buffer)
            reward_criterion = get_cbf_loss(counts)
        else:
            reward_criterion = nn.CrossEntropyLoss()

        iterator = trange(
            0,
            steps,
            rollout_len,
            desc='Training world model',
            unit_scale=rollout_len
        )
        for i in iterator:
            # Scheduled sampling
            if epoch == 0:
                decay_steps = self.config.scheduled_sampling_decay_steps
                inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
                epsilon = inv_base ** max(decay_steps // 4 - i, 0)
                progress = min(i / decay_steps, 1)
                progress = progress * (1 - 0.01) + 0.01
                epsilon *= progress
                epsilon = 1 - epsilon
            else:
                epsilon = 0

            indices = get_indices()
            frames = torch.empty((self.config.batch_size, c * self.config.stacking, h, w))
            frames = frames.to(self.config.device)

            for j in range(self.config.batch_size):
                frames[j] = env.buffer[indices[j]][0].clone()

            frames = preprocess_state(frames)

            n_losses = 5 if self.config.use_stochastic_model else 4
            losses = torch.empty((rollout_len, n_losses))

            if self.config.stack_internal_states:
                self.model.init_internal_states(self.config.batch_size)

            for j in range(rollout_len):
                _, actions, rewards, new_states, _, values = env.buffer[0]
                actions = torch.empty((self.config.batch_size, *actions.shape))
                actions = actions.to(self.config.device)
                rewards = torch.empty((self.config.batch_size, *rewards.shape), dtype=torch.long)
                rewards = rewards.to(self.config.device)
                new_states = torch.empty((self.config.batch_size, *new_states.shape), dtype=torch.long)
                new_states = new_states.to(self.config.device)
                values = torch.empty((self.config.batch_size, *values.shape))
                values = values.to(self.config.device)
                for k in range(self.config.batch_size):
                    actions[k] = env.buffer[indices[k] + j][1]
                    rewards[k] = env.buffer[indices[k] + j][2]
                    new_states[k] = env.buffer[indices[k] + j][3]
                    values[k] = env.buffer[indices[k] + j][5]

                new_states_input = new_states.float() / 255

                self.model.train()
                frames_pred, reward_pred, values_pred = self.model(frames, actions, new_states_input, epsilon)

                if j < rollout_len - 1:
                    for k in range(self.config.batch_size):
                        if float(torch.rand((1,))) < epsilon:
                            frame = new_states[k]
                        else:
                            frame = torch.argmax(frames_pred[k], dim=0)
                        frame = preprocess_state(frame)
                        frames[k] = torch.cat((frames[k, c:], frame), dim=0)

                loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(frames_pred, new_states)
                clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
                loss_reconstruct = torch.max(loss_reconstruct, clip)
                loss_reconstruct = loss_reconstruct.mean() - self.config.target_loss_clipping

                loss_value = nn.MSELoss()(values_pred, values)
                loss_reward = reward_criterion(reward_pred, rewards)
                loss = loss_reconstruct + loss_value
                if 'online' in self.config.strategy:
                    loss = loss + loss_reward
                if self.config.use_stochastic_model:
                    loss_lstm = self.model.stochastic_model.get_lstm_loss()
                    loss = loss + loss_lstm

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

                tab = [float(loss), float(loss_reconstruct), float(loss_value), float(loss_reward)]
                if self.config.use_stochastic_model:
                    tab.append(float(loss_lstm))
                losses[j] = torch.tensor(tab)

            losses = torch.mean(losses, dim=0)
            metrics = {
                'loss': float(losses[0]),
                'loss_reconstruct': float(losses[1]),
                'loss_value': float(losses[2]),
                'loss_reward': float(losses[3])
            }
            if self.config.use_stochastic_model:
                metrics.update({'loss_lstm': float(losses[4])})

            # if i // rollout_len in [steps // rollout_len - 1 - x * (steps // rollout_len // evaluations)
            #                         for x in range(evaluations)]:
            #     metrics.update(self.eval_world_model())

            if self.logger is not None:
                d = {'model_step': self.model_step, 'epsilon': epsilon}
                d.update(metrics)
                self.logger.log(d)
                self.model_step += rollout_len

            iterator.set_postfix(metrics)

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join('models', 'model.pt'))

    def train_reward_model(self, env):
        if self.config.stack_internal_states:
            raise NotImplementedError('Cannot train the reward model with `stack_internal_states` enabled.')

        batch_size = self.config.reward_model_batch_size
        # evaluations = 30

        if self.config.strategy == 'class_balanced':
            steps = 450
        elif self.config.strategy == 'square_root':
            steps = 475
        elif self.config.strategy == 'progressively_balanced':
            steps = 1000
        elif self.config.strategy == 'mbarc_tmp':
            steps = 1500
        else:
            raise NotImplementedError(f'{self.config.strategy} strategy is not implemented')

        c, h, w = self.config.frame_shape
        states, actions, rewards, new_states, _, _ = env.buffer[0]

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        assert new_states.dtype == torch.uint8

        reward_count = torch.tensor(count_rewards(env.buffer), dtype=torch.float32)
        reward_mask = torch.empty((len(env.buffer),), dtype=torch.long)
        for i, data in enumerate(env.buffer):
            r = int(data[2])
            reward_mask[i] = r
            reward_count[r] += 1

        def weights_to_indices(weights, size=steps * batch_size):
            epsilon = 1e-4
            weights += epsilon
            indices_weights = torch.gather(weights, 0, reward_mask)
            indices = torch.multinomial(indices_weights, size, replacement=True)
            return indices

        if self.config.strategy == 'class_balanced':
            weights = 1 - reward_count / reward_count.sum()
            indices = weights_to_indices(weights)
        elif self.config.strategy == 'square_root':
            weights = 1 - reward_count.sqrt() / reward_count.sqrt().sum()
            indices = weights_to_indices(weights)
        elif self.config.strategy == 'progressively_balanced':
            indices = []
            ct = 50
            for t in range(ct + 1):
                weights_cb = 1 - reward_count / reward_count.sum()
                weights_ib = torch.full((3,), 1 / 3)
                prop = t / ct
                weights = (1 - prop) * weights_ib + prop * weights_cb
                indices.append(weights_to_indices(weights, size=steps * batch_size // ct + 1))
            indices = torch.cat(indices)
        elif self.config.strategy == 'mbarc_tmp':
            d = torch.tensor(self.config.mbarc_dist)
            reward_count = torch.zeros((3,))  # ffw, ctr, or
            for i, data in enumerate(env.buffer):
                if int(data[2]) == 0:
                    raise NotImplementedError('This strategy is only implemented for two classes of rewards')
                elif int(data[2]) == 1:
                    left_condition = False
                    right_condition = False

                    if i > 0:
                        left_condition = int(env.buffer[i - 1][2]) == 2 and not int(env.buffer[i - 1][4]) == 1
                    elif i < len(env.buffer) - 1:
                        right_condition = int(env.buffer[i + 1][2]) == 2

                    if left_condition or right_condition:
                        reward_count[1] += 1
                        reward_mask[i] = 1
                    else:
                        reward_count[0] += 1
                        reward_mask[i] = 0
                else:
                    reward_count[2] += 1
                    reward_mask[i] = 2

            weights = [0] * 3
            for i in range(3):
                if reward_count[i] != 0:
                    weights[i] = d[i] * steps * batch_size / reward_count[i]

            indices = weights_to_indices(torch.tensor(weights))

        def preprocess_state(state):
            state = state.float() / 255
            return state

        metrics = {}
        iterator = trange(
            0,
            steps,
            desc='Training reward model'
        )
        for i in iterator:
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            frames = torch.empty((batch_size, c * self.config.stacking, h, w))
            frames = frames.to(self.config.device)

            for j in range(batch_size):
                frames[j] = env.buffer[batch_indices[j]][0]

            frames = preprocess_state(frames)

            _, actions, rewards, _, _, _ = env.buffer[0]
            actions = torch.empty((batch_size, *actions.shape))
            actions = actions.to(self.config.device)
            rewards = torch.empty((batch_size, *rewards.shape), dtype=torch.long)
            rewards = rewards.to(self.config.device)
            for k in range(batch_size):
                actions[k] = env.buffer[batch_indices[k]][1]
                rewards[k] = env.buffer[batch_indices[k]][2]

            self.model.eval()
            self.model.reward_estimator.train()

            reward_pred = self.model(frames, actions, None, 0)[1]

            loss = nn.CrossEntropyLoss()(reward_pred, rewards)

            self.reward_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.reward_estimator.parameters(), self.config.clip_grad_norm)
            self.reward_optimizer.step()

            metrics.update({'loss_reward': float(loss)})

            # if i in [steps - 1 - x * (steps // evaluations) for x in range(evaluations)]:
            #     metrics.update(self.eval_world_model())

            if self.logger is not None:
                self.logger.log(metrics)

            iterator.set_postfix(metrics)

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.reward_estimator.state_dict(), os.path.join('models', 'reward_model.pt'))

    def eval_world_model(self):
        if self.config.stack_internal_states:
            raise NotImplementedError('Cannot evaluate the world model with `stack_internal_states` enabled.')

        batch_size = 16
        steps = ceil(len(self.eval_buffer) / batch_size)
        states, actions, rewards, new_states, _, _ = self.eval_buffer[0]

        acc_reconstruct = 0
        correct = [0] * 3
        total = [0] * 3
        loss = 0
        loss_reconstruct = 0
        loss_reward = 0
        if self.config.use_stochastic_model:
            loss_lstm = 0

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        assert new_states.dtype == torch.uint8

        iterator = trange(steps, desc='Evaluating world model')
        for i in iterator:
            offset = i * batch_size

            if i + 1 == steps:
                batch_size = len(self.eval_buffer) - (steps - 1) * batch_size

            states, actions, rewards, new_states, _, _ = self.eval_buffer[0]
            states = torch.empty((batch_size, *states.shape))
            states = states.to(self.config.device)
            actions = torch.empty((batch_size, *actions.shape))
            actions = actions.to(self.config.device)
            rewards = torch.empty((batch_size, *rewards.shape), dtype=torch.long)
            rewards = rewards.to(self.config.device)
            new_states = torch.empty((batch_size, *new_states.shape), dtype=torch.long)
            new_states = new_states.to(self.config.device)
            for j in range(batch_size):
                states[j] = self.eval_buffer[offset + j][0]
                actions[j] = self.eval_buffer[offset + j][1]
                rewards[j] = self.eval_buffer[offset + j][2]
                new_states[j] = self.eval_buffer[offset + j][3]

            states = states / 255

            self.model.eval()
            with torch.no_grad():
                new_states_pred, reward_pred, _ = self.model(states, actions, None, 0)

            for j in range(batch_size):
                if (new_states[j] == torch.argmax(new_states_pred[j], 0)).all():
                    acc_reconstruct += 1
                total[rewards[j]] += 1
                if rewards[j] == torch.argmax(reward_pred[j], 0):
                    correct[rewards[j]] += 1

            if total[0] != 0:
                raise NotImplementedError('Evaluating with three rewards classes is not supported.')

            batch_loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(new_states_pred, new_states)
            clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
            batch_loss_reconstruct = torch.max(batch_loss_reconstruct, clip)
            batch_loss_reconstruct = batch_loss_reconstruct.mean() - self.config.target_loss_clipping

            batch_loss_reward = nn.CrossEntropyLoss()(reward_pred, rewards)
            batch_loss = batch_loss_reconstruct + batch_loss_reward
            if self.config.use_stochastic_model:
                batch_loss_lstm = self.model.stochastic_model.get_lstm_loss()
                batch_loss = batch_loss + batch_loss_lstm

            loss += float(batch_loss)
            loss_reconstruct += float(batch_loss_reconstruct)
            loss_reward += float(batch_loss_reward)
            if self.config.use_stochastic_model:
                loss_lstm += float(batch_loss_lstm)

        acc_reconstruct = 100 * acc_reconstruct / len(self.eval_buffer)
        acc_reward = 100 * (correct[1] + correct[2]) / (total[1] + total[2])
        recall_n = 100 * correct[1] / max(1, total[1])
        recall_p = 100 * correct[2] / max(1, total[2])
        loss /= steps
        loss_reconstruct /= steps
        loss_reward /= steps
        if self.config.use_stochastic_model:
            loss_lstm /= steps

        metrics = {
            'val_acc_reconstruct': acc_reconstruct,
            'val_acc_reward': acc_reward,
            'val_recall_n': recall_n,
            'val_recall_p': recall_p,
            'val_loss': loss,
            'val_loss_reconstruct': loss_reconstruct,
            'val_loss_reward': loss_reward
        }
        if self.config.use_stochastic_model:
            metrics.update({'val_loss_lstm': loss_lstm})

        if self.logger is not None:
            self.logger.log(metrics)

        return metrics
