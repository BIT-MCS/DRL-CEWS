import torch
from uav2_charge2.exper_dppo_curiosity.envs import *
from uav2_charge2.exper_dppo_curiosity.params import Params
from uav2_charge2.exper_dppo_curiosity.storage import RolloutStorage
from common_setting.path_setting import PATH
from uav2_charge2.exper_dppo_curiosity.model import Model, ICMModel
from uav2_charge2.exper_dppo_curiosity.icm_agent import ICMAgent
from uav2_charge2.exper_dppo_curiosity.RFAgent import RFAgent
from uav2_charge2.exper_dppo_curiosity.RNDAgent import RndAgent
from uav2_charge2.exper_dppo_curiosity.EMDAgent import EMDAgent
import random
from util.utils import Util
import csv
import matplotlib.pyplot as plt
import os
import datetime
import time

params = Params()
torch.set_num_threads(1)
if params.device_num == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(params.device_num))
# device = torch.device("cuda" if params.use_cuda else "cpu")
util = Util(device)


def action_convert(action):
    action = (util.to_numpy(action[0]), util.to_numpy(action[1]))
    num_of_process = action[0].shape[0]
    num_of_uav = int(action[1].shape[1])
    action_new = np.zeros([num_of_process, num_of_uav, 3], dtype=np.float32)
    for i in range(num_of_process):
        state = action[0][i, 0]
        for j in range(num_of_uav):
            if state % 2 == 0:
                action_new[i, j, 0] = 1  # collect data
            else:
                action_new[i, j, 0] = -1  # charge
            state = state // 2
            action_new[i, j, 1] = params.discrete_action[action[1][i, j]][0]
            action_new[i, j, 2] = params.discrete_action[action[1][i, j]][1]
    action_new = action_new.reshape([num_of_process, -1])
    return action_new


def train(rank, traffic_light, counter, shared_model, shared_icm_model, shared_grad_buffers, shared_grad_buffer_icm,
          local_time, son_process_counter):
    print('in UAV1 training process')
    seed = params.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_process = 1
    # ----------------make environment----------------------
    env = Make_Env(num_process, device, params.exploration_steps, local_time, rank, params.use_sparse_env)
    # -----------------load parameters----------------------
    use_icm = params.use_icm
    obs_shape = env.observ_shape
    uav_num = params.uav_num
    clip = params.clip
    use_gae = params.use_gae
    ent_coeff = params.ent_coeff
    value_coeff = params.value_coeff
    clip_coeff = params.clip_coeff
    gamma = params.gamma
    gae_param = params.gae_param
    use_obs_norm = params.use_obs_norm
    use_adv_norm = params.use_adv_norm
    beta = params.beta
    icm_feature = params.icm_feature
    use_sparse_env = params.use_sparse_env

    # --------------create name---------------------------
    method_name = 'U2C2-DPPO'
    if use_icm:
        method_name += ('-' + str(params.icm_feature))
    if use_sparse_env:
        method_name += '-Sparse'
    else:
        method_name += '-Dense'

    # -----------------create storage---------------------
    rollout = RolloutStorage(params.exploration_steps, params.mini_batch_num, obs_shape, uav_num)
    rollout.to(device)

    # ---------------create local model---------------------
    local_ppo_model = Model(obs_shape, uav_num, params.cat_ratio, params.dia_ratio, device, len(params.discrete_action))
    local_ppo_model.to(device)
    curiosity_agent = None
    if use_icm:
        if icm_feature == "IDF":
            curiosity_agent = ICMAgent(obs_shape, uav_num, params.eta, len(params.discrete_action), device)
        elif icm_feature == "RF":
            curiosity_agent = RFAgent(uav_num, params.eta, device)
        elif icm_feature == "RND":
            curiosity_agent = RndAgent(obs_shape, params.eta, device)
        elif icm_feature == "EMD":
            curiosity_agent = EMDAgent(uav_num, params.eta, len(params.discrete_action), device, rank)

    episode_length = 0
    interact_time = 0
    # --------------define file writer-----------------------
    file_root_path = os.path.join(PATH.root_path, str(local_time) + '/' + str(rank) + '/file')
    os.makedirs(file_root_path)

    loss_file = open(os.path.join(file_root_path, 'loss.csv'), 'w', newline='')
    loss_writer = csv.writer(loss_file)
    reward_file = open(os.path.join(file_root_path, 'reward.csv'), 'w', newline='')
    reward_writer = csv.writer(reward_file)
    if rank == 0:
        model_root_path = os.path.join(PATH.root_path, str(local_time) + '/ckpt/')
        os.makedirs(model_root_path)

    # load local model parameters
    local_ppo_model.load_state_dict(shared_model.state_dict())
    if use_icm:
        curiosity_agent.icm.load_state_dict(shared_icm_model.state_dict())
    done = False
    init_tau = 1.
    end_tau = 0.1
    tau = init_tau
    while True:

        if episode_length >= params.max_episode_length:
            print('training over')
            break
        if rank == 0:
            print('---------------in episode ', episode_length, '-----------------------')

        tau -= (init_tau - end_tau) / params.max_episode_length
        step = 0
        av_reward = 0

        obs, info = env.reset()
        pos = info['uav_position']
        rollout.after_update(obs, pos)

        while step < params.exploration_steps:
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                value, action, action_log_probs = local_ppo_model.act(obs, tau)
                obs, reward, done, info = env.step(action_convert(action), current_step=step)
                pos = info['uav_position']

            av_reward += reward
            # ---------judge if game over --------------------
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
            # ----------add to memory ---------------------------
            rollout.insert(obs.detach(), (action[0].detach(), action[1].detach()), action_log_probs.detach(),
                           value.detach(), reward.detach(), pos.detach(), masks.detach())
            # if episode_length % 10 == 0 and rank == 0:
            #     env.render()
            step = step + 1

        # --------------update---------------------------
        if use_icm:
            with torch.no_grad():
                pos_rollout, next_pos_rollout, action_rollout = rollout.icm_tuple()
                intrinsic_reward = curiosity_agent.compute_intrinsic_reward(pos_rollout, next_pos_rollout,
                                                                            action_rollout)
                rollout.update_reward(intrinsic_reward)

        done = done[0]
        with torch.no_grad():
            if done:
                next_value = torch.zeros(1)
            else:
                next_value = local_ppo_model.get_value(rollout.obs[-1:])

        rollout.compute_returns(next_value.detach(), use_gae, gamma, gae_param)

        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        if use_adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        av_value_loss = 0
        av_policy_loss = 0
        av_forward_loss = 0
        av_inverse_loss = 0
        av_ent_loss = 0
        loss_cnt = 0

        for _ in range(params.ppo_epoch):
            data_generator = rollout.feed_forward_generator(advantages)
            for samples in data_generator:
                signal_init = traffic_light.get()
                torch.cuda.empty_cache()
                obs_batch, next_obs_batch, action_batch, old_values, return_batch, masks_batch, \
                old_action_log_probs, advantages_batch, pos_batch, next_pos_batch = samples

                cur_values, cur_action_log_probs, dist_entropy = local_ppo_model.evaluate_actions(obs_batch,
                                                                                                  action_batch)

                # ----------use ppo clip to compute loss------------------------
                ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages_batch
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values + (cur_values - old_values).clamp(-clip, clip)
                value_losses = (cur_values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                value_loss = value_loss * value_coeff
                action_loss = action_loss * clip_coeff
                ent_loss = dist_entropy * ent_coeff
                # ------------------ for curiosity driven--------------------------
                forward_loss = 0
                inverse_loss = 0
                if use_icm:
                    inverse_loss, forward_loss = curiosity_agent.compute_loss(pos_batch, next_pos_batch,
                                                                              action_batch[1])
                    curiosity_agent.icm.zero_grad()
                curiosity_loss = beta * forward_loss + (1 - beta) * inverse_loss
                total_loss = value_loss + action_loss - ent_loss + 10 * curiosity_loss

                local_ppo_model.zero_grad()

                total_loss.backward()
                # ----------------- add model gradient ----------------------------
                shared_grad_buffers.add_gradient(local_ppo_model)
                if use_icm:
                    shared_grad_buffer_icm.add_gradient(curiosity_agent.icm)
                    # if rank == 0:
                    #     curiosity_agent.icm.print_grad()
                av_forward_loss += float(forward_loss)
                av_inverse_loss += float(inverse_loss)
                av_value_loss += float(value_loss)
                av_policy_loss += float(action_loss)
                av_ent_loss += float(ent_loss)
                loss_cnt += 1

                # ---------wait for update----------------------
                counter.increment()
                while traffic_light.get() == signal_init:
                    pass
                # update local_ppo_model and local_icm_model
                local_ppo_model.load_state_dict(shared_model.state_dict())
                if use_icm:
                    curiosity_agent.icm.load_state_dict(shared_icm_model.state_dict())

        av_value_loss /= loss_cnt
        av_policy_loss /= loss_cnt
        av_inverse_loss /= loss_cnt
        av_forward_loss /= loss_cnt
        av_ent_loss /= loss_cnt
        # --------------- draw & log -----------------------------
        # if episode_length % 10 == 0:
        env.draw_path(episode_length)

        # ---------------- average reward -----------------------------
        av_reward = av_reward.cpu().mean().numpy()
        reward_writer.writerow([np.mean(av_reward)])

        # -------------- average loss ----------------------------------
        if rank == 0:
            print('average reward: ', av_reward)
            print('value_loss: ', av_value_loss, 'policy_loss:', av_policy_loss, 'forward_loss: ', av_forward_loss,
                  'inverse_loss', av_inverse_loss)

        loss_writer.writerow(
            [episode_length, av_value_loss, av_policy_loss, av_ent_loss, av_forward_loss, av_inverse_loss])

        episode_length = episode_length + 1
        if episode_length % 500 == 0 and rank == 0:
            model_path = os.path.join(model_root_path, 'ppo_model_' + str(episode_length) + '.pt')
            torch.save(local_ppo_model.state_dict(), model_path)

    loss_file.close()
    reward_file.close()
    son_process_counter.increment()
