import time
import threading
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

from torch.multiprocessing import Process
from torch.multiprocessing import Manager

import environment
import job_distribution
import pg_network
import slow_down_cdf
import math

def discount_adv(x, values, gamma, tau):
    v = x[-1]
    y_batch = np.zeros(len(x))
    adv_batch = np.zeros(len(x))
    y_batch[-1] = x[-1]
    adv_batch[-1] = x[-1]-values[-1]
    for i in reversed(range(len(x)-1)):
        v = x[i] + gamma * v
        y_batch[i] = v
        adv_batch[i] = v - values[i]

    return y_batch, adv_batch

def discount(x, values, gamma, tau):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    gae = 0
    v = x[-1]
    return_v = x[-1]
    y_batch = np.zeros(len(x))
    adv_batch = np.zeros(len(x))
    y_batch[-1] = x[-1]
    gae = gae * gamma * tau
    gae = gae + x[-1] - values[-1]
    adv_batch[-1] = gae
    return_v = values[-1]

    for i in reversed(range(len(x)-1)):
        gae = gae * gamma * tau
        gae = gae + x[i] + gamma * return_v - values[i]
        return_v = values[i]
        v = x[i] + gamma * v
        y_batch[i] = v
        adv_batch[i] = gae
    return y_batch, adv_batch

def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec+1e-7))
    if np.isnan(entropy):
        entropy = 0
    return entropy


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    teacher_acts = []
    old_log_pis = []
    values = []
    rews = []
    entropy = []
    info = []
    done = False

    ob = env.observe()

    for _ in range(episode_max_length):
        act_prob, v, teacher_prob = agent.get_one_act_prob(ob)
        act = np.random.choice(np.arange(act_prob.shape[0]), 1, p=act_prob)
        teacher = np.random.choice(np.arange(teacher_prob.shape[0]), 1, p=teacher_prob)
        a = act[0]
        a_t = teacher[0]
        old_log_pis.append(np.log(act_prob[a]+1e-7))
        values.append(v)
        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        running_jobs = ob[:,-2].reshape(-1).copy()
        feat_size = env.pa.running_feat
        running_jobs = running_jobs.reshape((-1, feat_size))
        running_size = running_jobs.shape[0]
        kill_flag = False
        for i in range(running_size):
            if (running_jobs[i]==0.0).all(): continue
            running_jobs[i,0] = int(running_jobs[i,0]*100)
            running_jobs[i,1] = int(running_jobs[i,1]*100)
            running_jobs[i,3] = int(running_jobs[i,3]*100)
            if running_jobs[i,1]<=running_jobs[i,0] and abs(running_jobs[i,3]-running_jobs[i,0])/running_jobs[i,0]>1.0 and running_jobs[i,3]>10 and running_jobs[i,0]<10:
                print("вот кто убивает черт")
                teacher_acts.append(i+env.pa.num_nw+1)
                kill_flag = True
                break
        if kill_flag == False:
            teacher_acts.append(a_t)

        ob, rew, done, info = env.step(a, repeat=True)
        rew = rew/1000.0
        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: 
            break

    # В функции get_traj() после завершения эпизода:
    episode_length = len(rews)  # Количество шагов в эпизоде
    with open("episode_lengths.txt", "a") as f:
        f.write(f"{episode_length}\n")

    if done == False:
        rews[-1] = -10.0

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'teacher_action':np.array(teacher_acts),
            'old_log_pis': np.array(old_log_pis),
            'values': np.array(values),
            'entropy': entropy,
            'info': info
            }


def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width))

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)
    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, 1, pa.network_input_height, pa.network_input_width))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact


def process_all_info(trajs):
    enter_time = []
    start_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))
        start_time.append(np.array([traj['info'].record[i].start_time for i in range(len(traj['info'].record))]))


    enter_time = np.concatenate(enter_time)
    start_time = np.concatenate(start_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)
    # for i in range(len(enter_time)):
    #     print(enter_time[i], start_time[i], finish_time[i], job_len[i])
    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    cmap = matplotlib.cm.get_cmap("Spectral")
    colors = [cmap(1.*i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))
    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    cmap = matplotlib.cm.get_cmap("Spectral")
    colors = [cmap(1.*i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.yticks(np.arange(0, 19, 2))
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")
    plt.close()


def get_traj_worker(pg_learner, env, pa, result):
    trajs = []
    for i in range(pa.num_seq_per_batch):
        traj = get_traj(pg_learner, env, pa.episode_max_length)
        trajs.append(traj)
    all_ob = concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [] 
    advs = []
    for traj in trajs:
        ret, adv = discount(traj["reward"], traj['values'], pa.discount, pa.ppo_tau)
        rets.append(ret)
        advs.append(adv)
    all_ret = np.concatenate(rets)
    all_adv = np.concatenate(advs)

    # Compute advantage function
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_teacher_action = np.concatenate([traj["teacher_action"] for traj in trajs])

    all_eprews = np.array([rets[i][0] for i in range(pa.num_seq_per_batch)])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    f = open('output.txt', 'w')
    for i in range(len(enter_time)):
        f.write("{} {} {}\n".format(enter_time[i], job_len[i], finish_time[i]))
    f.close()
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
    # print(finish_time[finished_idx], enter_time[finished_idx], job_len[finished_idx])

    all_entropy = np.concatenate([traj["entropy"] for traj in trajs])
    all_old_log_pis = np.concatenate([traj["old_log_pis"] for traj in trajs])

    result.append({"all_ob": all_ob,
                   "all_action": all_action,
                   "all_teacher_action": all_teacher_action,
                   "all_old_log_pis": all_old_log_pis,
                   "all_values": all_ret,
                   "all_adv": all_adv,
                   "all_eprews": all_eprews,
                   "all_eplens": all_eplens,
                   "all_slowdown": all_slowdown,
                   "all_entropy": all_entropy})


def launch(pa, pg_resume=None, pg_teacher=None, render=False, repre='image', end='all_done'):#'no_new_job'):
    cuda = pa.cuda
    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    nw_len_seqs, nw_len_para_seqs, nw_len_running_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=18)

    for ex in range(pa.num_ex):

        print("-prepare for env-" + str(ex))

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_len_para_seqs=nw_len_para_seqs, nw_len_running_seqs=nw_len_running_seqs, nw_size_seqs=nw_size_seqs,
                              render=render, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    for ex in range(pa.num_ex + 1):  # last worker for updating the parameters

        print("-prepare for worker-" + str(ex))

        pg_learner = pg_network.PGLearner(pa)

        if pg_resume is not None:
            if cuda:
                pg_learner.set_net_params(torch.load(pg_resume))
            else:
                pg_learner.set_net_params(torch.load(pg_resume, map_location='cpu'))

        if pg_teacher is not None:
            if cuda:
                pg_learner.set_t_net_params(torch.load(pg_teacher))
            else:
                pg_learner.set_t_net_params(torch.load(pg_teacher, map_location='cpu'))

        pg_learners.append(pg_learner)

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------
    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    #ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pa.output_filename + '_' + str(0) + '.pkl', render=False, plot=True, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

        pg_train = pg_learners[pa.num_ex]
        for param_group in pg_train.optimizer.param_groups:
            pg_train.lr_rate = pg_train.lr_rate*0.95
            param_group['lr'] = pg_train.lr_rate

        ps = []  # threads
        manager = Manager()  # managing return results
        manager_result = manager.list([])

        ex_indices = np.arange(pa.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        grads_all = []
        loss_all = []
        eprews = []
        eplens = []
        all_slowdown = []
        all_entropy = []

        ex_counter = 0
        for ex in range(pa.num_ex):
            ex_idx = ex_indices[ex]
            p = Process(target=get_traj_worker,
                        args=(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, ))
            ps.append(p)

            ex_counter += 1

            if ex == pa.num_ex - 1:

                print(str(ex) + "out of" + str(pa.num_ex))

                ex_counter = 0

                for p in ps:
                    p.start()

                for p in ps:
                    p.join()

                result = []  # convert list from shared memory
                for r in manager_result:
                    result.append(r)

                ps = []
                manager_result = manager.list([])

                all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
                all_action = np.concatenate([r["all_action"] for r in result])
                all_teacher_action = np.concatenate([r["all_teacher_action"] for r in result])
                all_old_log_pis = np.concatenate([r["all_old_log_pis"] for r in result])
                all_values = np.concatenate([r["all_values"] for r in result])
                all_adv = np.concatenate([r["all_adv"] for r in result])

                # Do policy gradient update step, using the first agent
                # put the new parameter in the last 'worker', then propagate the update at the end
                all_size = all_adv.shape[0]
                #all_adv = all_adv.reshape(-1)
                #all_adv = (all_adv-np.mean(all_adv))/(np.std(all_adv)+pa.SMALL_NUM)
                print('adv mean:'+str(np.mean(all_adv)))
                print('values mean:'+str(np.mean(all_values)))
                
                for i in range(pa.ppo_epochs):
                    indice = torch.randperm(all_size)
                    batch_num = math.ceil(all_size/pa.ppo_batch)
                    kl = 0.0
                    v_losses = []
                    p_losses = []
                    for j in range(batch_num):
                        batch_indices = indice[int(j * pa.ppo_batch): int((j+1) * pa.ppo_batch)]
                        loss, kl_curr, p_loss, v_loss = pg_learners[pa.num_ex].train(all_ob[batch_indices], all_action[batch_indices].reshape((-1,1)), all_teacher_action[batch_indices].reshape((-1,1)), all_old_log_pis[batch_indices].reshape((-1,1)), all_values[batch_indices].reshape((-1,1)), all_adv[batch_indices].reshape((-1,1)))
                        v_losses.append(v_loss)
                        p_losses.append(p_loss)
                        kl += kl_curr
                    print('p loss:'+str(np.mean(p_losses)))
                    print('v loss:'+str(np.mean(v_losses)))
                    kl = kl/all_size
                    if kl>1.5*pa.target_kl: 
                        break

                all_eprews.extend([r["all_eprews"] for r in result])

                eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths
                
                all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))
                all_entropy.extend(np.concatenate([r["all_entropy"] for r in result]))

        
        params = pg_learners[pa.num_ex].l_out.state_dict()
        for i in range(pa.num_ex):
            pg_learners[i].l_out.load_state_dict(params)

        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprews))
        print("NumTimesteps: \t %i" % np.sum(eplens))
        # print "Loss:     \t %s" % np.mean(loss_all)
        print("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print("MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews)))
        print("MeanSlowdown: \t %s" % np.mean(all_slowdown))
        print("MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens)))
        print("MeanEntropy \t %s" % (np.mean(all_entropy)))
        print("Elapsed time\t %s %s" % (str(timer_end - timer_start), "seconds"))
        print("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            param_file = pa.output_filename + '_' + str(0) + '.pkl'
            torch.save(pg_learners[pa.num_ex].l_out.state_dict(), param_file)

            pa.unseen = True
            slow_down_cdf.launch(pa, pa.output_filename + '_' + str(0) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end)
            pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 50  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
