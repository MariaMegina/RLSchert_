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


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
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
    old_log_pis = []
    values = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        act_prob, v = agent.get_one_act_prob(ob)
        act = np.random.choice(np.arange(act_prob.shape[0]), 1, p=act_prob)
        a = act[0]
        old_log_pis.append(np.log(act_prob[a]+1e-7))
        values.append(v)
        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: break

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
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
        all_ob_contact[prev_samp : total_samp, 0, :, :] = all_ob[i]

    return all_ob_contact


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

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
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")
    plt.close()


def get_traj_worker(pg_learner, env, pa, result):

    traj = get_traj(pg_learner, env, pa.episode_max_length)

    all_ob = traj['ob']

    # Compute discounted sums of rewards
    rets = discount(traj["reward"], pa.discount)

    baseline = traj['values']

    # Compute advantage function
    advs = rets - baseline
    all_action = traj['action']
    all_adv = advs

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0]])  # episode total rewards
    all_eplens = np.array([len(traj["reward"])])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info([traj])
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    all_entropy = traj["entropy"]

    result.append({"all_ob": all_ob,
                   "all_action": all_action,
                   "all_old_log_pis": traj["old_log_pis"],
                   "all_values": rets,
                   "all_adv": all_adv,
                   "all_eprews": all_eprews,
                   "all_eplens": all_eplens,
                   "all_slowdown": all_slowdown,
                   "all_entropy": all_entropy})


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):
    cuda = pa.cuda
    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)
    nw_len_hat_seqs = np.copy(nw_len_seqs)
    if pa.pred_err > 0:
        #randnum = np.random.random(nw_len_hat_seqs.shape)
        #randnum = -pa.pred_err+2*pa.pred_err*randnum
        randnum_large = np.random.normal(0, pa.pred_err, size=nw_len_hat_seqs.shape)
        randnum_small = np.random.normal(0, pa.small_err, size=nw_len_hat_seqs.shape)
        randnum = np.where(nw_len_seqs>3, randnum_large, randnum_small)
        nw_len_hat_seqs = nw_len_hat_seqs + nw_len_hat_seqs*randnum
        nw_len_hat_seqs = np.round(nw_len_hat_seqs).clip(1,pa.max_job_len).astype(np.int32)

    for ex in range(pa.num_ex):

        print("-prepare for env-" + str(ex))

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_len_hat_seqs=nw_len_hat_seqs, nw_size_seqs=nw_size_seqs,
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

        pg_learners.append(pg_learner)

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

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
                all_old_log_pis = np.concatenate([r["all_old_log_pis"] for r in result])
                all_values = np.concatenate([r["all_values"] for r in result])
                all_adv = np.concatenate([r["all_adv"] for r in result])

                # Do policy gradient update step, using the first agent
                # put the new parameter in the last 'worker', then propagate the update at the end
                all_size = all_adv.shape[0]
                for i in range(pa.ppo_epochs):
                    indice = torch.randperm(all_size)
                    batch_num = int(all_size/pa.ppo_batch)
                    kl = 0.0
                    for j in range(batch_num):
                        batch_indices = indice[int(j * pa.ppo_batch): int((j+1) * pa.ppo_batch)]
                        loss, kl_curr = pg_learners[pa.num_ex].train(all_ob[batch_indices], all_action[batch_indices].reshape((-1,1)), all_old_log_pis[batch_indices].reshape((-1,1)), all_values[batch_indices].reshape((-1,1)), all_adv[batch_indices].reshape((-1,1)))
                        kl += kl_curr
                    kl = kl/all_size
                    if kl>1.5*pa.target_kl: 
                        print(i)
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
