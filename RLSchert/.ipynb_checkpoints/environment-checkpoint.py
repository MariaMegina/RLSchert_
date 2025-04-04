import numpy as np
import math
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import parameters

class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_len_para_seqs=None, nw_len_running_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='no_new_job'):

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0
        self.pred_err = pa.pred_err
        self.small_err = pa.small_err

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_len_para_seqs, self.nw_len_running_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            self.workload = np.zeros(pa.num_res)
            for i in range(pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_len_para_seqs = np.reshape(self.nw_len_para_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_len_running_seqs = np.reshape(self.nw_len_running_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.max_job_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_len_para_seqs = nw_len_para_seqs
            self.nw_len_running_seqs = nw_len_running_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_len_para_seq = np.zeros(simu_len, dtype=int)
        nw_len_running_seq = np.zeros((simu_len, self.pa.max_job_len), dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)

        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_len_para_seq[i], nw_len_running_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_len_para_seq, nw_len_running_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_len_para=self.nw_len_para_seqs[seq_no, seq_idx],
                      job_len_running=self.nw_len_running_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        self.pa.network_input_height = int(self.pa.network_input_height)
        self.pa.network_input_width = int(self.pa.network_input_width)
        if self.repre == 'image':

            image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

            ir_pt = 0

            for i in range(self.pa.num_res):

                image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = np.sort(self.machine.canvas[i, :, :], -1)
                ir_pt += self.pa.res_slot

                for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:  # fill in a block of work
                        image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_job_size

                running_job_feats = []
                for j in range(self.pa.max_run_job):
                    feat = np.zeros(self.pa.running_feat)
                    if self.machine.already_run_job[j] is not None:
                        del_job = self.machine.already_run_job[j]
                        feat[0] = del_job.len_hat/100.0
                        used_time = self.curr_time - del_job.start_time
                        feat[1] = used_time/100.0
                        feat[2] = abs(del_job.len_hat_runtime[used_time] - del_job.len_hat)/del_job.len_hat
                        feat[3] = del_job.len_hat_runtime[used_time]/100.0
                        feat[4] = del_job.enter_time/200.0
                        feat[5] = del_job.res_vec[0]/10.0
                    running_job_feats.append(feat)

            running_job_feats = np.concatenate(running_job_feats).reshape((-1,1))
            image_repr[:, ir_pt: ir_pt + 1] = running_job_feats
            ir_pt += 1
            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                              float(self.extra_info.max_tracking_time_since_last_job)
            ir_pt += 1

            assert ir_pt == image_repr.shape[1]

            return image_repr

    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: int(self.job_backlog.curr_size / backlog_width), : backlog_width] = 1
        backlog[int(self.job_backlog.curr_size / backlog_width), : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        #plt.show()     # manual
        plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len_hat_runtime[max(self.curr_time-j.start_time, 0)])

        for j in self.machine.ready_job:
            reward += self.pa.delay_penalty / float(j.len_hat)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len_hat)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len_hat)

        return reward

    def step(self, a, repeat=False):
        status = None

        done = False
        reward = 0
        info = None
        if a > self.pa.num_nw:
            del_action = a - (self.pa.num_nw + 1)
            if self.machine.already_run_job[del_action] is None:
                status = 'MoveOn'
            else:
                djob = self.machine.already_run_job[del_action]
                used_time = self.curr_time-djob.start_time
                #print('kill:'+str(djob.id)+' '+str(djob.len_hat_runtime[used_time])+' '+str(djob.len_hat)+' '+str(djob.len)+' '+str(used_time))
                self.machine.running_job.remove(djob)
                self.machine.already_run_job[del_action] = None
                del self.machine.already_id[djob.id]
                djob.start_time = -1
                djob.finish_time = -1
                djob.finish_time_hat = -1
                djob.len_hat = djob.len_hat_runtime[used_time]
                self.machine.free_res(djob)
                djob.color = -1
                self.job_record.record[djob.id] = djob
                place = False
                for ji in range(len(self.job_slot.slot)):
                    if self.job_slot.slot[ji] == None:
                        self.job_slot.slot[ji] = djob
                        place = True
                        break
                if place == False:
                    self.job_slot.slot.insert(0, djob)
                self.job_record.record[djob.id] = djob
                copy_list = self.machine.running_job.copy()
                for rj in copy_list:
                    if rj not in self.machine.running_job: continue
                    if rj.start_time>=self.curr_time:
                        self.machine.running_job.remove(rj)
                        self.machine.ready_job.append(rj)
                        self.machine.free_res(rj)
                
                status = 'DELETE'
                #reward = -(used_time/djob.len_hat_runtime[used_time])
        elif len(self.machine.ready_job) > 0:
            status = 'MoveOn'
        elif a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif a < self.pa.num_nw and self.job_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            #ready job insert
            self.machine.ready_job.sort(key=lambda x:x.start_time)
            ready_copy = self.machine.ready_job.copy()
            for reaj in ready_copy:
                if reaj not in self.machine.ready_job: continue
                allocated = self.machine.allocate_job(reaj, self.curr_time)
                if allocated == False: break
                self.machine.ready_job.remove(reaj)
                self.job_record.record[reaj.id] = reaj
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1

            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   len(self.machine.ready_job) == 0 and \
                   len(self.machine.del_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
            if len(self.job_slot.slot) > self.pa.num_nw:
                del self.job_slot.slot[a]
            else: self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0 and self.job_slot.slot[a] == None:
                self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        arange_slot = []
        all_size = len(self.job_slot.slot)
        for i in range(all_size):
            if self.job_slot.slot[i] is not None:
                arange_slot.append(self.job_slot.slot[i])
        non_size = len(arange_slot)
        if non_size > 0:
            arange_slot = sorted(arange_slot, key=lambda x:x.id)
        for i in range(all_size-non_size):
            arange_slot.append(None)
        self.job_slot.slot = arange_slot
        
        ob = self.observe()

        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()
        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, res_vec, job_len, job_len_para, job_len_running, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.len_hat = job_len_para
        self.len_hat_runtime = job_len_running
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.finish_time_hat = -1
        self.color = -1
        self.res_idx = None


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []
        self.ready_job = []
        self.del_job = []

        self.already_run_job = [None]*pa.max_run_job
        self.max_run_job = pa.max_run_job
        self.already_id = {}

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def re_estimate(self, job):
        para = np.array([job.len_hat]*job.len_hat_runtime.shape[0])
        ppred = np.array([job.len]*job.len_hat_runtime.shape[0])
        abs_para = np.abs(para-ppred)
        abs_pred = np.abs(job.len_hat_runtime-ppred)
        new_pred = np.where(abs_pred<=abs_para, job.len_hat_runtime, para)
        job.len_hat_runtime = new_pred
        return job

    def allocate_job(self, job, curr_time):
        job = self.re_estimate(job)
        allocated = False
        job_len = round(job.len_hat)
        for t in range(0, self.time_horizon - job_len):
            new_avbl_res = self.avbl_slot[t: t + job_len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job_len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.finish_time_hat = round(job.start_time + job.len_hat)

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                assert job.finish_time_hat > job.start_time
                job.color = new_color
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                canvas_end_time_hat = job.finish_time_hat - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time_hat):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color
                break
        return allocated

    def free_res(self, job):
        idxes = np.argwhere(self.canvas==job.color)
        cnt = idxes.shape[0]
        if cnt == 0: return False
        for idx in idxes:
            self.canvas[idx[0],idx[1],idx[2]] = 0
            self.avbl_slot[idx[1],idx[0]] += 1.0
        return True

    def runtime_estimate(self, curr_time):
        copy_list = self.running_job.copy()
        for job in copy_list:
            if job not in self.running_job: continue
            if job.start_time < curr_time:
                real_time = job.len
                pred_time = job.len_hat
                used_time = curr_time - job.start_time
                job.len_hat_runtime[used_time] = max(job.len_hat_runtime[used_time], used_time+1)
                if (job.start_time + job.len_hat_runtime[used_time]) == job.finish_time_hat: continue
                if job.start_time + job.len_hat_runtime[used_time] < job.finish_time_hat:
                    delete_res = round(job.finish_time_hat - (job.start_time + job.len_hat_runtime[used_time]))
                    if delete_res > 0:
                        delete_s = round(job.finish_time_hat-curr_time-delete_res)
                        delete_e = round(job.finish_time_hat-curr_time)
                        #print('del res:'+str(job.id)+' '+str(job.len)+' '+str(job.start_time)+' '+str(job.len_hat_runtime[used_time])+' '+str(job.finish_time_hat)+' '+str(curr_time)+' '+str(delete_s)+' '+str(delete_e))
                        self.avbl_slot[delete_s:delete_e,:] += job.res_vec
                        for t in range(delete_s, delete_e):
                            for res in range(self.num_res):
                                self.canvas[res,t,:] = np.where(self.canvas[res,t,:]==job.color, 0, self.canvas[res,t,:])
                        for rj in copy_list:
                            if rj not in self.running_job: continue
                            if rj.start_time>=curr_time:
                                self.running_job.remove(rj)
                                self.ready_job.append(rj)
                                self.free_res(rj)
                    job.finish_time_hat = round(job.start_time + job.len_hat_runtime[used_time])
                else:
                    add_res = round((job.start_time + job.len_hat_runtime[used_time]) - job.finish_time_hat)
                    if add_res == 0: continue
                    add_s = round(job.finish_time_hat-curr_time)
                    add_e = add_s + add_res
                    for rj in copy_list:
                        if rj not in self.running_job: continue
                        if rj.start_time>=curr_time:
                            self.running_job.remove(rj)
                            self.ready_job.append(rj)
                            self.free_res(rj)
                    self.avbl_slot[add_s:add_e,:] -= job.res_vec
                    #print('add res:'+str(job.id)+' '+str(job.finish_time_hat)+' '+str(curr_time)+' '+str(add_s)+' '+str(add_e))
                    for res in range(self.num_res):
                        for t in range(add_s, add_e):
                            avbl_slot = np.where(self.canvas[res, t, :] == 0)[0]
                            self.canvas[res, t, avbl_slot[: job.res_vec[res]]] = job.color
                    job.finish_time_hat = round(job.start_time + job.len_hat_runtime[used_time])

    def time_proceed(self, curr_time):
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0
        copy_list = self.running_job.copy()
        pt=[j.len_hat for j in copy_list]
        #print(pt)
        for job in copy_list:
            if job not in self.running_job: continue
            if job.finish_time <= curr_time: #over estimate finish time
                self.running_job.remove(job)
                #print('finish:'+str(job.id)+' '+str(job.finish_time_hat-job.start_time))
                if job.id in self.already_id:
                    self.already_run_job[self.already_id[job.id]] = None
                    del self.already_id[job.id]
                overest = self.free_res(job)
                if not overest: continue
                for rj in copy_list:
                    if rj not in self.running_job: continue
                    if rj.start_time>=curr_time:
                        self.running_job.remove(rj)
                        self.ready_job.append(rj)
                        self.free_res(rj)
            elif job.finish_time_hat <= curr_time: #under estimate finish time
                add_s = round(job.finish_time_hat-curr_time)
                add_e = add_s + 1
                #print('add res before:'+str(job.id)+' '+str(job.finish_time_hat)+' '+str(curr_time)+' '+str(add_s)+' '+str(add_e))
                for rj in copy_list:
                    if rj not in self.running_job: continue
                    if rj.start_time>=curr_time:
                        self.running_job.remove(rj)
                        self.ready_job.append(rj)
                        self.free_res(rj)
                self.avbl_slot[add_s:add_e,:] -= job.res_vec
                for res in range(self.num_res):
                    for t in range(add_s, add_e):
                        avbl_slot = np.where(self.canvas[res, t, :] == 0)[0]
                        self.canvas[res, t, avbl_slot[: job.res_vec[res]]] = job.color
                job.finish_time_hat += 1
        self.runtime_estimate(curr_time)
        for job in self.running_job:
            if job.start_time<curr_time and job.id not in self.already_id:
                for i in range(self.max_run_job):
                    if self.already_run_job[i] is None:
                        self.already_id[job.id] = i
                        self.already_run_job[i] = job
                        break

class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: " + str(end_time - start_time) + "sec -")


def test_image_speed(render):

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=render, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: " + str(end_time - start_time) + "sec -")


if __name__ == '__main__':
    #test_backlog()
    #test_compact_speed()
    test_image_speed(render=True)


