num_cpu = []
start_time = []
finish_time = []

f = open("RLScheduler_schedule.txt", 'r')
for s in f:
    str = s.split()
    num_cpu.append(str[0])
    start_time.append(str[3])
    finish_time.append(str[4])

f.close()
up_count = [0] * 170
for i in range(1, len(start_time)):
    for j in range(int(start_time[i]), int(finish_time[i])):
        up_count[j-1] += int(num_cpu[i])
    print(start_time[i], finish_time[i], num_cpu[i])

time = 0
for i in up_count:
    str = f"{time} " + '*' * i + '-' * (20 - i)
    with open("rlschert_used_processors.txt", "a") as f:
        f.write(f"{str}\n")
    time += 1