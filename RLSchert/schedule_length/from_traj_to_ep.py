f1 = open("episode_lengths(sl_50_with_check_num_nw_10).txt", 'r')
f2 = open("episode_lengths(sl_50_with_check_num_nw_10_episodes).txt", 'w')
i = 0

for l in f1:
    i += 1
    if i% 10 == 0:
        f2.write(l)

f1.close()
f2.close()
