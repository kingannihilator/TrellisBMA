import sys




def countFullClusters(max_drift, min_traces, normal_len):
    sys.stdin = open('recon_input/aging/reads.txt')
    count = 0
    cluster = []
    while True:
        try:
            line = input().strip()
            # if i % 100 == 0:
            #     print(i)
            if len(line) == 0 or line[0] == "=":
                strands_within_drift = 0
                for strand in cluster:
                    if normal_len - max_drift <= len(strand) <= normal_len + max_drift:
                        strands_within_drift += 1
                #print(strands_within_drift)
                if strands_within_drift < min_traces:
                    count += 1

                cluster = []
                continue
            cluster.append(line)
        except EOFError:
            break
    return count

max_drift = 6
min_traces = 6
normal_len = 232
cluster_count = 30400
for min_traces in range(5, 11):
    for max_drift in range(3, 7):
        count = countFullClusters(max_drift, min_traces, normal_len)
        print(f"{count} out of {cluster_count} ({100*count / cluster_count:.2f}%) for max_drift: {max_drift}, min_traces: {min_traces}, normal_len: {normal_len}")



'''
1460 out of 30400 (4.80%) for max_drift: 3, min_traces: 5, normal_len: 232
1076 out of 30400 (3.54%) for max_drift: 4, min_traces: 5, normal_len: 232
954 out of 30400 (3.14%) for max_drift: 5, min_traces: 5, normal_len: 232
925 out of 30400 (3.04%) for max_drift: 6, min_traces: 5, normal_len: 232

3362 out of 30400 (11.06%) for max_drift: 3, min_traces: 6, normal_len: 232
2762 out of 30400 (9.09%) for max_drift: 4, min_traces: 6, normal_len: 232
2521 out of 30400 (8.29%) for max_drift: 5, min_traces: 6, normal_len: 232
2441 out of 30400 (8.03%) for max_drift: 6, min_traces: 6, normal_len: 232

6174 out of 30400 (20.31%) for max_drift: 3, min_traces: 7, normal_len: 232
5277 out of 30400 (17.36%) for max_drift: 4, min_traces: 7, normal_len: 232
4960 out of 30400 (16.32%) for max_drift: 5, min_traces: 7, normal_len: 232
4846 out of 30400 (15.94%) for max_drift: 6, min_traces: 7, normal_len: 232

8307 out of 30400 (27.33%) for max_drift: 4, min_traces: 8, normal_len: 232
7921 out of 30400 (26.06%) for max_drift: 5, min_traces: 8, normal_len: 232
7799 out of 30400 (25.65%) for max_drift: 6, min_traces: 8, normal_len: 232

12458 out of 30400 (40.98%) for max_drift: 3, min_traces: 9, normal_len: 232
11333 out of 30400 (37.28%) for max_drift: 4, min_traces: 9, normal_len: 232
10909 out of 30400 (35.88%) for max_drift: 5, min_traces: 9, normal_len: 232
10767 out of 30400 (35.42%) for max_drift: 6, min_traces: 9, normal_len: 232

15532 out of 30400 (51.09%) for max_drift: 3, min_traces: 10, normal_len: 232
14393 out of 30400 (47.35%) for max_drift: 4, min_traces: 10, normal_len: 232
13987 out of 30400 (46.01%) for max_drift: 5, min_traces: 10, normal_len: 232
13846 out of 30400 (45.55%) for max_drift: 6, min_traces: 10, normal_len: 232
'''