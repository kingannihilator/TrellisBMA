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
7470 out of 357588 (2.08900%)for max_drift = 4 and normal_len = 232
2605 out of 357588 (0.72849%)for max_drift = 5 and normal_len = 232
891 out of 357588 (0.24917%)for max_drift = 6 and normal_len = 232

'''