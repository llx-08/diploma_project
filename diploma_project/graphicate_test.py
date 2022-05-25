import matplotlib.pyplot as plt
import numpy as np
import csv
import sys, getopt

plt.rcParams['font.sans-serif'] = ['simhei']  # 添加中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False


def test_result():
    # Input test file
    fileList = ['save/l12b_test.csv', 'save/l14b_test.csv', 'save/l16b_test.csv', 'save/l18b_test.csv',
                'save/ll20b_test.csv', 'save/ll24b_test.csv', 'save/ll28b_test.csv', 'save/ll30b_test.csv']

    final_reward = []
    final_penalty = []
    final_occ_penalty = []
    final_bandwidth_penalty = []
    final_latency_penalty = []
    final_gaReward = []
    final_gaPenalty = []
    final_ffReward = []
    final_rl_err = []
    final_ga_err = []
    final_ff_err = []

    rl_solve_num = []
    ff_solve_num = []

    # 从csv中恢复变量
    for file in fileList:

        reward = []
        penalty = []
        occ_penalty = []
        bandwidth_penalty = []
        latency_penalty = []
        gaReward = []
        gaPenalty = []
        ga_occ_penalty = []
        ffReward = []
        rl_err = []
        ga_err = []
        ff_err = 0
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) == 0:
                    continue

                tmp = row[3].split()
                reward.append(float(tmp[1]))
                tmp = row[4].split()
                penalty.append(float(tmp[1]))
                tmp = row[5].split()
                occ_penalty.append(float(tmp[1]))
                tmp = row[6].split()
                bandwidth_penalty.append(float(tmp[1]))
                tmp = row[7].split()
                latency_penalty.append(float(tmp[1]))

                tmp = row[9].split()
                gaReward.append(float(tmp[1]))
                tmp = row[10].split()
                gaPenalty.append(float(tmp[1]))
                tmp = row[11].split()
                ga_occ_penalty.append(float(tmp[1]))

                tmp = row[15].split()
                print(tmp)
                if float(tmp[1]) != 0.0:
                    ffReward.append(float(tmp[1]))
                else:
                    ff_err += 1

        final_reward.append(sum(reward) / (len(reward) + 1))
        final_penalty.append(sum(penalty) / (len(penalty) + 1))
        final_gaReward.append(sum(gaReward) / (len(gaReward) + 1))
        final_gaPenalty.append(sum(gaPenalty) / (len(gaPenalty) + 1))
        final_ffReward.append(sum(ffReward) / (len(ffReward) + 1))
        final_occ_penalty.append(occ_penalty)
        final_bandwidth_penalty.append(bandwidth_penalty)
        final_latency_penalty.append(latency_penalty)

        rl_solve_num.append(len(reward))
        ff_solve_num.append(len(ffReward))

        tmp = 0
        for o in occ_penalty:
            if o > 0.0:
                tmp += 1
        final_rl_err.append(tmp / len(reward))

        tmp = 0
        for o in ga_occ_penalty:
            if o > 0.0:
                tmp += 1
        final_ga_err.append((tmp / len(gaReward)))
        final_ff_err.append((ff_err / (len(ffReward) + ff_err)))

    plot_rl_cons_ratio(final_occ_penalty, final_bandwidth_penalty, final_latency_penalty)
    plot_err_ratio(final_rl_err, final_ga_err, final_ff_err)
    fig, ax = plt.subplots(2, 1)

    print(final_gaReward)
    sfc_length = ['12', '14', '16', '18', '20', '24', '28', '30']

    ax[0].plot(final_reward, label='强化学习方法产生能耗')
    ax[0].plot(final_gaReward, label='遗传算法产生能耗')
    ax[0].plot(final_ffReward, label='First-Fit算法产生能耗')

    ax[0].legend()
    ax[0].set(ylabel='能耗', title='测试结果')
    ax[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax[0].set_xticklabels(sfc_length)
    ax[0].grid()

    ax[1].plot(final_penalty, label='强化学习方法惩罚项')
    ax[1].plot(final_gaPenalty, label='遗传算法惩罚项')
    # ax[1].plot(final_ffPenalty, label='FF energy')

    ax[1].legend()
    ax[1].set(ylabel='惩罚项')
    ax[1].grid()
    ax[1].set(xlabel='服务功能链长度')
    ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax[1].set_xticklabels(sfc_length)
    plt.show()

    target_eva      = []
    cost_eva        = []
    penalty_eva     = []
    solve_num_eva   = []
    print("target evaluation:")
    for i in range(len(final_reward)):
        target_eva.append((final_gaReward[i]+final_gaPenalty[i]-final_reward[i]-final_penalty[i])/(final_gaReward[i]+final_gaPenalty[i]))
    print(target_eva)
    plt.plot(target_eva)
    plt.grid()
    plt.xticks(range(len(sfc_length)), sfc_length)
    plt.xlabel('服务功能链长度')
    plt.ylabel('目标值优化')
    plt.show()

    print("cost evaluation:")
    for i in range(len(final_reward)):
        cost_eva.append((final_gaReward[i]-final_reward[i])/final_gaReward[i])
    print(cost_eva)
    plt.plot(cost_eva)
    plt.grid()
    plt.xticks(range(len(sfc_length)), sfc_length)
    plt.xlabel('服务功能链长度')
    plt.ylabel('能耗优化')
    plt.show()

    print("penalty evaluation:")
    for i in range(len(final_reward)):
        penalty_eva.append((final_gaPenalty[i]-final_penalty[i])/final_gaPenalty[i])
    print(penalty_eva)
    plt.plot(penalty_eva)
    plt.grid()
    plt.xticks(range(len(sfc_length)), sfc_length)
    plt.xlabel('服务功能链长度')
    plt.ylabel('惩罚项优化')
    plt.show()

    print("solve num evaluation:")
    print(rl_solve_num)
    print(ff_solve_num)
    for i in range(len(final_reward)):
        solve_num_eva.append((rl_solve_num[i]-ff_solve_num[i])/ff_solve_num[i])
    print(solve_num_eva)



    # print("No Error agent: {}".format(np.count_nonzero(penalty)))
    # print("No Error solver: {}".format(len(sReward) - np.count_nonzero(sReward)))
    # print("No Error heuristic: {}".format(np.count_nonzero(hPenalty)))


# 画三种方法产生错误的比例，柱状图
def plot_err_ratio(rl_err, ga_err, ff_err):

    sfc_length = [12, 14, 16, 18, 20, 24, 28, 30]
    x_ = list(range(len(sfc_length)))
    total_width, n = 0.8, 3
    width = total_width / n
    plt.barh(x_, rl_err, height=width, label="RL Error Ratio", fc='r')
    for i in range(len(x_)):
        x_[i] = x_[i] + width
    plt.barh(x_, ga_err, height=width, label="GA Error Ratio", fc='b')
    for i in range(len(x_)):
        x_[i] = x_[i] + width
    plt.barh(x_, ff_err, height=width, label="FF Error Ratio", tick_label=sfc_length, fc='y')

    plt.legend()
    plt.show()


# 模型评估
def plot_rl_cons_ratio(occ_err, bw_err, lat_err):
    occ_times = []
    band_times = []
    lat_times = []

    occ_value = []
    band_value = []
    lat_value = []
    for o in occ_err:
        tmp = 0
        for o_ in o:
            if o_ > 0.0:
                tmp += 1
        occ_times.append(float(tmp / len(o)))
        occ_value.append(sum(o))
    for b in bw_err:
        tmp = 0
        for o_ in b:
            if o_ > 0.0:
                tmp += 1
        band_times.append(float(tmp / len(b)))
        band_value.append(sum(b))
    for a in lat_err:
        tmp = 0
        for o_ in a:
            if o_ > 0.0:
                tmp += 1
        lat_times.append(float(tmp / len(a)))
        lat_value.append(sum(a))

    sfc_length = [12, 14, 16, 18, 20, 24, 28, 30]
    x_ = list(range(len(sfc_length)))
    total_width, n = 0.8, 3
    width = total_width / n
    plt.bar(x_, occ_times, width=width, label="CPU资源违反率", fc='r')
    for i in range(len(x_)):
        x_[i] = x_[i] + width
    plt.bar(x_, band_times, width=width, label="带宽限制违反率", fc='b')
    for i in range(len(x_)):
        x_[i] = x_[i] + width
    plt.bar(x_, lat_times, width=width, label="时延限制违反率", tick_label=sfc_length, fc='g')
    plt.ylabel("限制违反率")
    plt.xlabel("服务功能链长度")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_result()

# 在结果分析里面新增两节：模型评估，算法对比
