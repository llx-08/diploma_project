import matplotlib.pyplot as plt
import csv
import sys, getopt

plt.rcParams['font.sans-serif'] = ['simhei']  # 添加中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False


def main(argv):
    print(sys.argv[0])
    print(sys.argv[1])

    inputfile = ''

    try:
        opts, args = getopt.getopt(argv,"f:",["file="])
    except getopt.GetoptError:
        print("test.py -f --file <file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            inputfile = arg

    reward = []
    baseline = []
    advantage = []
    penalty = []
    loss_agent = []
    target = []

    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 0:
                continue
            tmp = row[3].split()
            reward.append(float(tmp[1]))
            tmp = row[4].split()
            target.append(float(tmp[1]))
            tmp = row[5].split()
            baseline.append(float(tmp[1]))
            tmp = row[6].split()
            advantage.append(float(tmp[1]))
            tmp = row[7].split()
            penalty.append(float(tmp[1]))
            tmp = row[8].split()
            loss_agent.append(float(tmp[1]))
    print(loss_agent)
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(reward, label='能耗')
    ax[0].plot(baseline, label='基线')
    ax[0].plot(penalty, label='惩罚项')
    ax[0].plot(target, label='target')
    ax[0].legend()
    ax[0].set(ylabel='能耗', title='学习过程')
    ax[0].grid()

    ax[1].plot(loss_agent, label='Agent损失')
    ax[1].grid()
    ax[1].set(xlabel='迭代次数(x100)', ylabel='Agent损失')

    plt.show()


if __name__ == "__main__":

    main(sys.argv[1:])






