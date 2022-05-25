import numpy as np


class ServiceBatchGenerator(object):

    def __init__(self, batch_size, min_service_length, max_service_length, vocab_size):
        self.batch_size = batch_size                  # SFC生成数量（per batch）
        self.min_service_length = min_service_length  # SFC最小长度
        self.max_service_length = max_service_length  # SFC最大长度
        self.vocab_size = vocab_size                  # 词向量大小（context vector）

        # service_length[batch_size] 记录sfc长度
        self.service_length = np.zeros(self.batch_size,  dtype='int32')
        # state[batch_size, max_service_length] 一个batch的sfc --- LSTM输入格式[batch_size, time_step, input_dim]
        self.state = np.zeros((self.batch_size, self.max_service_length),  dtype='int32')

    def getNewState(self):  # 生成一个batch的 state
        # 初始化
        self.state = np.zeros((self.batch_size, self.max_service_length), dtype='int32')
        self.service_length = np.zeros(self.batch_size,  dtype='int32')

        # Compute random services
        for batch in range(self.batch_size):
            self.service_length[batch] = np.random.randint(self.min_service_length, self.max_service_length+1, dtype='int32')
            for i in range(self.service_length[batch]):
                vnf_id = np.random.randint(1, self.vocab_size,  dtype='int32')
                self.state[batch][i] = vnf_id


if __name__ == "__main__":

    # test
    batch_size = 5
    min_service_length = 2
    max_service_length = 6
    vocab_size = 8

    env = ServiceBatchGenerator(batch_size, min_service_length, max_service_length, vocab_size)
    env.getNewState()
    print(env.state)


