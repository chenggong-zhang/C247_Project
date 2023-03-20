import gym
import numpy as np
from gym import spaces
import copy
from talib import *
import pandas as pd
from datetime import datetime


class EnvConfig(object):
    def __init__(self):
        self.train_data = None
        self.eval_data =None
        self.mode = None
        self.use_data_aug = None
        self.min_and_max = None


class MarketEnv(gym.Env):
    environment_name = 'MarketEnv'

    def __init__(self, config):
        '''
        :param config: config should be an object that has the following attributes:
        1)config.train_data
        2)config.eval_data
        3)config.mode
        4)config.use_data_aug
        '''
        self.config = config
        self.id = 'MarketEnv'
        self.reward_threshold = 0.0
        self.trials = 10

        # self.max_episode_steps = self.reward_for_achieving_goal
        self.max_episode_steps = None

        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(29,),
            dtype=np.float32
        )
        # min_and_max = np.load('../environments/Market_Environments/min_and_max.npy', allow_pickle=True).item()
        min_and_max = config.min_and_max

        self.MAX_OBS = np.concatenate((
            10000*5*np.ones(1),
            10000*5*np.ones(1),
            min_and_max['max_obs'],
        ), dtype=np.float32
        )

        self.MIN_OBS = np.concatenate(
            (
            0 * np.ones(1),
            0 * np.ones(1),
            min_and_max['min_obs']
             ), dtype=np.float32
        )

    def normalize_obs(self):
        return (self.obs - self.MIN_OBS) / (self.MAX_OBS - self.MIN_OBS)

    def train(self):
        self.config.mode = 'train'

    def eval(self):
        self.config.mode = 'eval'

    def generate_track(self):

        volume = self.track_data[:, 1]
        open = self.track_data[:, 2]
        high = self.track_data[:, 3]
        low = self.track_data[:, 4]
        close = self.track_data[:, 5]
        # volume = self.track_data[:,6]

        log_return = np.log(close/open)
        log_high = np.log(high/open)
        log_low = np.log(low/open)


        uniswap_feat = np.zeros((self.track_data.shape[0], 23))


        uniswap_feat[:, 0] = DEMA(close, timeperiod=30) / open
        uniswap_feat[:, 1] = SAR(high, low, acceleration=0, maximum=0) / open
        uniswap_feat[:, 2] = ADX(high, low, close, timeperiod=14)
        uniswap_feat[:, 3] = APO(close, fastperiod=12, slowperiod=26, matype=0)
        uniswap_feat[:, 4] = AROONOSC(high, low, timeperiod=14)
        uniswap_feat[:, 5] = BOP(open, high, low, close)
        uniswap_feat[:, 6] = CCI(high, low, close, timeperiod=3)
        uniswap_feat[:, 7] = CCI(high, low, close, timeperiod=10)
        uniswap_feat[:, 8] = CMO(close, timeperiod=14)
        uniswap_feat[:, 9] = DX(high, low, close, timeperiod=14)
        uniswap_feat[:, 10] = MINUS_DM(high, low, timeperiod=14)
        uniswap_feat[:, 11] = MOM(close, timeperiod=3)
        uniswap_feat[:, 12] = MOM(close, timeperiod=5)
        uniswap_feat[:, 13] = MOM(close, timeperiod=10)
        uniswap_feat[:, 14] = PLUS_DM(high, low, timeperiod=14)
        uniswap_feat[:, 15] = TRIX(close, timeperiod=30)
        uniswap_feat[:, 16] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        slowk, slowd = STOCH(high, low, close, fastk_period=5,
                             slowk_period=3, slowk_matype=0,
                             slowd_period=3, slowd_matype=0)
        fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        uniswap_feat[:, 17] = slowd
        uniswap_feat[:, 18] = fastk
        uniswap_feat[:, 19] = NATR(high, low, close, timeperiod=14)
        uniswap_feat[:, 20] = TRANGE(high, low, close)
        # cycle indicator
        uniswap_feat[:, 21] = HT_DCPERIOD(close)
        uniswap_feat[:, 22] = HT_DCPHASE(close)


        self.state_data = np.concatenate((log_return.reshape(-1, 1),
                                          log_high.reshape(-1, 1),
                                          log_low.reshape(-1, 1),
                                          volume.reshape(-1, 1),
                                          uniswap_feat), 1)[88:]

        # self.state_data = np.concatenate((open.reshape(-1, 1),
        #                                   high.reshape(-1, 1),
        #                                   low.reshape(-1, 1),
        #                                   close.reshape(-1, 1),
        #                                   volume.reshape(-1,1),
        #                                   uniswap_feat), 1)[88:]

        self.track_data = self.track_data[88:]

        # max = np.max(self.state_data, 0)
        # min = np.min(self.state_data, 0)
        # min_and_max = {'min_obs':min, 'max_obs':max}
        # np.save('min_and_max.npy', min_and_max)

    def reset(self):
        if self.config.mode == 'train':
            print('environment in train mode')
            step = np.random.choice(self.config.train_data.shape[0]-4000)
            self.tmp_data = self.config.train_data[step:step+4000]
            self.track_data = copy.deepcopy(self.tmp_data)

            if self.config.use_data_aug:
                lam = np.random.uniform(0, 0.99, self.tmp_data.shape[0])
                for i in range(1, self.tmp_data.shape[0]):
                    self.track_data[i, 1:] = (1 - lam[i]) * self.tmp_data[i, 1:] \
                                             + lam[i] * self.track_data[i - 1, 1:]
        elif self.config.mode == 'eval':
            print('environment in evaluate mode')
            self.track_data = copy.deepcopy(self.config.eval_data)
        else:
            raise ValueError("mode of MarketEnv can only be 'train' or 'eval'.")


        print('episode start time:{}, episode end time:{}' .format(datetime.utcfromtimestamp(self.track_data[0,0]).strftime('%Y-%m-%d %H:%M:%S'),
                                                 datetime.utcfromtimestamp(self.track_data[-1,0]).strftime('%Y-%m-%d %H:%M:%S'))
              )

        self.generate_track()

        initial_proportion = np.random.randn(2)
        initial_proportion = np.exp(initial_proportion)/np.sum(np.exp(initial_proportion))
        initial_capital = 10000
        contract_price = self.track_data[0,5]

        self.obs = np.concatenate((initial_capital*initial_proportion, self.state_data[0]), axis=0)
        self.obs[1] = self.obs[1]/contract_price

        self.capital = initial_capital
        self.cur_step = 0

        return self.normalize_obs()

    def step(self, action):


        # self.track_data
        # [volume, open, high, low, close]

        action = np.clip(action, 0, 1)

        contract_price = self.track_data[self.cur_step,5]
        value = self.obs[0] + self.obs[1] * contract_price
        new_value = value*np.array([action[0], 1-action[0]])
        self.obs[0] = new_value[0]
        self.obs[1] = new_value[1]/contract_price

        self.cur_step += 1
        next_contract_price = self.track_data[self.cur_step, 5]
        next_value = self.obs[0] + self.obs[1] * next_contract_price
        self.obs[2:] = self.state_data[self.cur_step]

        obs = self.normalize_obs()
        reward = 100*np.log(next_value / value)
        done = (self.cur_step+1)>=self.track_data.shape[0]
        info = {'position_value':new_value}

        return obs, reward, done, info

if __name__== '__main__':
    import time
    df = pd.read_csv('Bitfinex_ETHUSD_1h.csv')
    data = df.to_numpy()[:, [1, 7, 3, 4, 5, 6]]
    data = np.flip(data, axis=0)
    for i in range(data.shape[0]):
        data[i, 0] = time.mktime(time.strptime(data[i, 0], '%Y/%m/%d %H:%M'))

    data = (np.array(data, dtype=np.float64))
    Envconfig = EnvConfig()
    # 1)config.train_data
    # 2)config.eval_data
    # 3)config.mode
    # 4)config.use_data_aug
    Envconfig.train_data = data[:int(0.8 * data.shape[0])]
    # Envconfig.eval_data = data[:int(0.8*data.shape[0])]
    Envconfig.eval_data = data[int(0.8 * data.shape[0]):int(0.9 * data.shape[0])]
    Envconfig.mode = 'eval'
    Envconfig.use_data_aug = False
    Envconfig.min_and_max = np.load('min_and_max.npy', allow_pickle=True).item()
    # Envconfig.test_data = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]
    env = MarketEnv(Envconfig)
    reward_list = []

    for i in range(100):
        env.eval()
        env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.random.rand() * np.ones(1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # reward_list.append(reward)
        print('cumulative score: {}'.format(total_reward))
        reward_list.append(total_reward)
    np.savetxt('random_policy_results.txt', reward_list)