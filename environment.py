import gym
import numpy as np
from gym import spaces
import copy
from talib import *
import pandas as pd

class EnvConfig(object):
    def __init__(self):
        self.train_data = None
        self.eval_data =None
        self.mode = None
        self.use_data_aug = None


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
        self._max_episode_steps = None

        self.action_space = spaces.Box(
            low=0, high=1, shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(29,),
            dtype=np.float32
        )
        min_and_max = np.load('min_and_max.npy', allow_pickle=True).item()

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
        self.track_data = self.track_data[88:]

        max = np.max(self.state_data, 0)
        min = np.min(self.state_data, 0)
        min_and_max = {'min_obs':min, 'max_obs':max}
        np.save('min_and_max.npy', min_and_max)

    def reset(self):
        if self.config.mode == 'train':

            self.track_data = copy.deepcopy(self.config.train_data)
            if self.config.use_data_aug:
                lam = np.random.uniform(0, 0.99, self.config.train_data.shape[0])
                for i in range(1, self.config.train_data.shape[0]):
                    self.track_data[i, 1:] = (1 - lam[i]) * self.config.train_data[i, 1:] \
                                             + lam[i] * self.config.train_data[i - 1, 1:]
        elif self.config.mode == 'eval':
            self.track_data = copy.deepcopy(self.config.eval_data)
        else:
            raise ValueError("mode of MarketEnv can only be 'train' or 'eval'.")

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
        contract_price = self.track_data[self.cur_step,5]
        value = self.obs[0] + self.obs[1] * contract_price
        new_value = value*action
        self.obs[0] = new_value[0]
        self.obs[1] = new_value[1]/contract_price

        self.cur_step += 1
        next_contract_price = self.track_data[self.cur_step, 5]
        next_value = self.obs[0] + self.obs[1] * next_contract_price
        self.obs[2:] = self.state_data[self.cur_step]

        obs = self.normalize_obs()
        reward = 100*np.log(next_value / value)
        done = self.cur_step>=self.track_data.shape[0]
        info = {}

        return obs, reward, done, info

if __name__ == '__main__':
    df = pd.read_csv('Bitfinex_ETHUSD_1h.csv')
    data = df.to_numpy()[:,[0,7,3,4,5,6]]
    data = np.array(data,dtype=np.float64)
    config = EnvConfig()
    config.train_data = data[:int(0.8*data.shape[0])]
    config.eval_data = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]
    config.use_data_aug = True
    config.mode = 'eval'
    env = MarketEnv(config)
    env.reset()
    action = np.array([0.4,0.6])
    obs, reward, done, info=env.step(action)
    # config.test_data = data[int(0.8*data.shape[0]):int(0.9*data.shape[0])]


