import os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch, random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from utils.hex_environment import create_hexagons_polygon, located_grid
from utils.util import fcs_loc, fcs_data, POLYGON, station_queue_update
from utils.classes import Fcs, Mcs, Hex, BidirectionalDict
from utils.hex_map import hex_map
from simulator import simulator
from datetime import datetime, timedelta
from loguru import logger
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from utils.static_init_hex import init_hex
from copy import deepcopy


class MobileChargingEnvironment:
    def __init__(self, HEX, FCS, MCS, starttime, endtime, time_step, f_dir, f_name, lambda1, prob_dist, prob_wait, num_actions=19):
        # Enviornment setting
        self.num_actions = num_actions
        # Fixed variable
        self.HEX = HEX
        self.FCS = FCS
        self.starttime = starttime
        self.endtime = endtime
        self.time_step = time_step
        self.lambda1 = lambda1
        self.prob_dist = prob_dist
        self.prob_wait = prob_wait
        # Dynamic variable
        self.MCS = MCS
        self.timestamp = None
        # Useful variable
        self.mcs_hexs = [s.hex for s in MCS]
        self.grid_size = len(self.HEX)
        # Dataset
        self.f_path = f_dir + f_name
        self.df = None
        # self.state_scaler = StandardScaler()


    def modify_mcs_hex(self, hexs):
        if len(hexs) != len(self.MCS):
            raise Exception("Method: modify_mcs_hex() error!")
        else:
            new_hexs = []
            for i, mcs in enumerate(self.MCS):
                new_hex = hexs[i]
                mcs.hex = new_hex
                mcs.location = self.HEX[new_hex].centroid
                new_hexs.append(new_hex)
            self.mcs_hexs = new_hexs
    
    def reset_df(self) -> None:
        df = pd.read_csv(self.f_path)
        df = df[["User ID", "Start DateTime", "Time Difference", "req_lat", "req_lng"]]
        df["Start DateTime"] = pd.to_datetime(df["Start DateTime"])
        df["Time Difference"] = pd.to_timedelta(df["Time Difference"])
        df[["Hex ID","Target Station ID", "Perfect Score", "Traveling time", "Waiting time"]] = None
        self.df = df

    def reset_stations(self) -> None:
        for stations in [self.FCS, self.MCS]:
            for s in stations:
                s.avai_time = [datetime.fromisoformat("2000-01-01")] * s.connector
                s.queue = []

    def reset(self):
        # Reset the environment and return the initial state
        self.timestamp = self.starttime
        self.reset_df()
        self.reset_stations()
        states = []
        # Set the random seed
        sample = init_hex[len(self.MCS)][self.timestamp]
        self.modify_mcs_hex(sample)
        # State: [[demand, num_fcs, cv_f]*19, U_m, U_std_all_f, HoD, DoW]
        # grid info
        for mcs in self.MCS:
            state = []
            mcs_hex = mcs.hex
            neighbors_dic = hex_map[mcs_hex]
            for i in range(19):
                neighbor_hex = neighbors_dic.get(i, None)
                if neighbor_hex:
                    num_fcs = float(len(self.HEX[neighbor_hex].fcs_list))
                    demand, cv_f = 0.0, 1.0
                else:
                    demand, num_fcs, cv_f = 0.0, 0.0, 1.0
                state.extend([demand, num_fcs, cv_f])
            # util
            U_m = 0.0
            U_std_all_f = 0.5
            state.extend([U_m, U_std_all_f])

            # time
            HoD = self.timestamp.hour
            DoW = self.timestamp.weekday()
            state.extend([HoD, DoW])

            states.append(state)
        # # Fit the state scaler
        # self.state_scaler.fit(np.array(states))
        
        # # Scale the states
        # states = self.state_scaler.transform(np.array(states))

        states = np.array(states)
        
        return states

    def step(self, actions):
        # Execute the actions and update the environment
        old_hexs = self.mcs_hexs
        if self.timestamp.hour == 0:
            new_hexs = init_hex[len(self.MCS)][self.timestamp]
        else:
            new_hexs = [hex_map[old_hexs[i]][actions[i]] for i in range(len(self.MCS))]
        self.modify_mcs_hex(new_hexs)

        # Simulation
        starttime, endtime = self.timestamp, self.timestamp+self.time_step
        demands, perfect_scores, utilizations = self.simulation(starttime, endtime)
        self.timestamp += self.time_step

        # Return the next state, rewards, done, and any additional information
        next_states = []  # Next states
        R_m_list = []  # Rewards
        for mcs in self.MCS:
            # Next state: [[demand, num_fcs, cv_f]*19, U_m, U_std_all_f, HoD, DoW]
            next_state = []
            mcs_hex = mcs.hex
            neighbors_dic = hex_map[mcs_hex]
            # grid info
            for i in range(19):
                neighbor_hex = neighbors_dic.get(i, None)
                if neighbor_hex:
                    demand = demands[neighbor_hex]
                    fcs_list = self.HEX[neighbor_hex].fcs_list
                    num_fcs = float(len(fcs_list))
                    U_values = [utilizations[fcs] for fcs in fcs_list] if fcs_list else [0.0]
                    cv_f = np.std(U_values) / (np.mean(U_values) + 0.0001) if np.mean(U_values) != 0.0 else 1.0
                else:
                    demand, num_fcs, cv_f = 0.0, 0.0, 1.0
                next_state.extend([demand, num_fcs, cv_f])

            # util
            U_m = utilizations[mcs.id]
            U_all_f = [utilizations[fcs.id] for fcs in self.FCS]
            U_std_all_f = np.std(U_all_f)
            next_state.extend([U_m, U_std_all_f])

            # time
            HoD = self.timestamp.hour
            DoW = self.timestamp.weekday()
            next_state.extend([HoD, DoW])

            # set current agent next state
            next_states.append(next_state)

            # R_m reward
            fcs_list = self.HEX[mcs_hex].fcs_list
            R_m = 1 / (1 + np.exp(self.lambda1 * perfect_scores[mcs.id])) * utilizations[mcs.id]
            R_m_list.append(R_m)

        # Scale the next states
        # next_states = self.state_scaler.transform(np.array(next_states))
        next_states = np.array(next_states)
        
        # R_f
        f_perfect = np.array(perfect_scores[:len(self.FCS)])
        f_util = np.array(utilizations[:len(self.FCS)])
        R_f = 1 / (1 + np.exp(self.lambda1 * f_perfect)) * f_util
        R_total_np = np.array(R_m_list) + (R_f.sum() / len(self.MCS))
        logger.debug(f"Period: {self.timestamp.strftime('%Y-%m-%d_%H:%M:%S')} -> R_total={R_total_np.sum()}")
        
        # done
        done = self.timestamp == self.endtime

        return next_states, R_total_np, done

    def simulation(self, starttime, endtime):
        # target df
        temp_df = self.df.loc[(self.df["Start DateTime"] >= starttime) & (self.df["Start DateTime"] < endtime)].copy()

        # create empty lists to store the results
        req_hex_ids = []
        target_station_ids = []
        target_station_ps = []
        target_station_tt = []
        target_station_wt = []

        # stations
        stations = self.FCS + self.MCS
        last_stations = deepcopy(stations)

        # iterate over each row of the DataFrame
        for index, row in temp_df.iterrows():
            # apply the simulator function to the current row and store the results
            stations, target_station_id, travel_time, wait_time, perfect_score = simulator(row, stations, self.prob_dist, self.prob_wait, 0.0)
            req_hex_ids.append(located_grid(location=[row["req_lat"], row["req_lng"]], hexagons=self.HEX))
            target_station_ids.append(target_station_id)
            target_station_ps.append(perfect_score)
            target_station_tt.append(travel_time)
            target_station_wt.append(wait_time)
        
        # update FCS, MCS
        self.FCS = stations[:len(self.FCS)]
        self.MCS = stations[len(self.FCS):]

        # add the results to the DataFrame
        temp_df["Hex ID"] = req_hex_ids
        temp_df["Target Station ID"] = target_station_ids
        temp_df["Perfect Score"] = target_station_ps
        temp_df["Traveling time"] = target_station_tt
        temp_df["Waiting time"] = target_station_wt

        # next state inputs: demand of hex
        demands = []
        for i in range(len(self.HEX)):
            demands.append(len(temp_df[temp_df["Hex ID"]==i]))

        # next state inputs, rewards: perfect scores, utilizations
        perfect_scores = []
        utilizations = []
        for station in last_stations:
            time_in_charge = timedelta(seconds=0)
            station.queue, station.avai_time = station_queue_update(station, starttime)
            station_avai_time = np.array(station.avai_time)
            for req_list in station.queue:
                if all([c > endtime for c in station_avai_time]):
                    time_in_charge = self.time_step * len(station_avai_time)
                    break
                char_id = np.argmin(station_avai_time)
                station_avai_time[char_id] += req_list[1]
            for char in station_avai_time:
                if char < starttime:
                    continue
                time_in_charge += char-starttime if char < endtime else self.time_step
            util = time_in_charge / (station.connector * self.time_step)
            # utilization
            utilizations.append(round(util,2) if util <= 1.0 else 1.0)
        del last_stations

        for i in range(len(stations)):
            # get target demands
            dt = temp_df[temp_df["Target Station ID"] == i].copy()
            # perfect scores
            perfect_scores.append(dt["Perfect Score"].sum())
            del dt
        
        self.df.update(temp_df)
        return demands, perfect_scores, utilizations

# Define the VDN class
class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()

    # `forward` is called when the network is evaluated on an input
    def forward(self, q_values):
        return torch.sum(q_values, dim=1)

# Define the QNetwork class
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # three fully-connected layers (input, output)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    # the state passes through two fully-connected layers with relu, and returns the output of the final layer
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Agent class
class Agent:
    def __init__(self, shared_q_network, lr):
        self.q_network = shared_q_network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.hex = None

    # select action with epsilon-greedy strategy
    def select_action(self, state, epsilon):
        valid_actions = list(hex_map[self.hex].keys())
        q_values = self.q_network(state)
        if torch.rand(1).item() < epsilon:
            # return a random action from the valid actions
            return random.choice(valid_actions)
        else:
            # Create a mask of the same shape as q_values, with all elements set to False
            mask = torch.full_like(q_values, fill_value=False, dtype=torch.bool)

            # Get valid actions as a tensor
            valid_actions_tensor = torch.tensor(valid_actions).to(device)

            # Set the elements corresponding to valid_actions to True
            mask[valid_actions_tensor] = True

            # Set the Q-values of invalid actions to -inf
            q_values[~mask] = float('-inf')

            # Get the index of the action with the highest Q-value
            best_action_index = torch.argmax(q_values)

            return best_action_index.item()  # assuming you want to return the index as a Python number

def eval_df(df, num_hex, stations, starttime, endtime):
    df = df.loc[(df["Start DateTime"] >= starttime) & (df["Start DateTime"] < endtime)].copy()
    # next state inputs: demand of hex
    demands = []
    for i in range(num_hex):
        demands.append(len(df[df["Hex ID"]==i]))

    # next state inputs, rewards: perfect scores, utilizations
    perfect_scores = []
    utilizations = []
    for i in range(len(stations)):
        # get target demands
        dt = df[df["Target Station ID"] == i].copy()
        # perfect scores
        perfect_scores.append(dt["Perfect Score"].sum())
        # utilization
        total_avai_time = stations[i].connector * (endtime - starttime)
        total_time_diff = dt["Time Difference"].sum()
        utilization = total_time_diff / total_avai_time
        utilizations.append(round(utilization,2) if utilization <= 1.0 else 1.0)
        del dt
    
    return demands, perfect_scores, utilizations

def eval_action_step(actions):
    steps = 0
    for action in actions:
        if action == 0:
            steps += 0
        elif action in range(1,8):
            steps += 1
        elif action in range(8,19):
            steps += 2
    return steps

def save_action(mcs_start_hex, mcs_actions, s_quote):
    dic = dict()
    for i in range(len(mcs_start_hex)):
        dic[i] = {
            "start_hex": mcs_start_hex[i],
            "actions": mcs_actions[i]
        }
    s_ver = 0
    while True:
        if os.path.isfile(f"{s_quote}_{s_ver}.txt"):
            s_ver += 1
        else:
            with open(f"{s_quote}_{s_ver}.txt", 'w') as file:
                file.write(f"{dic}")
            break

def train_vdn(agents, env, s_qoute, episodes, gamma, epsilon_start, epsilon_end, epsilon_decay):
    vdn = VDN()
    epsilon = epsilon_start
    episode_rewards = []
    episode_steps = []
    episode_demands = []
    episode_perfect_scores = []
    episode_utilizations = []

    for ep in range(episodes):
        states = torch.from_numpy(env.reset()).float().to(device)
        done = False
        total_rewards = 0
        steps = 0
        # debug
        mcs_start_hex = env.mcs_hexs
        mcs_actions = [[] for _ in range(len(env.MCS))]
        while not done:
            # 1. Each agent select a action
            actions = []
            for idx, agent in enumerate(agents):
                agent.hex = env.mcs_hexs[idx]
                action = agent.select_action(states[idx], epsilon)
                actions.append(action)

            # 2. Go to next state: next_state_np
            # 3. Get reward: rewards_np
            for i in range(len(env.MCS)):
                mcs_actions[i].append(actions[i])
            next_state_np, R_total_np, done = env.step(actions)

            next_states = torch.from_numpy(next_state_np).float().to(device)
            rewards = torch.from_numpy(R_total_np).float().unsqueeze(1).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float).to(device)

            q_values = torch.stack([agent.q_network(states[idx]) for idx, agent in enumerate(agents)], dim=1)
            next_q_values = torch.stack([agent.q_network(next_states[idx]) for idx, agent in enumerate(agents)], dim=1)
            vdn_next_q_values = vdn(next_q_values)
            target_q_values = rewards + gamma * vdn_next_q_values * (1 - done_tensor)
            q_value_loss = (vdn(q_values) - target_q_values).pow(2).mean()

            for agent in agents:
                agent.optimizer.zero_grad()
            q_value_loss.backward()
            for agent in agents:
                agent.optimizer.step()

            states = next_states
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

            # Update the reward function calculation according to your scenario
            total_rewards += rewards.sum()
            steps += eval_action_step(actions)

        episode_rewards.append(total_rewards)
        episode_steps.append(steps)
        demands, perfect_scores, utilizations = eval_df(df=env.df, num_hex=len(env.HEX), stations=env.FCS+env.MCS, starttime=env.starttime, endtime=env.endtime)
        episode_demands.append(sum(demands))
        episode_perfect_scores.append(sum(perfect_scores))
        episode_utilizations.append(np.array(utilizations).mean().round(4))
        logger.debug(f"{mcs_start_hex=}")
        logger.debug(f"{mcs_actions=}")
        logger.debug(f"Episode {ep}, Reward: {episode_rewards[-1]}, Steps: {episode_steps[-1]}")

    env_df = env.df.loc[(env.df["Start DateTime"] >= starttime) & (env.df["Start DateTime"] < endtime)].copy()
    # After training
    torch.save(agent.q_network.state_dict(), f'trained_model_{s_quote}.pth')
    # save mcs action of last episode
    # save_action(mcs_start_hex, mcs_actions, s_quote)

    # turn tensor from gpu to cpu to np
    for rt in [episode_rewards, episode_steps, episode_demands, episode_perfect_scores, episode_utilizations]:
        for i in range(len(rt)):
            try:
                rt[i] = rt[i].cpu().numpy()
            except:
                pass

    return episode_rewards, episode_steps, episode_demands, episode_perfect_scores, episode_utilizations


if __name__ == "__main__":
    # Initialize
    if len(sys.argv) > 3:
        _n = int(sys.argv[1])
        _epsilon_decay = float(sys.argv[2])
        _lr = float(sys.argv[3])
    else:
        _n = 20  # Default value
        _epsilon_decay = 0.995
        _lr = 0.001
    lambda1 = 0.2
    logger.debug(f"lambda1={lambda1},_n={_n},_epsilon_decay={_epsilon_decay},_lr={_lr}")

    # Set the parameters for the environment and the VDN agents
    N = num_agents = _n
    K = num_actions = 19
    HEXG = grid_size = 121  # = len(HEX)
    lr = _lr
    prob_dist, prob_wait = 0.85, 0.15

    starttime = datetime(2019,1,1)
    endtime = datetime(2019,2,1)
    period = "2w"
    time_step = timedelta(hours=1)
    episodes, gamma, epsilon_start, epsilon_end, epsilon_decay=100, 0.99, 1.0, 0.1, _epsilon_decay
    f_dir = "./dataset/preprocess_data/"
    f_name = "Palo_Alto_2019_preprocess.csv"
    algo = "vdn-static_141-1"

    s_quote = f"{algo}_{period}_lr{lr}_epi{episodes}_prob{prob_dist}_lb{lambda1}_epsilon_decay{epsilon_decay}-{N}"


    fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    logger.add(f"./log/{algo}/{s_quote}_{datetime.now().strftime('%Y%m-%d_%H:%M:%S')}.log", format=fmt, rotation="10 MB", enqueue=True, backtrace=True)
    logger.info("Function called")# logger config

    # Set the device to use (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The size of state and action
    state_size = 3 * num_actions + 2 + 2
    action_size = num_actions

    # HEX
    resolution = 8
    hexagons = create_hexagons_polygon(POLYGON, resolution)
    HEX = [Hex(id=idx, polygon=p) for idx,p in enumerate(hexagons)]

    # FCS(id=0~17)
    fcs_hex_bidic = BidirectionalDict()
    for idx, fcs in enumerate(fcs_loc):
        fcs_hex_bidic[idx] = located_grid(fcs,HEX)
    FCS = [Fcs(id=idx, connector=fcs_data[idx][3], location=fcs_loc[idx], hex=hex_id) for idx, hex_id in fcs_hex_bidic.items()]

    # Update HEX
    for hex_id, fcs_list in fcs_hex_bidic.inverse.items():
        HEX[hex_id].fcs_list = fcs_list

    # MCS(ID=18~mcs+18-1)
    MCS = [Mcs(id=idx+18, connector=2, hex=0) for idx in range(num_agents)]

    # Create the VDN agents and env
    shared_q_network = QNetwork(state_size, action_size).to(device)
    # Load the weights from the saved model
    if os.path.isfile(f'trained_model_{s_quote}.pth'):
        logger.debug("Loading model...")
        raise Exception("Model is not supposed to be loaded")
        # shared_q_network.load_state_dict(torch.load(f'trained_model_{s_quote}.pth'))

    agents = [Agent(shared_q_network, lr) for _ in range(num_agents)]
    env = MobileChargingEnvironment(HEX, FCS, MCS, starttime, endtime, time_step, f_dir, f_name, lambda1, prob_dist, prob_wait, num_actions=19)

    # Train the VDN
    episode_rewards, episode_steps, episode_demands, episode_perfect_scores, episode_utilizations = train_vdn(agents, env, s_quote, episodes, gamma, epsilon_start, epsilon_end, epsilon_decay)

    # Plot the episode rewards and steps
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # First subplot: Episode Rewards
    axs[0, 0].plot(episode_rewards)
    axs[0, 0].set_title("Episode Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")

    # Second subplot: Episode Steps
    axs[0, 1].plot(episode_steps)
    axs[0, 1].set_title("Episode Steps")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Number of Steps")

    # Third subplot: Additional data
    axs[1, 0].plot(episode_perfect_scores)
    axs[1, 0].set_title("Episode Perfect Scores")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Total Perfect Scores")

    # Fourth subplot: More data
    axs[1, 1].plot(episode_utilizations)
    axs[1, 1].set_title("Episode Utilizations")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Avg Utilization")

    # Save the figure
    s_ver = 0
    while True:
        if os.path.isfile(f"{s_quote}_{s_ver}.png"):
            s_ver += 1
        else:
            plt.savefig(f'{s_quote}_{s_ver}.png')
            break
