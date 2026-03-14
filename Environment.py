import numpy as np
import random
import math
import os
import sys
import traci
from sumolib import checkBinary

# --- SUMO 环境配置 ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请先声明环境变量 'SUMO_HOME'")

def generate_sumo_files():
    """生成十字路口路网文件"""
    # 1. 节点文件 (中心点 250,250 对应基站)
    with open("cross_500.nod.xml", "w") as f:
        f.write('<nodes>\n')
        f.write('  <node id="center" x="250" y="250" type="traffic_light"/>\n')
        f.write('  <node id="N" x="250" y="500"/>\n')
        f.write('  <node id="S" x="250" y="0"/>\n')
        f.write('  <node id="E" x="500" y="250"/>\n')
        f.write('  <node id="W" x="0" y="250"/>\n')
        f.write('</nodes>\n')

    # 2. 边文件 (限速 8.3m/s)
    with open("cross_500.edg.xml", "w") as f:
        f.write('<edges>\n')
        f.write('  <edge id="N2C" from="N" to="center" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="C2S" from="center" to="S" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="S2C" from="S" to="center" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="C2N" from="center" to="N" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="W2C" from="W" to="center" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="C2E" from="center" to="E" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="E2C" from="E" to="center" numLanes="1" speed="8.3"/>\n')
        f.write('  <edge id="C2W" from="center" to="W" numLanes="1" speed="8.3"/>\n')
        f.write('</edges>\n')

    # 3. 路由文件 (IDM 模型参数)
    with open("cross_500.rou.xml", "w") as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" carFollowModel="IDM" maxSpeed="8.3" accel="3.0" decel="4.5" length="5.0" minGap="2.5" tau="1.0"/>\n')
        routes = [
            ("N2C", "C2S"), ("N2C", "C2E"), ("N2C", "C2W"),
            ("S2C", "C2N"), ("S2C", "C2W"), ("S2C", "C2E"),
            ("E2C", "C2W"), ("E2C", "C2S"), ("E2C", "C2N"),
            ("W2C", "C2E"), ("W2C", "C2N"), ("W2C", "C2S")
        ]
        for i, (start, end) in enumerate(routes):
            f.write(f'  <route id="route_{i}" edges="{start} {end}"/>\n')
        f.write('</routes>\n')

    netconvert_binary = checkBinary('netconvert')
    os.system(f'"{netconvert_binary}" --node-files cross_500.nod.xml --edge-files cross_500.edg.xml --output-file cross_500.net.xml --tls.green.time 30 --tls.yellow.time 4 --no-internal-links')

    # 4. 配置文件
    with open("cross_500.sumocfg", "w") as f:
        f.write('<configuration>\n')
        f.write('  <input><net-file value="cross_500.net.xml"/><route-files value="cross_500.rou.xml"/></input>\n')
        f.write('  <time><begin value="0"/><end value="10000"/><step-length value="0.1"/></time>\n')
        f.write('</configuration>\n')

class V2Vchannels:
    def __init__(self, n_Veh, shadow_std=3):
        self.fc = 2 
        self.h_bs, self.h_ms = 1.5, 1.5
        self.decorrelation_distance = 10
        self.shadow_std = shadow_std
        self.Shadow = np.random.normal(0, self.shadow_std, size=(n_Veh, n_Veh))
        self.PathLoss = np.zeros((n_Veh, n_Veh))
        self.positions = []

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        n = len(self.positions)
        self.PathLoss = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d1 = abs(self.positions[i][0] - self.positions[j][0])
                d2 = abs(self.positions[i][1] - self.positions[j][1])
                d = math.hypot(d1, d2) + 0.001
                # 简化版 V2V 损耗
                if d <= 3:
                    self.PathLoss[i,j] = 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    self.PathLoss[i,j] = 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)

    def update_shadow(self, delta_distance_list):
        n = len(delta_distance_list)
        dist_vec = np.asarray(delta_distance_list).reshape(-1, 1)
        rho = np.exp(-1 * ((dist_vec + dist_vec.T) / self.decorrelation_distance))
        noise = np.random.normal(0, self.shadow_std, size=(n, n))
        self.Shadow = rho * self.Shadow + np.sqrt(1 - rho**2) * noise

class V2Ichannels:
    def __init__(self, n_Veh):
        self.h_bs, self.h_ms = 25, 1.5
        self.bs_position = [250, 250]
        self.shadow_std = 8
        self.decorrelation_distance = 50
        self.Shadow = np.random.normal(0, self.shadow_std, n_Veh)
        self.PathLoss = np.zeros(n_Veh)
        self.positions = []

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        for i, pos in enumerate(self.positions):
            distance = math.hypot(pos[0]-self.bs_position[0], pos[1]-self.bs_position[1]) + 0.1
            self.PathLoss[i] = 128.1 + 37.6 * np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000)

    def update_shadow(self, delta_dist):
        rho = np.exp(-1 * (np.asarray(delta_dist) / self.decorrelation_distance))
        noise = np.random.normal(0, self.shadow_std, len(delta_dist))
        self.Shadow = rho * self.Shadow + np.sqrt(1 - rho**2) * noise

class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh=40):
        generate_sumo_files()
        self.n_Veh = n_veh
        self.timestep = 0.1 # 与 sumocfg 保持一致
        self.veh_counter = 0
        self.agents_sumo_id = [None] * n_veh
        self.agent_positions = [ [0.0, 0.0] for _ in range(n_veh) ]
        
        sumo_binary = checkBinary('sumo') # 训练建议用 sumo，调试用 sumo-gui
        traci.start([sumo_binary, "-c", "cross_500.sumocfg", "--step-length", str(self.timestep), "--no-warnings"])
        
        self.V2Vchannels = V2Vchannels(self.n_Veh)
        self.V2Ichannels = V2Ichannels(self.n_Veh)

    def _spawn_vehicle(self, agent_idx):
        """为特定 Agent 槽位生成新车"""
        veh_id = f"v_{self.veh_counter}"
        self.veh_counter += 1
        route_id = f"route_{random.randint(0, 11)}"
        # 初始位置设在边缘，避免重叠
        traci.vehicle.add(veh_id, route_id, typeID="car", departSpeed="8.3")
        self.agents_sumo_id[agent_idx] = veh_id
        return veh_id

    def add_new_vehicles_by_number(self, n):
        """初始填充车辆"""
        for i in range(self.n_Veh):
            self._spawn_vehicle(i)
        # 热身：让车跑进路网，否则初始 Reward 会因为车都在路口外而异常
        for _ in range(50):
            traci.simulationStep()

    def renew_positions(self):
        traci.simulationStep()

    def renew_channels(self):
        active_ids = traci.vehicle.getIDList()
        current_positions = []
        delta_dists = []

        for i in range(self.n_Veh):
            vid = self.agents_sumo_id[i]
            # 如果车跑出了路网，立刻重生
            if vid not in active_ids:
                vid = self._spawn_vehicle(i)
                # 给一个路口边缘的默认位置，防止计算 Pathloss 时距离为 0
                pos = (0.0, 250.0) if "W2C" in traci.vehicle.getRoute(vid)[0] else (250.0, 0.0)
                speed = 8.3
            else:
                pos = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
            
            self.agent_positions[i] = list(pos)
            current_positions.append(pos)
            delta_dists.append(speed * self.timestep)

        # 更新信道
        self.V2Ichannels.update_positions(current_positions)
        self.V2Vchannels.update_positions(current_positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        self.V2Ichannels.update_shadow(delta_dists)
        self.V2Vchannels.update_shadow(delta_dists)

        # 计算增益 (dB -> Linear)
        v2i_gain = 10**((-self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow)/10) * np.random.exponential(1.0, self.n_Veh)
        v2v_gain = 10**((-self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow)/10) * np.random.exponential(1.0, (self.n_Veh, self.n_Veh))

        return v2v_gain, v2i_gain

    def close(self):
        traci.close()
