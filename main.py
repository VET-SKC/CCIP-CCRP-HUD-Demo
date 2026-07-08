from ursina import *
from panda3d.core import Quat  # 导入四元数库
import math
import numpy as np
from PIL import Image


FEET_TO_METERS = 0.3048
METERS_TO_FEET = 3.28084
MS_TO_KNOTS = 1.94384
KNOTS_TO_KPH = 1.852
POUND_TO_KG = 0.454
G = 9.80665


BOMBS = {
    'MK81_LD': {
        'name': 'MK81 LD',
        'weight': 250 * POUND_TO_KG,
        'drag_coefficient': 0.05,
        'ejection_velocity': Vec3(0, -2, 0)  # 相对飞机向下抛射的速度 (m/s)
    },
    'MK82_LD': {
        'name': 'MK82 LD',
        'weight': 500 * POUND_TO_KG,
        'drag_coefficient': 0.05,
        'ejection_velocity': Vec3(0, -2, 0)
    },
    'MK83_LD': {
        'name': 'MK83 LD',
        'weight': 1000 * POUND_TO_KG,
        'drag_coefficient': 0.05,
        'ejection_velocity': Vec3(0, -2, 0)
    },
    'MK84_LD': {
        'name': 'MK84 LD',
        'weight': 2000 * POUND_TO_KG,
        'drag_coefficient': 0.05,
        'ejection_velocity': Vec3(0, -2, 0)
    },
    'MK82_HD': {
        'name': 'MK82 HD',
        'weight': 530 * POUND_TO_KG,  # 高阻尾翼会增加一些重量
        'drag_coefficient': 0.8,  # 阻力系数显著增大
        'ejection_velocity': Vec3(0, -2, -1)
    }
}
BOMB_TYPES = list(BOMBS.keys())
BOMB_PRESET = {
    1: {'MK81_LD': 2, 'MK82_LD': 2, 'MK83_LD': 2, 'MK84_LD': 1, 'MK82_HD': 2},  # type -> count
    2: {'MK81_LD': 8, 'MK82_LD': 2, 'MK83_LD': 4, 'MK84_LD': 4, 'MK82_HD': 8},
    99: {'MK81_LD': 10, 'MK82_LD': 10, 'MK83_LD': 10, 'MK84_LD': 10, 'MK82_HD': 10}
}


class FlightController(Entity):
    def __init__(self, bomb_preset=1, **kwargs):
        super().__init__(**kwargs)

        # --- 飞行性能参数 ---
        self.speed = 100  # 飞机的巡航速度 (m/s)
        self.base_pitch_speed = 30  # 俯仰速率 (度/秒)
        self.pitch_speed = 30
        self.base_roll_speed = 195  # 滚转速率 (度/秒)
        self.roll_speed = 195
        self.base_aoa_recovery = 2.1  # 用于让速度矢量平滑地对准机头方向，提供惯性
        self.aoa_recovery = 2.1
        self.yaw_from_roll = 0.098  # 从滚转自动产生的偏航速率 (用线性近似的协调转弯) (=g/v)

        # --- 物理模型参数 ---
        self.max_thrust = 60000  # 最大推力 (N) 60 kN
        self.drag_base_coefficient = 3  # 基础阻力系数 (高速时产生)
        self.base_drag_induced_coefficient = 3.75  # 诱导阻力系数 (拉杆/高G时产生)
        self.drag_induced_coefficient = 3.75
        self.spoiler_drag_coefficient = 20  # 减速板阻力系数

        # -- 挂载及重量设定 --
        self.base_weight = 20000  # 无挂载重量 (kg) 20t
        self.max_payload = 10000  # 挂载限重 10t
        self.bomb_inventory, self.ini_payload = self.init_payload(bomb_preset)
        self.mass = self.base_weight + self.ini_payload
        self.selected_bomb_index = 0

        self.k_pitch_speed = -0.001  # 每公斤挂载，俯仰率降低0.001度/秒 10t降低10，到20
        self.k_roll_speed = -0.006  # 每公斤挂载，滚转率降低0.006度/秒 10t降低60，到135
        self.k_aoa_recovery = -0.00008  # 每公斤挂载，恢复速度降低 (更具惯性) 10t降低0.8，到1.3
        self.k_induced_drag = 0.00007  # 每公斤挂载，诱导阻力增加 (能量损失更快) 10t升高0.7，到4.45

        # --- 飞控逻辑参数 ---
        self.AGGRESSIVE_BANK_THRESHOLD = 33  # 区分温和转弯与战斗机动的坡度阈值 (度)
        self.COORDINATED_TURN_END_ANGLE = 90.0  # 协调转弯效果完全消失的坡度
        self.ALTITUDE_LOSS_END_ANGLE = 180.0  # 最大掉高的坡度
        self.max_altitude_loss_speed = 20  # 大坡度下每秒最大掉高速度 (m/s)
        self.linear_approx_tan = True

        # --- 自动回正逻辑参数 ---
        self.no_pitch_input_timer = 0.0  # 计时器，用于检测无操作状态
        self.no_roll_input_timer = 0.0
        self.AUTO_LEVEL_DELAY = 0.25  # 无操作多少秒后开始自动回正
        self.AUTO_LEVEL_PITCH_RECOVERY_STRENGTH = 0.5  # 回正的力度 (值越小越平滑)
        self.AUTO_LEVEL_ROLL_RECOVERY_STRENGTH = 0.75
        self.STABLE_FLIGHT_PITCH_ANGLE = 1  # 激活所需角度范围
        self.STABLE_FLIGHT_BANK_ANGLE = 3.5

        # --- 状态变量 ---
        self.throttle = 0.5  # 玩家控制的油门 (0.0 to 1.0)
        self.velocity = self.forward * self.speed  # 初始速度矢量
        self.acceleration = 0.0  # 当前加速度 (m/s^2)
        self.aoa = 0.0  # 迎角 (度)

        self.hud_message_timer = 0

        self.WEAPON_MODE = 0  # 0: NAV, 1: CCIP, 2: CCRP
        self.ccrp_target_point = None  # 存储被锁定的ccrp目标点世界坐标
        self.fire_button_held = False  # 标记是否正按住开火键进行投弹
        self.chosen_fac_dir = None  # ccrp航向道
        self.suggested_bank_angle = 0.0  # ccrp坡度指引

        # --- 摄像机与控制 ---
        camera.parent = self
        camera.position = (0, 0, 0)
        camera.fov = 75  # 垂直
        camera.aspect_ratio = 1.778  # 16:9

    def init_payload(self, preset_key):
        inventory = BOMB_PRESET.get(preset_key, BOMB_PRESET[1])
        payload = calculate_payload(inventory)
        if payload > self.max_payload:
            return BOMB_PRESET[1].copy(), calculate_payload(BOMB_PRESET[1])
        else:
            return inventory.copy(), payload

    def update_flight_performance_from_weight(self):
        curr_payload = self.mass - self.base_weight
        # current = base + (curr_payload * k)
        self.pitch_speed = self.base_pitch_speed + curr_payload * self.k_pitch_speed
        self.roll_speed = self.base_roll_speed + curr_payload * self.k_roll_speed
        self.aoa_recovery = self.base_aoa_recovery + curr_payload * self.k_aoa_recovery
        self.drag_induced_coefficient = self.base_drag_induced_coefficient + curr_payload * self.k_induced_drag

    def drop_bomb(self):
        bomb_type_key = BOMB_TYPES[self.selected_bomb_index]
        if self.bomb_inventory[bomb_type_key] > 0:
            self.bomb_inventory[bomb_type_key] -= 1
            self.mass -= BOMBS[bomb_type_key]['weight']
            return True  # 成功投弹
        return False  # 没有弹药

    def input(self, key):
        # 控制油门，一格一格加减，慢车之后有两格减速板
        if key == 'w':
            self.throttle += 0.1
        if key == 's':
            self.throttle -= 0.1
        self.throttle = clamp(self.throttle, -0.2, 1.0)
        self.throttle = round(self.throttle, 1)

        # CCRP TDC (Target Designation Cursor) 游标控制
        if player.WEAPON_MODE == 2 and player.ccrp_target_point is None:
            ccrp_tdc.x += (held_keys['l'] - held_keys['j']) * 0.25 * time.dt
            ccrp_tdc.y += (held_keys['i'] - held_keys['k']) * 0.25 * time.dt
            # 限制TDC在屏幕范围内
            ccrp_tdc.x = clamp(ccrp_tdc.x, -0.35 * camera.aspect_ratio, 0.35 * camera.aspect_ratio)
            ccrp_tdc.y = clamp(ccrp_tdc.y, -0.45, 0.25)

        # 切换弹药
        if key == 'r':
            self.selected_bomb_index = (self.selected_bomb_index + 1) % len(BOMB_TYPES)

    def update(self):
        self.update_flight_performance_from_weight()

        # --- 1. 获取玩家输入 ---
        pitch_input = (held_keys['up arrow'] - held_keys['down arrow'])
        roll_input = (held_keys['left arrow'] - held_keys['right arrow'])

        # --- 2. 自动回正逻辑 (Auto-Leveling) ---
        if pitch_input == 0:
            self.no_pitch_input_timer += time.dt
            if self.no_pitch_input_timer > self.AUTO_LEVEL_DELAY:
                if abs(self.rotation_x) < self.STABLE_FLIGHT_PITCH_ANGLE:
                    self.rotation_x = lerp(self.rotation_x, 0,
                                           time.dt * self.AUTO_LEVEL_PITCH_RECOVERY_STRENGTH)  # 使用lerp平滑地将姿态恢复到0
        else:
            self.no_pitch_input_timer = 0

        if roll_input == 0:
            self.no_roll_input_timer += time.dt
            if self.no_roll_input_timer > self.AUTO_LEVEL_DELAY:
                if abs(self.rotation_z - self.suggested_bank_angle) < self.STABLE_FLIGHT_BANK_ANGLE:
                    self.rotation_z = lerp(self.rotation_z, self.suggested_bank_angle,
                                           time.dt * self.AUTO_LEVEL_ROLL_RECOVERY_STRENGTH)
        else:
            self.no_roll_input_timer = 0

        # --- 3. super FBW，使用四元数进行姿态更新 ---
        # 计算姿态变化(直接控制姿态，而不是力)
        # 有效坡度角
        effective_bank_angle = self.rotation_z  # -180 to 180 去除倒飞角度
        if effective_bank_angle > 90:
            effective_bank_angle = 180 - effective_bank_angle
        if effective_bank_angle < -90:
            effective_bank_angle = -180 - effective_bank_angle

        # 协调转弯与高度保持辅助强度
        # 协调转弯强度 线性衰减 角度区间 1到0
        turn_assist_strength = remap(abs(effective_bank_angle),
                                     self.AGGRESSIVE_BANK_THRESHOLD, self.COORDINATED_TURN_END_ANGLE,
                                     1, 0)
        turn_assist_strength = clamp(turn_assist_strength, 0, 1)
        # 模拟掉高强度 线性增加 角度区间 0到1
        loss_assist_strength = remap(abs(self.rotation_z),
                                     self.AGGRESSIVE_BANK_THRESHOLD, self.ALTITUDE_LOSS_END_ANGLE,
                                     0, 1)
        loss_assist_strength = clamp(loss_assist_strength, 0, 1)

        # a) 创建代表本帧俯仰的旋转 (绕着飞机的 right 轴)
        q_pitch = Quat()
        q_pitch.setFromAxisAngle(pitch_input * self.pitch_speed * time.dt, self.right)
        # b) 创建代表本帧滚转的旋转 (绕着飞机的 forward 轴)
        q_roll = Quat()
        q_roll.setFromAxisAngle(roll_input * self.roll_speed * time.dt, self.forward)
        # c) 自动协调转弯：创建代表本帧偏航的旋转（self.up是本地轴，还是选择Vec3(0, 1, 0)世界向上吧）
        # 偏航的速率由当前的滚转角角度self.rotation_z决定
        # 根据当前的滚转角度(坡度)，自动产生一个偏航(转向)力矩
        q_yaw = Quat()
        if self.speed > 1:
            if self.linear_approx_tan:
                self.yaw_from_roll = G / self.speed
                q_yaw.setFromAxisAngle(effective_bank_angle * self.yaw_from_roll * time.dt * turn_assist_strength,
                                       Vec3(0, 1, 0))
            else:
                # 将飞机的坡度角从度转换为弧度
                bank_angle_rad = math.radians(effective_bank_angle)
                # 转弯角速度 (ω) = (g * tan(横滚坡度角)) / 速度 (v)
                # 单位是 弧度/秒
                turn_rate_rad_per_sec = (G * math.tan(bank_angle_rad)) / self.speed
                # 转换为 度/秒
                turn_rate_deg_per_sec = math.degrees(turn_rate_rad_per_sec)
                # 应用
                q_yaw.setFromAxisAngle(turn_rate_deg_per_sec * time.dt * turn_assist_strength,
                                       Vec3(0, 1, 0))
        # d) 将所有旋转组合起来，应用到飞机当前的姿态上
        # 顺序：先应用偏航，再应用俯仰，最后应用滚转
        self.quaternion = self.quaternion * q_yaw * q_pitch * q_roll

        # --- 4. 推力-阻力-势能-目标速度模型 ---
        # a) 计算推力、阻力、重力分量，得到净加速度
        # 如果油门为正，产生推力；如果为负，推力为0
        thrust = max(0.0, self.throttle) * self.max_thrust
        # （这里是用的speed计算阻力，而不是velocity）
        base_drag = self.speed ** 2 * self.drag_base_coefficient
        induced_drag = self.speed ** 2 * abs(pitch_input) * self.drag_induced_coefficient * (self.aoa / 10) ** 2  # 二值
        spoiler_drag = self.speed ** 2 * abs(self.throttle) * self.spoiler_drag_coefficient if self.throttle < 0 else 0
        self.acceleration = (thrust - base_drag - induced_drag - spoiler_drag) / self.mass  # 数值
        self.acceleration -= G * self.velocity.normalized().y  # sin俯仰角
        # b) 更新当前的速度值
        self.speed += self.acceleration * time.dt  # 数值
        self.speed = max(self.speed, 1)  # 避免速度降为0
        # c) 将最终速度应用到速度矢量上
        # 模拟掉高（先进行，用后续的平滑削弱一部分影响）
        if loss_assist_strength > 0:
            # 指数使其在小角度时效果更弱，大角度时更强
            altitude_loss_vector = Vec3(0, -1, 0) * self.max_altitude_loss_speed * (loss_assist_strength**2)
            self.velocity += altitude_loss_vector * time.dt
        # 使用lerp（线性插值）让速度矢量平滑地对准机头方向
        # 这里的设定分离了speed（目标）与velocity.length()（实际）
        self.velocity = lerp(self.velocity, self.forward * self.speed, time.dt * self.aoa_recovery)

        # 计算迎角
        if self.velocity.length() > 1:
            # 速度矢量和机头指向的点积，得到它们夹角的余弦值
            dot_product = self.velocity.normalized().dot(self.forward)
            # 反余弦得到弧度，再转为度
            self.aoa = math.degrees(math.acos(clamp(dot_product, -1, 1)))
        else:
            self.aoa = 0

        # --- 5. 更新位置 ---
        self.position += self.velocity * time.dt


class LiveBomb(Entity):
    def __init__(self, start_pos, start_vel, bomb_properties, **kwargs):
        super().__init__(
            model='sphere',
            scale=1,  # 直径1米
            color=color.orange,
            position=start_pos,
            collider='sphere',
            **kwargs
        )
        self.velocity = start_vel
        self.properties = bomb_properties
        self.is_active = True

        # 轨迹
        self.trail_points = [self.position, self.position]
        self.trail_renderer = Entity(model=Mesh(vertices=self.trail_points, mode='line', thickness=2),
                                     color=color.orange)
        self.trail_update_timer = 0.0

    def update(self):
        if not self.is_active:
            return

        # --- 1. 物理计算 (与预测逻辑相同，但使用 time.dt) ---
        # a) 空气阻力
        drag_force_magnitude = self.velocity.length_squared() * self.properties['drag_coefficient']
        drag_acceleration_magnitude = drag_force_magnitude / self.properties['weight']
        self.velocity -= self.velocity.normalized() * drag_acceleration_magnitude * time.dt

        # b) 重力
        self.velocity.y -= G * time.dt

        # --- 2. 碰撞检测 ---
        hit_info = raycast(
            origin=self.position,
            direction=self.velocity.normalized(),
            distance=self.velocity.length() * time.dt,
            ignore=[player, self]  # 忽略玩家和自身
        )

        if hit_info.hit:
            self.position = hit_info.world_point
            self.is_active = False
            self.color = color.black
            return

        # --- 3. 更新位置 ---
        self.position += self.velocity * time.dt

        # --- 4. 更新轨迹 ---
        self.trail_update_timer += time.dt
        if self.trail_update_timer > 0.1:  # 每0.1秒记录一个点
            self.trail_points.append(self.position)
            self.trail_renderer.model.vertices = self.trail_points
            self.trail_renderer.model.generate()
            self.trail_update_timer = 0

        # 如果炸弹掉出世界，自我销毁
        if self.y < -10:
            destroy(self.trail_renderer)
            destroy(self)


def remap(value, x1, x2, y1, y2):
    return y1 + (value - x1) * (y2 - y1) / (x2 - x1)


def calculate_payload(inventory):
    return sum(BOMBS[k]['weight'] * v for k, v in inventory.items())


def calculate_ideal_fall_time(from_y, to_y, v_y):
    altitude_diff = from_y - to_y
    a = -0.5 * G
    b = v_y
    c = altitude_diff
    discriminant = b ** 2 - 4 * a * c
    return ((-b - math.sqrt(discriminant)) / (2 * a)) if discriminant >= 0 else 99


def calculate_ideal_fall_raycast(start_pos, start_vel):
    """
    :param start_pos: Vec3
    :param start_vel: Vec3
    :return:
    """
    time_step = 0.05  # 模拟的时间步长
    max_steps = 200  # 最多模拟200步

    pos = start_pos
    vel = start_vel

    for _ in range(max_steps):
        next_pos = pos + vel * time_step
        # 从上一步的位置向下一步的位置发射光线
        hit_info = raycast(origin=pos, direction=(next_pos - pos).normalized(), distance=vel.length() * time_step,
                           ignore=[player, ])
        if hit_info.hit:
            # 如果光线撞到了东西，返回撞击点
            return hit_info.world_point
        # 如果没撞到，更新位置和速度，继续模拟
        pos = next_pos
        vel.y -= G * time_step

    # 超过最大模拟步数返回None
    return None


def input(key):
    # Q键: 循环切换武器模式
    if key == 'q':
        player.WEAPON_MODE = (player.WEAPON_MODE + 1) % 3
        # 切换模式时，重置状态
        player.ccrp_target_point = None
        player.fire_button_held = False
        player.chosen_fac_dir = None
        player.suggested_bank_angle = 0.0
        ccrp_tdc.position = (0, 0, 0)

    # E键: 在CCRP模式下用于指定目标
    if key == 'e' and player.WEAPON_MODE == 2:
        if player.ccrp_target_point is None:
            # 获取摄像机在TDC屏幕坐标处发出的射线方向
            ray_direction = screen_point_to_ray(ccrp_tdc.position)
            hit_info = raycast(origin=camera.world_position, direction=ray_direction, ignore=[player, ])
            if hit_info.hit:
                player.ccrp_target_point = hit_info.world_point
                ccrp_tdc_locked.position = ccrp_tdc.position
        else:
            player.ccrp_target_point = None  # 再次按下取消
            ccrp_tdc.position = ccrp_tdc_locked.position

    # 空格键: 统一的开火键
    if key == 'space':
        player.fire_button_held = True
    if key == 'space up':
        player.fire_button_held = False


def update():
    # --- 1. 更新姿态显示 (枢轴与滑块逻辑) ---
    # 步骤A: 将滚转角度应用到"枢轴"上。
    attitude_pivot.rotation_z = -player.world_rotation_z
    roll_pivot.rotation_z = -player.world_rotation_z
    suggest_roll_pivot.rotation_z = -player.suggested_bank_angle

    # 步骤B: 计算俯仰位移，并将其应用到"滑块"的Y轴上。
    # 因为滑块是枢轴的子元素，它的Y轴已经随着枢轴旋转了。
    # 所以Y轴平移会自动变成沿着旋转后的“垂直”方向的滑动。
    pitch_slider.y = player.world_rotation_x * 0.0175

    # --- 2. 更新HUD消息 ---
    if player.hud_message_timer > 0:
        player.hud_message_timer -= time.dt
        if player.hud_message_timer <= 0:
            hud_message.enabled = False

    # --- 3. 更新其他HUD读数 ---
    # 顶部罗盘
    heading = (math.degrees(math.atan2(player.forward.x, player.forward.z)) + 360) % 360
    heading_text.text = f"{int(heading):03d}"

    # 地平线刻度
    horizon_compass_content.x = -heading * HORIZON_HEADING_SCALE

    # 速度
    current_speed_knots = player.velocity.length() * MS_TO_KNOTS
    speed_text.text = f"{int(current_speed_knots)}"

    # 高度
    current_altitude_feet = player.y * METERS_TO_FEET
    altitude_text.text = f"{int(current_altitude_feet)}"

    # 垂直速度 (Vertical Speed)
    vs_fpm = player.velocity.y * METERS_TO_FEET * 60  # m/s -> ft/min
    vs_text.text = f"VS {int(vs_fpm)}"

    # 无线电高度 (Radio Altimeter)
    hit_info = raycast(origin=player.world_position, direction=Vec3(0, -1, 0), ignore=[player, ],
                       distance=2500 * FEET_TO_METERS)
    if hit_info.hit:
        radio_alt_feet = hit_info.distance * METERS_TO_FEET
        radio_alt_text.text = f"R {int(radio_alt_feet)}"
    else:
        radio_alt_text.text = "R ----"

    # 重量
    gross_weight_text.text = f"GW\n{int(player.mass)}KG"

    # 预测速度
    predicted_speed = (player.speed + player.acceleration * 10) * MS_TO_KNOTS  # 10秒后速度(节)
    predicted_speed_text.text = f"V {int(predicted_speed)}"

    # 迎角 (Angle of Attack)
    aoa_text.text = f"AOA {player.aoa:.1f}"

    # 油门/减速板状态
    if player.throttle < 0:
        throttle_text.text = f"THR IDLE\nSPD BRK\nSPLR {int(abs(player.throttle) * 10)}/2"
    elif player.throttle == 0:
        throttle_text.text = "IDLE"
    elif player.throttle < 1.0:
        throttle_text.text = f"MIL {int(player.throttle * 100)}%"
    else:
        throttle_text.text = "AB"  # Afterburner

    # --- 4. 更新飞行路径矢量 (FPV) 位置 ---
    velocity_vector = player.velocity
    if velocity_vector.length() > 0:  # >1
        FPV_proxy.position = player.position + velocity_vector.normalized() * 100  # 投射到100米外
        FPV.position = FPV_proxy.screen_position

    # --- 5. 更新武器HUD ---
    weapon_mode_text.text = ['NAV', 'CCIP', 'CCRP'][player.WEAPON_MODE]
    selected_bomb_key = BOMB_TYPES[player.selected_bomb_index]
    selected_bomb = BOMBS[selected_bomb_key]
    remaining_count = player.bomb_inventory[selected_bomb_key]
    weapon_select_text.text = f"{selected_bomb['name']}\n{remaining_count}" if player.WEAPON_MODE else ""

    # 默认隐藏所有武器HUD
    ccip_pipper.enabled = False
    ccip_offscreen_indicator.enabled = False
    ccip_bomb_fall_line.enabled = False
    ccrp_steering_line.enabled = False
    ccrp_release_cue.enabled = False
    ccrp_info_text.enabled = False
    ccrp_tdc.enabled = False
    ccrp_tdc_locked.enabled = False
    loc_guide_text.enabled = False

    # 默认值
    player.AUTO_LEVEL_ROLL_RECOVERY_STRENGTH = 0.75

    # --- CCIP 计算与显示 (边缘指示器) ---
    if player.WEAPON_MODE == 1:
        impact_world_point = calculate_bomb_trajectory(player.position, Vec3(player.velocity), selected_bomb)
        if impact_world_point:
            direction_to_target = (impact_world_point - camera.world_position).normalized()
            if camera.forward.dot(direction_to_target) > 0:
                ccip_proxy.position = impact_world_point
                screen_pos = ccip_proxy.screen_position
                if abs(screen_pos.x) < 0.5 and abs(screen_pos.y) < 0.5:
                    ccip_pipper.enabled = True
                    ccip_pipper.position = screen_pos
                    # 显示竖线
                    ccip_bomb_fall_line.enabled = True
                    ccip_bomb_fall_line.model.vertices = [Vec3(FPV.x, FPV.y, 0), Vec3(screen_pos.x, screen_pos.y, 0)]
                    ccip_bomb_fall_line.model.generate()
                    # 投弹
                    if player.fire_button_held:
                        release_bomb()
                else:
                    ccip_offscreen_indicator.enabled = True
                    clamped_x = clamp(screen_pos.x, -0.5, 0.5)
                    clamped_y = clamp(screen_pos.y, -0.48, 0.48)
                    ccip_offscreen_indicator.position = (clamped_x, clamped_y)
                    angle = math.degrees(math.atan2(screen_pos.y, screen_pos.x * camera.aspect_ratio))
                    ccip_offscreen_indicator.rotation_z = -angle

    # --- CCRP 计算与显示 ---
    elif player.WEAPON_MODE == 2:
        if player.ccrp_target_point is None:
            ccrp_tdc.enabled = True
        else:
            # 已锁定目标，显示目标框 (使用代理实体)
            direction_to_target = (player.ccrp_target_point - camera.world_position).normalized()
            if camera.forward.dot(direction_to_target) > 0:
                ccrp_target_proxy.position = player.ccrp_target_point
                target_screen_pos = ccrp_target_proxy.screen_position

                ccrp_tdc_locked.enabled = True
                ccrp_tdc_locked.position = target_screen_pos

            # --- 如果按住开火键，开始投弹引导 ---
            if player.fire_button_held:
                # A. 计算目标相对于飞机的方位角 (Azimuth)
                vec_to_target = player.ccrp_target_point - player.position
                vec_to_target_xz = Vec3(vec_to_target.x, 0, vec_to_target.z)
                vec_to_target_xz_dir = vec_to_target_xz.normalized()
                velocity_xz = Vec3(player.velocity.x, 0, player.velocity.z)
                velocity_xz_dir = velocity_xz.normalized()
                # 计算带符号的角度，即航迹误差
                azimuth_error_deg = velocity_xz_dir.signed_angle_deg(vec_to_target_xz_dir, Vec3(0, 1, 0))
                # 飞机速度方向与目标方向的夹角 (航迹误差角) 的余弦值
                cos_azimuth_error = velocity_xz_dir.dot(vec_to_target_xz_dir)  # 点积得到cos(theta)
                # math.cos(math.radians(azimuth_error_deg)) == cos_azimuth_error

                # B. 将角度误差转换为屏幕上的X坐标：航迹方位引导线 (ASL - Azimuth Steering Line) 的位置
                ASL_x_pos = azimuth_error_deg / (camera.fov * camera.aspect_ratio / 2) * 0.5
                ASL_x_pos += FPV.position.x  # 以FPV为基准
                ASL_x_pos = clamp(ASL_x_pos, -0.45 * camera.aspect_ratio, 0.45 * camera.aspect_ratio)

                # C. 坡度指引
                loc_guide_text.enabled = True
                player.AUTO_LEVEL_ROLL_RECOVERY_STRENGTH = 5
                calculate_crs_and_roll_guide(velocity_xz, azimuth_error_deg)

                # D. 计算释放倒计时 (TTR - Time To Release)
                # 根据预测轨迹，得到水平距离
                impact_point = calculate_bomb_trajectory(player.position, Vec3(player.velocity), selected_bomb)
                if impact_point and abs(azimuth_error_deg) <= 30:
                    bomb_traj_xz = Vec3(impact_point.x - player.x, 0, impact_point.z - player.z)
                    # 理想释放点 Required Release Point
                    # 水平面上，距离目标bomb_traj_xz.length()的点
                    #  (飞机到目标的水平距离 - 炸弹未修正角度分量的完整水平射程) / cos(航迹误差角)
                    dist_to_release_point = (vec_to_target_xz.length() - bomb_traj_xz.length()) / cos_azimuth_error \
                        if cos_azimuth_error > 0 else float('inf')
                    # 距释放点剩余时间 = 距释放点剩余水平距离 / 余弦修正的飞机水平接近速度
                    time_to_release = dist_to_release_point / velocity_xz.length() \
                        if velocity_xz.length() > 1 else 99

                    # E. 释放
                    if abs(time_to_release) <= 0.05:
                        release_bomb()

                    # F. 显示HUD信息
                    # 将10秒内的倒计时映射到屏幕Y轴，超过10秒则固定在顶部
                    ccrp_release_cue.y = clamp(time_to_release, 0, 10) * 0.04  # Y位置由TTR决定
                    ccrp_info_text.text = f"LR: {vec_to_target.length() / 1000:.1f} km\nTTR: {time_to_release:.1f} s"
                elif abs(azimuth_error_deg) > 30:
                    # 航迹偏差太大，TTR无效
                    ccrp_release_cue.y = 0.4
                    ccrp_info_text.text = f"LR: {vec_to_target.length() / 1000:.1f} km\nTTR: -- s\nALIGN COURSE"
                else:
                    ccrp_release_cue.y = 0.4
                    ccrp_info_text.text = f"LR: {vec_to_target.length() / 1000:.1f} km\nTTR: -- s"

                ccrp_steering_line.x = ASL_x_pos
                ccrp_release_cue.x = ASL_x_pos  # 提示符始终在引导线上
                ccrp_steering_line.enabled = True  # 引导线 (ASL)
                ccrp_release_cue.enabled = True  # 释放提示符
                ccrp_info_text.enabled = True  # 距离和时间倒计时


def calculate_bomb_trajectory(start_pos, aircraft_vel, bomb_properties):
    time_step = 0.1
    max_steps = 300

    # 炸弹的初始速度 = 飞机速度 + 抛射速度 (抛射速度需要从飞机坐标系转到世界坐标系)
    ej = bomb_properties['ejection_velocity']
    ejection_world_vel = (
            player.right * ej.x +
            player.up * ej.y +
            player.forward * ej.z
    )
    vel = aircraft_vel + ejection_world_vel
    pos = start_pos + (vel * time_step)  # 从飞机外一点开始，避免立刻撞到飞机自身

    for _ in range(max_steps):
        # 简化的空气阻力模型: drag = -k * v^2
        drag_force_magnitude = vel.length_squared() * bomb_properties['drag_coefficient']
        drag_acceleration_magnitude = drag_force_magnitude / bomb_properties['weight']
        vel -= vel.normalized() * drag_acceleration_magnitude * time_step

        # 重力
        vel.y -= G * time_step

        # 更新位置并进行碰撞检测
        next_pos = pos + vel * time_step
        hit_info = raycast(origin=pos, direction=(next_pos - pos).normalized(), distance=vel.length() * time_step,
                           ignore=[player, ])
        if hit_info.hit:
            return hit_info.world_point
        pos = next_pos

        # 飞到地下提前停止
        if pos.y < -10:
            return None

    # 超过最大模拟步数返回None
    return None


def screen_point_to_ray(screen_pos):
    """
    将Ursina的UI坐标转换为世界空间的射线
    Inputs:
        screen_pos: Vec3
    Outputs:
        ray_direction: Vec3 normalized direction vector of the ray.
    """
    # resolution = window.size  # Vec2(1536, 864)
    sx, sy, _ = screen_pos  # 屏幕中间是Vec3(0, 0, 0)，符号按标准象限

    # --- 1. 将 Ursina UI 坐标转换为归一化设备坐标 (NDC) ---
    # Ursina UI 范围: [-0.5, 0.5]
    # NDC 范围: [-1, 1]
    # 简单的线性缩放
    ndc_x = sx / 0.5
    ndc_y = sy / 0.5

    # --- 2. 将 NDC 坐标转换为相机空间中的方向 ---
    # 计算在距离相机camera.aspect_ratio个单位的近裁剪平面上，这个点的位置。
    tan_half_fov = math.tan(math.radians(camera.fov / 2.0))
    camera_space_x = ndc_x * tan_half_fov
    camera_space_y = ndc_y * tan_half_fov
    camera_space_z = camera.aspect_ratio  # 就是这个数，但没明白为什么

    # --- 3. 将相机空间方向矢量转换到世界空间 ---
    # 在相机空间中，朝向Z+方向看
    # 从相机原点到近裁剪平面上点的方向矢量是 Vec3(camera_space_x, camera_space_y, camera_space_z)
    # 使用相机的世界坐标系基向量 (right, up, forward) 进行变换
    # 这等同于将 camera_space_dir 左乘相机的世界旋转矩阵
    world_space_dir = (camera.right * camera_space_x +
                       camera.up * camera_space_y +
                       camera.forward * camera_space_z)

    return world_space_dir.normalized()


def release_bomb():
    # 一次只能单发投放一种航弹
    if player.drop_bomb():
        spawn_live_bomb()
        set_hud_message("BOMB AWAY!", 3)
        player.ccrp_target_point = None
        player.chosen_fac_dir = None
        player.suggested_bank_angle = 0.0
    else:
        set_hud_message("NO ORDNANCE", 2)
    player.fire_button_held = False


def spawn_live_bomb():
    """根据玩家状态创建一个真实的、可见的炸弹实体"""
    # a) 获取当前选择的炸弹属性
    bomb_properties = BOMBS[BOMB_TYPES[player.selected_bomb_index]]

    # b) 计算炸弹的初始速度 (与预测逻辑相同)
    ej = bomb_properties['ejection_velocity']
    ejection_world_vel = (
            player.right * ej.x +
            player.up * ej.y +
            player.forward * ej.z
    )
    initial_vel = Vec3(player.velocity) + ejection_world_vel

    # c) 计算初始位置
    initial_pos = player.position

    # d) 创建 LiveBomb 实例
    LiveBomb(start_pos=initial_pos, start_vel=initial_vel, bomb_properties=bomb_properties)


def set_hud_message(input_text, display_time):
    hud_message.enabled = True
    hud_message.text = input_text
    player.hud_message_timer = display_time


def calculate_crs_and_roll_guide(velocity_xz, azimuth_error_deg):
    """
    :param velocity_xz: 飞机当前水平速度矢量 Vec3
    :param azimuth_error_deg: 航迹误差 度

    player.suggested_bank_angle: 建议坡度角 度
    player.chosen_fac_dir: 被选定的FAC方向 Vec3，用于调试或显示
    """
    player_pos_xz = Vec3(player.position.x, 0, player.position.z)
    target_pos_xz = Vec3(player.ccrp_target_point.x, 0, player.ccrp_target_point.z)
    vel_xz_dir = velocity_xz.normalized()
    horizontal_speed = velocity_xz.length()

    if horizontal_speed < 10.0:
        return

    # --- 阶段一：寻找FAC ---
    _REPLAN_AZIMUTH_THRESHOLD = 5.0
    _PLANNED_BANK_ANGLE = 20.0  # 用于计算转弯半径的基准坡度 (度)
    _MAX_SUGGESTED_BANK_ANGLE = 30.0
    if player.chosen_fac_dir is None or \
            abs(vel_xz_dir.signed_angle_deg(player.chosen_fac_dir, Vec3(0, 1, 0))) > _REPLAN_AZIMUTH_THRESHOLD:
        # 1. 计划的转弯半径
        planned_turn_radius = horizontal_speed ** 2 / (G * math.tan(math.radians(_PLANNED_BANK_ANGLE)))

        # 2. 计算左右转弯的圆心
        # 与速度方向垂直的向量 右侧
        vel_xz_dir_right = Vec3(vel_xz_dir.z, 0, -vel_xz_dir.x)
        if azimuth_error_deg >= 0:
            # right_turn
            center = player_pos_xz + vel_xz_dir_right * planned_turn_radius
            player.suggested_bank_angle = _PLANNED_BANK_ANGLE
        else:
            # left_turn
            center = player_pos_xz - vel_xz_dir_right * planned_turn_radius
            player.suggested_bank_angle = -_PLANNED_BANK_ANGLE

        # 3. 计算从目标点出发的两条切线
        vec_center_to_target = target_pos_xz - center
        dist_center_to_target = vec_center_to_target.length()

        # 目标在转弯圆内，无法规划切线
        if dist_center_to_target <= planned_turn_radius:
            set_hud_message("TOO CLOSE", 2)
            return

        # a) 计算角度
        # 圆心到目标点的角度
        theta = math.atan2(vec_center_to_target.z, vec_center_to_target.x)
        # 目标点向圆心方向看两个切点的视角的一半 用于计算切线
        alpha = math.asin(planned_turn_radius / dist_center_to_target)
        # 圆心向目标点方向看两个切点的视角的一半 用于计算切点
        beta = math.acos(planned_turn_radius / dist_center_to_target)

        # b) 计算切线切点
        fac_dir1 = Vec3(math.cos(theta - alpha), 0, math.sin(theta - alpha))
        fac_dir2 = Vec3(math.cos(theta + alpha), 0, math.sin(theta + alpha))
        fac_dir1_tangent = center + Vec3(math.cos(theta + beta), 0, math.sin(theta + beta)) * planned_turn_radius
        fac_dir2_tangent = center + Vec3(math.cos(theta - beta), 0, math.sin(theta - beta)) * planned_turn_radius

        # c) 找到正确的切线：切点必在前方转弯圆的四分之一圆弧上
        if (fac_dir1_tangent - player_pos_xz).length() < (fac_dir2_tangent - player_pos_xz).length():
            player.chosen_fac_dir = fac_dir1
            tangent = fac_dir1_tangent
        else:
            player.chosen_fac_dir = fac_dir2
            tangent = fac_dir2_tangent

        # HUD提示
        loc_guide_text.text = "LOC *"

        # debug 计算它们之间的夹角 (0-180度)
        arc_angle = math.degrees(math.acos(clamp(
            (player_pos_xz - center).normalized().dot((tangent - center).normalized()),
            -1, 1)))
        if not 0 <= arc_angle <= 90:
            print("[debug] arc_angle")

        # # debug 可视化入航航道
        # if player.chosen_fac_dir:
        #     p1 = Vec3(target_pos_xz)
        #     p2 = target_pos_xz - player.chosen_fac_dir * 100000
        #     p1.y = p2.y = player.y
        #     Entity(model=Mesh(vertices=[p1, p2], mode='line'), color=color.blue)

    # fac is not None and deg <= threshold
    else:
        # --- 阶段二：执行引导 ---
        _L1_PERIOD = 12.0  # L1时间常数 (秒)。值越大，响应越平缓，转弯半径越大。

        # a) 计算XTE
        vec_target_to_player_xz = player_pos_xz - target_pos_xz
        # ax-b=-(axb)
        # .x: ay·bz - az·by, 0
        # .y: az·bx - ax·bz, +-|a||b|sin[0, 180] 左手
        # .z: ax·by - ay·bx, 0
        cross_track_error = player.chosen_fac_dir.cross(vec_target_to_player_xz).y  # 偏左负号 偏右正号

        # b) 计算L1距离
        l1_distance = horizontal_speed * _L1_PERIOD / math.pi

        # c) 计算η
        heading_error_rad = math.radians(vel_xz_dir.signed_angle_deg(player.chosen_fac_dir, Vec3(0, 1, 0)))
        eta_rad = math.atan2(-cross_track_error, l1_distance) + heading_error_rad  # P D

        # d) 计算加速度指令
        a_cmd = (2 * horizontal_speed ** 2 / l1_distance) * math.sin(eta_rad)

        # e) 转换为坡度指令 左横滚是负值 右横滚是正值
        tan_bank = a_cmd / G
        bank_angle = math.degrees(math.atan(tan_bank))
        player.suggested_bank_angle = clamp(bank_angle, -_MAX_SUGGESTED_BANK_ANGLE, _MAX_SUGGESTED_BANK_ANGLE)

        # HUD提示
        loc_guide_text.text = "LOC"

    return


# --- 主程序 ---
app = Ursina()

sky = Sky()

# --- 程序化生成高度图 ---
print("正在生成地形高度图...")
MAP_SIZE = 1024  # 高度图的分辨率 (必须是2的n次方+1，或者直接用2的n次方)

# 1. 创建一个空的numpy数组来存储高度数据
height_data = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)

# 2. 创建“台阶”地形：X方向从0到XXX英尺
max_height_value = 255  # 对应XXX英尺
for x in range(MAP_SIZE):
    # 创建一个从0到1的线性渐变
    gradient = x / (MAP_SIZE - 1)
    height_data[:, x] = int(gradient * max_height_value)

# 3. 在NE高原部分挖出“洼地”
hole_center_x, hole_center_y = MAP_SIZE // 4 * 3, MAP_SIZE // 4
hole_radius = MAP_SIZE // 8
y, x = np.ogrid[:MAP_SIZE, :MAP_SIZE]
mask = (x - hole_center_x)**2 + (y - hole_center_y)**2 < hole_radius**2
# 在圆形区域内，将高度降低
height_data[mask] = 0

# 4. 将numpy数组转换为图像并保存
heightmap_image = Image.fromarray(height_data)
heightmap_image.save('terrain_heightmap.png')
print("地形高度图 'terrain_heightmap.png' 已生成。")

# 5. 使用高度图创建地形
terrain = Terrain(
    heightmap='terrain_heightmap.png',
    skip=64  # 优化性能，每64个像素取一个顶点
)
terrain_entity = Entity(
    model=terrain,
    scale=(20000, 2000 * FEET_TO_METERS, 20000),  # X, Y(总高度), Z
    collider='mesh',
    texture='grass'
)

# --- 场景设置 ---
player = FlightController(position=(0, 3000 * FEET_TO_METERS, 0))  # 玩家放在地图中间
target1 = Entity(model='cube', color=color.red, scale=20, position=(5000, 0, 5000))  # 目标1放在NE坑里
target2 = Entity(model='cube', color=color.red, scale=35, position=(-5000, 500 * FEET_TO_METERS, 5000))  # 目标2在WE


# --- HUD ---
hud = Entity(parent=camera.ui)

# --- 1. 固定的HUD元素 ---
# 空心圆环（用 line 模式的 Mesh）
circle_vertices_segments = 24
circle_vertices_radius = 0.01
circle_vertices = []
for i in range(circle_vertices_segments + 1):
    angle = 2 * math.pi * i / circle_vertices_segments
    circle_vertices.append(Vec3(math.cos(angle) * circle_vertices_radius, math.sin(angle) * circle_vertices_radius, 0))

# 正方形边框，用4条线段拼凑
square_vertices_size = 0.01  # 半边长
square_vertices = [
    # 上边
    Vec3(-square_vertices_size, square_vertices_size, 0), Vec3(square_vertices_size, square_vertices_size, 0),
    # 右边
    Vec3(square_vertices_size, square_vertices_size, 0), Vec3(square_vertices_size, -square_vertices_size, 0),
    # 下边
    Vec3(square_vertices_size, -square_vertices_size, 0), Vec3(-square_vertices_size, -square_vertices_size, 0),
    # 左边
    Vec3(-square_vertices_size, -square_vertices_size, 0), Vec3(-square_vertices_size, square_vertices_size, 0),
]

# 菱形边框，用4条线段拼凑
diamond_vertices_size = 0.01
diamond_vertices = [
    # 上 -> 右
    Vec3(0, diamond_vertices_size, 0), Vec3(diamond_vertices_size, 0, 0),
    # 右 -> 下
    Vec3(diamond_vertices_size, 0, 0), Vec3(0, -diamond_vertices_size, 0),
    # 下 -> 左
    Vec3(0, -diamond_vertices_size, 0), Vec3(-diamond_vertices_size, 0, 0),
    # 左 -> 上
    Vec3(-diamond_vertices_size, 0, 0), Vec3(0, diamond_vertices_size, 0),
]

# 三角形
tri_up_vertices_base = 0.0125  # 底边半边长
tri_up_vertices_height = 0.01  # 三角形半高度
triangle_up_vertices = [
    # 左边
    Vec3(-tri_up_vertices_base, -tri_up_vertices_height, 0), Vec3(0, tri_up_vertices_height, 0),
    # 右边
    Vec3(0, tri_up_vertices_height, 0), Vec3(tri_up_vertices_base, -tri_up_vertices_height, 0),
    # 底边
    Vec3(tri_up_vertices_base, -tri_up_vertices_height, 0), Vec3(-tri_up_vertices_base, -tri_up_vertices_height, 0),
]
tri_dn_vertices_base = 0.01  # 顶边半边长
tri_dn_vertices_height = 0.008  # 三角形半高度
triangle_down_vertices = [
    Vec3(-tri_dn_vertices_base, tri_dn_vertices_height, 0),  Vec3(0, -tri_dn_vertices_height, 0),
    Vec3(0, -tri_dn_vertices_height, 0),   Vec3(tri_dn_vertices_base, tri_dn_vertices_height, 0),
    Vec3(tri_dn_vertices_base, tri_dn_vertices_height, 0), Vec3(-tri_dn_vertices_base, tri_dn_vertices_height, 0),
]

# 航向带
heading_tape = Entity(parent=hud, y=0.45)
heading_bg = Entity(parent=heading_tape, model='quad', scale=(0.5, 0.05), color=color.black33)
heading_text = Text(parent=heading_tape, text='000', scale=2, y=-0.005, origin=(0, 0))

# 速度带
speed_tape = Entity(parent=hud, x=-0.5 * camera.aspect_ratio + 0.1)
speed_bg = Entity(parent=speed_tape, model='quad', scale=(0.15, 0.5), color=color.black33)
speed_text = Text(parent=speed_tape, text='0', scale=2.5, origin=(0, 0), color=color.white)
predicted_speed_text = Text(parent=speed_tape, text='V 100', scale=1.5, y=0.3, origin=(0, 0), color=color.white)
aoa_text = Text(parent=speed_tape, text='AOA 0.0', scale=1.5, y=-0.3, origin=(0, 0), color=color.white)
throttle_text = Text(parent=speed_tape, text='IDLE', scale=1.5, y=-0.4, origin=(0, 0), color=color.white)

# 高度带
altitude_tape = Entity(parent=hud, x=0.5 * camera.aspect_ratio - 0.1)
altitude_bg = Entity(parent=altitude_tape, model='quad', scale=(0.15, 0.5), color=color.black33)
altitude_text = Text(parent=altitude_tape, text='0', scale=2.5, origin=(0, 0), color=color.white)
vs_text = Text(parent=altitude_tape, text='VS 0', scale=1.5, y=0.3, origin=(0, 0), color=color.white)
radio_alt_text = Text(parent=altitude_tape, text='R 0', scale=1.5, y=-0.3, origin=(0, 0), color=color.white)
gross_weight_text = Text(parent=altitude_tape, text='GW\n00000KG', scale=1.5, y=-0.4, origin=(0, 0), color=color.white)


# --- 2. 姿态显示系统 ---
# 2.1 "枢轴" (Pivot): 这是一个位于屏幕中心的、不可见的父实体。它只负责滚转。
attitude_pivot = Entity(parent=hud)
roll_pivot = Entity(parent=hud)
suggest_roll_pivot = Entity(parent=hud)

# 2.2 "滑块" (Slider): 这是枢轴的子实体。它只负责沿着枢轴旋转后的Y轴上下滑动（俯仰）。
pitch_slider = Entity(parent=attitude_pivot)

# 2.3 将姿态元素作为"滑块"的子元素。继承滑块的位移和枢轴的旋转，作为一个整体运动。
# 地平线
horizon_line = Entity(parent=pitch_slider, model='quad', scale=(4, 0.005), color=color.green)
horizon_4text = Entity(parent=pitch_slider, model='quad', scale=(1, 1), color=(0, 0, 0, 0))
# 创建一个父实体来容纳所有滚动的刻度，它是 horizon_line 的子元素
horizon_compass_content = Entity(parent=horizon_line)
# 这个比例因子决定了地平线上每度航向占据的宽度
HORIZON_HEADING_SCALE = camera.fov / 7500
heading_labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
for angle in range(-180, 540, 5):  # 使用5度间隔，让刻度更密集
    x_pos = angle * HORIZON_HEADING_SCALE
    # 每5度一个短刻度
    tick = Entity(parent=horizon_compass_content, model='quad', scale=(0.001, 1), position=(x_pos, -1, 0),
                  color=color.green)
    # 每30度一个长刻度，每10度显示数字
    display_angle = angle % 360
    if display_angle % 10 == 0:
        if display_angle % 30 == 0:
            tick.y = 0
            tick.scale_y = 5
        # 获取要显示的文本，优先使用N,E,S,W
        label = heading_labels.get(display_angle, f"{display_angle // 10:02d}")
        Text(parent=horizon_compass_content, text=label, position=(x_pos, 5, 0), scale=(0.25, 200), origin=(0, 0),
             color=color.green)

# 俯仰坡度
for i in range(-9, 10):
    if i == 0:
        continue
    angle = i * 10
    Entity(parent=pitch_slider, model='quad', scale=(0.1, 0.004), x=-0.11, y=angle * 0.0175, color=color.green)
    Entity(parent=pitch_slider, model='quad', scale=(0.1, 0.004), x=0.11, y=angle * 0.0175, color=color.green)
    line4text = Entity(parent=horizon_4text, model='quad', scale=(0.025, 0.025), y=angle * 0.0175, color=(0, 0, 0, 0))
    Text(parent=line4text, text=str(abs(angle)), scale=40, x=-8.5, origin=(0, 0), color=color.green)
    Text(parent=line4text, text=str(abs(angle)), scale=40, x=8.5, origin=(0, 0), color=color.green)

# 横滚坡度指示器 HUD下方
ROLL_SCALE_RADIUS = 0.5  # 刻度盘的半径
ROLL_SCALE_CENTER_Y = 0.06  # 圆心偏移位置
# a) 固定的刻度盘
roll_scale_fixed = Entity(parent=hud, y=ROLL_SCALE_CENTER_Y)
roll_tick_angles = {'0': 0.022, '10': 0.01, '20': 0.01, '30': 0.022, '45': 0.01, '60': 0.022}  # 角度 -> 刻度线长度
for angle_deg_str, length in roll_tick_angles.items():
    angle_deg = int(angle_deg_str)
    angle_rad = math.radians(angle_deg)
    # 下方
    y_pos = -ROLL_SCALE_RADIUS * math.cos(angle_rad)
    # 右侧刻度
    x_pos = ROLL_SCALE_RADIUS * math.sin(angle_rad)
    right_tick = Entity(parent=roll_scale_fixed, model='quad',
                        origin=(0, -length / 2),  # 从中心点向上延伸
                        scale=(0.004, length),
                        position=(x_pos, y_pos),
                        rotation_z=-angle_deg,  # 让刻度线垂直于圆弧
                        color=color.green)
    # 左侧刻度
    x_pos = -ROLL_SCALE_RADIUS * math.sin(angle_rad)
    left_tick = Entity(parent=roll_scale_fixed, model='quad',
                       origin=(0, -length / 2),
                       scale=(0.004, length),
                       position=(x_pos, y_pos),
                       rotation_z=angle_deg,
                       color=color.green)
# b) 移动的指针
roll_pivot.y = ROLL_SCALE_CENTER_Y
suggest_roll_pivot.y = ROLL_SCALE_CENTER_Y
# 飞机姿态
Entity(parent=roll_pivot, model=Mesh(vertices=triangle_down_vertices, mode='line', thickness=3),
       y=-ROLL_SCALE_RADIUS + 0.03,
       color=color.green)
# 坡度指引/回正参考
Entity(parent=suggest_roll_pivot, model=Mesh(vertices=triangle_up_vertices, mode='line', thickness=2),
       y=-ROLL_SCALE_RADIUS,
       color=color.green)

# 速度矢量/飞行路径矢量 Flight Path Vector FPV
boresight = Entity(parent=hud, model='circle', scale=0.005, thickness=2, color=color.cyan)
FPV = Entity(parent=hud)
FPV_proxy = Entity()
FPV_ring = Entity(parent=FPV, model=Mesh(vertices=circle_vertices, mode='line'), color=color.green)
# 短线
tick_offset = circle_vertices_radius + 0.004  # 圆环外侧留一点间隙
FPV_U = Entity(parent=FPV, model='quad', scale=(0.003, 0.0065), position=(0, tick_offset), color=color.green)
FPV_L = Entity(parent=FPV, model='quad', scale=(0.01, 0.003), position=(-tick_offset, 0), color=color.green)
FPV_R = Entity(parent=FPV, model='quad', scale=(0.01, 0.003), position=(tick_offset, 0), color=color.green)


# --- 3. 武器HUD元素 ---
weapon_mode_text = Text(parent=hud, origin=(0, 0), x=0.5, y=0.35, text='NAV', scale=1.5, color=color.green)
weapon_select_text = Text(parent=hud, origin=(0, 0), x=-0.5, y=0.25, text='', scale=1.5, color=color.green)
loc_guide_text = Text(parent=hud, origin=(0, 0), x=0.1, y=-0.47, text='', scale=0.8, color=color.green)
hud_message = Text(parent=hud, origin=(0, 0), y=-0.2, text='', color=color.orange, scale=1.5, enabled=False)

# CCIP
ccip_pipper = Entity(parent=hud, model='circle', scale=0.02, color=color.yellow, thickness=2)
ccip_offscreen_indicator = Entity(parent=hud, model='diamond', scale=(0.02, 0.03), color=color.yellow)
ccip_bomb_fall_line = Entity(parent=hud, model=Mesh(vertices=[(0, 0, 0), (0, 0, 0)], mode='line', thickness=2),
                             color=color.yellow)
ccip_proxy = Entity()  # 用于3D->2D投影得到屏幕坐标

# CCRP
# 目标锁定框
ccrp_tdc = Entity(parent=hud)
ccrp_tdc_frame = Entity(parent=ccrp_tdc, model=Mesh(vertices=square_vertices, mode='line', thickness=2),
                        color=color.green)
ccrp_tdc_locked = Entity(parent=hud)
ccrp_tdc_locked_frame = Entity(parent=ccrp_tdc_locked, model=Mesh(vertices=diamond_vertices, mode='line', thickness=2),
                               color=color.green)
# 飞行引导线
ccrp_steering_line = Entity(parent=hud, model='quad', scale=(0.005, 0.8), color=color.green)
# 释放提示符 (会沿着引导线下滑)
ccrp_release_cue = Entity(parent=hud, model='quad', scale=(0.08, 0.008), color=color.green)
# CCRP信息文本 (距离和时间)
ccrp_info_text = Text(parent=hud, text='', origin=(0, 0), x=0.5, y=0.15, scale=1.5, color=color.green)
# CCRP的代理实体
ccrp_target_proxy = Entity()
ccrp_release_proxy = Entity()


app.run()
