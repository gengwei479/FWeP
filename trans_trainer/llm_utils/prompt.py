import numpy as np
from utils.loadcsv import trans_from_zjenv_to_mrad
# from trans_trainer.llm_utils.tools import get_function_tools, convert_list_to_str, convert_list_to_str_without_id, get_function_tools_info, gen_tools_desc

# @tool
def Straight(master_data):
    # "Find the straight trajectory from expert data: master_data."
    diff = np.sum(abs(np.diff(master_data[:, :, 0])) + np.sum(np.diff(master_data[:, :, 1])), axis = 1)
    filtered_diff = master_data[np.argsort(diff)[0]]
    return filtered_diff
# @tool
def Down(master_data, turn_weight: float):
    # "Plan descending trajectories based on master_data, where parameter: turn_weight represents the execution extent of this action within [0.1, 0.9]"
    cur_traj = Straight(master_data)
    h_diff = np.sum(master_data[:, :, 0] - cur_traj[:, 0], axis = 1)
    h_filtered_diff = master_data[np.argsort(abs(h_diff))[: int(len(master_data) * 0.15)]]
    v_diff = np.sum(h_filtered_diff[:, :, 2] - cur_traj[:, 2], axis = 1)
    v_diff = np.where(v_diff > 0, 0, v_diff)
    v_filtered_diff = h_filtered_diff[np.argsort(v_diff)[int(np.count_nonzero(v_diff) * (1 - turn_weight))]]
    return v_filtered_diff
# @tool
def Climb(master_data, turn_weight: float):
    # "Plan ascending trajectories based on master_data, where parameter: turn_weight represents the execution extent of this action within [0.1, 0.9]"
    cur_traj = Straight(master_data)
    h_diff = np.sum(master_data[:, :, 0] - cur_traj[:, 0], axis = 1)
    h_filtered_diff = master_data[np.argsort(abs(h_diff))[: int(len(master_data) * 0.15)]]
    v_diff = np.sum(h_filtered_diff[:, :, 2] - cur_traj[:, 2], axis = 1)
    v_diff = np.where(v_diff < 0, 0, v_diff)
    v_filtered_diff = h_filtered_diff[np.argsort(v_diff)[::-1][int(np.count_nonzero(v_diff) * (1 - turn_weight))]]
    return v_filtered_diff
# @tool
def TurnLeft(master_data, turn_weight: float):
    # "Plan left-turning trajectories based on master_data, where parameter: turn_weight represents the execution extent of this action within [0.1, 0.9]"
    cur_traj = Straight(master_data)
    v_diff = np.sum(master_data[:, :, 2] - cur_traj[:, 2], axis = 1)
    v_filtered_diff = master_data[np.argsort(abs(v_diff))[: int(len(master_data) * 0.15)]]
    h_diff = np.sum(v_filtered_diff[:, :, 0] - cur_traj[:, 0], axis = 1)
    h_diff = np.where(h_diff > 0, 0, h_diff)
    h_filtered_diff = v_filtered_diff[np.argsort(h_diff)[int(np.count_nonzero(h_diff) * (1 - turn_weight))]]
    return h_filtered_diff
# @tool
def TurnRight(master_data, turn_weight: float):
    # "Plan right-turning trajectories based on master_data, where parameter: turn_weight represents the execution extent of this action within [0.1, 0.9]"
    cur_traj = Straight(master_data)
    v_diff = np.sum(master_data[:, :, 2] - cur_traj[:, 2], axis = 1)
    v_filtered_diff = master_data[np.argsort(abs(v_diff))[: int(len(master_data) * 0.15)]]
    h_diff = np.sum(v_filtered_diff[:, :, 0] - cur_traj[:, 0], axis = 1)
    h_diff = np.where(h_diff < 0, 0, h_diff)
    h_filtered_diff = v_filtered_diff[np.argsort(h_diff)[::-1][int(np.count_nonzero(h_diff) * (1 - turn_weight))]]
    return h_filtered_diff
# @tool
def CalRelAngle(scratch, destination_pos: list):
    # "Calculate the relative angle (which is in [-pi, pi], > 0 means left action, otherwise right) between the front and the destination, where parameter: scratch represents the observation history, destination_pos is the destination position"
    norm_direct = (destination_pos[:2] - scratch[-1][:2]) / np.linalg.norm(destination_pos[:2] - scratch[-1][:2])
    angle = np.sign(destination_pos[0] - scratch[-1][0]) * np.arccos(np.dot(norm_direct, np.array([0, 1])))
    rel_angle = angle - scratch[-1][-1]
    if rel_angle > np.pi:
        rel_angle = 2 * np.pi - rel_angle
    elif rel_angle < -np.pi:
        rel_angle = 2 * np.pi + rel_angle
    return rel_angle
# @tool
def CalRelHei(scratch, destination_pos: list):
    # "Calculate the relative height (> 0 means climb in next step, otherwise go down) between the current level and the destination, where destination_pos is the destination position"
    height_diff = destination_pos[-1] - scratch[-1][2]
    return np.clip(height_diff, -3000, 3000) / 3000

def CalDestPos(lat: float, log: float, hei: float):
    # "Determine the position of destination, where lat, log, hei are latitude, longitude and height, respectively"
    from utils.loadcsv import trans_from_zjenv_to_mrad
    return trans_from_zjenv_to_mrad([lat, log, hei])

def CompleteAction():
    pass

def convert_list_to_str(input_list):
    return '\n' + '\n'.join([f'{id + 1}. {con}' for id, con in enumerate(input_list)])

def convert_list_to_str_without_id(input_list):
    return '[' + ', '.join([f'{con}' for con in input_list]) + ']'

def get_function_tools():
    tools = [Straight, Down, Climb, TurnLeft, TurnRight]
    tool_names = [item.name for item in tools]
    return tools, tool_names

def get_function_tools_info():
    tools_info = [
        {
            "name": "Straight",
            "description": "find the straight trajectory from expert data",
            "args": [
                {
                    "name": "master_data",
                    "type": "str",
                    "description": "should be 'master_data', which is expert data",
                }
            ],
            "return description": "trajectory"
        },
        {
            "name": "Down",
            "description": "plan descending trajectories based on expert data",
            "args": [
                {
                    "name": "master_data",
                    "type": "str",
                    "description": "should be 'master_data', which is expert data",
                },
                {
                    "name": "turn_weight",
                    "type": "float",
                    "description": "the execution extent of this action within [0.01, 0.25], you need to decide it based on the absolute value of relative height, which within [0, 100]",
                }
            ],
            "return description": "trajectory"
        },
        {
            "name": "Climb",
            "description": "plan ascending trajectories based on expert data",
            "args": [
                {
                    "name": "master_data",
                    "type": "str",
                    "description": "should be 'master_data', which is expert data"
                },
                {
                    "name": "turn_weight",
                    "type": "float",
                    "description": "the execution extent of this action within [0.01, 0.75], you need to decide it based on the absolute value of relative height, which within [0, 100]",
                }
            ],
            "return description": "trajectory"
        },
        {
            "name": "TurnLeft",
            "description": "plan left-turning trajectories based on expert data",
            "args": [
                {
                    "name": "master_data",
                    "type": "str",
                    "description": "should be 'master_data', which is expert data"
                },
                {
                    "name": "turn_weight",
                    "type": "float",
                    "description": "the execution extent of this action within [0.01, 0.75], you need to decide it based on the absolute value of relative angle, which within [0, 314]",
                }
            ],
            "return description": "trajectory"
        },
        {
            "name": "TurnRight",
            "description": "plan right-turning trajectories based on expert data",
            "args": [
                {
                    "name": "master_data",
                    "type": "str",
                    "description": "should be 'master_data', which is expert data"
                },
                {
                    "name": "turn_weight",
                    "type": "float",
                    "description": "the execution extent of this action within [0.01, 0.75], you need to decide it based on the absolute value of relative angle, which within [0, 314]",
                }
            ],
            "return description": "trajectory"
        },
        {
            "name": "CalRelAngle",
            "description": "Calculate the relative angle between the front and the destination",
            "args": [
                {
                    "name": "scratch",
                    "type": "str",
                    "description": "should be 'scratch'"
                },
                {
                    "name": "destination_pos",
                    "type": "list, whose elements are float with 2 decimal places",
                    "description": "destination position",
                }
            ],
            "return description": "relative angle"
        },
        {
            "name": "CalRelHei",
            "description": "Calculate the relative height between the current level and the destination",
            "args": [
                {
                    "name": "scratch",
                    "type": "str",
                    "description": "should be 'scratch'"
                },
                {
                    "name": "destination_pos",
                    "type": "list, whose elements are float with 2 decimal places",
                    "description": "destination position",
                }
            ],
            "return description": "relative height"
        },
        {
            "name": "CalDestPos",
            "description": "Determine the geographical coordinates of destination",
            "args": [
                {
                    "name": "lat",
                    "type": "float",
                    "description": "latitude of destination"
                },
                {
                    "name": "log",
                    "type": "float",
                    "description": "longitude of destination"
                },
                {
                    "name": "hei",
                    "type": "float",
                    "description": "height of destination"    
                }
            ],
            "return description": "destination position"
        },
        {
            "name": "CompleteAction",
            "description": "Execute this action when all steps are completed",
            "args": [],
            "return description": "finished"
        }
    ]
    tools_map = {
        "Straight": Straight,
        "Down": Down,
        "Climb": Climb,
        "TurnLeft": TurnLeft,
        "TurnRight": TurnRight,
        "CalRelAngle": CalRelAngle,
        "CalRelHei": CalRelHei,
        "CalDestPos": CalDestPos,
        "CompleteAction": CompleteAction
    }
    return tools_info, tools_map

def gen_tools_desc(tools_info):
    import json
    tools_desc = []
    for idx, t in enumerate(tools_info):
        args_desc = []
        for info in t["args"]:
            args_desc.append({
                "name": info["name"],
                "description": info["description"],
                "type": info["type"]
            })
        args_desc = json.dumps(args_desc, ensure_ascii=False)
        tool_desc = f"{idx+1}.{t['name']}:{t['description']}, args: {args_desc}, return description: {t['return description']}"
        tools_desc.append(tool_desc)
    tools_prompt = "\n".join(tools_desc)
    return tools_prompt

class PromptData():
    def __init__(self, master_data) -> None:
        self.master_data = master_data
        self.scratch = []
        self.tools_info, self.tools_map = get_function_tools_info()
        self.tools_desc = gen_tools_desc(self.tools_info)

        self.constraint = [
            "You can only use tools in " + convert_list_to_str_without_id(list(self.tools_map.keys())),
            # "You can answer the question in several steps, include analyzing the relative position of destination and obstacles (if any), and providing corresponding actions based on relative positions",
            "The execution of each action comes with certain cost, and you need to complete the task with as few steps as possible",
            "Parameter: turn_weight must be value, cannot contain mathematical operators",
            # "String type parameters in JSON format must use quotation marks",
            "Please note that you cannot assume any intermediate results",
            # "Notice that when relative height is positive, climb and go down must be choosen, otherwise climb or go down. "
        ]
        self.resources = [
            "All observations consist of [x_position, y_position, z_position, pitch, roll, yaw], and position is [x_position, y_position, z_position]",
            "Master data: 'master_data' consists of several trajectories, each consisting of several steps. Every step is 6-dimensional vector: [x_position, y_position, z_position, pitch, roll, yaw]",
            "Your historical observations are saved in 'scratch'",
            "You are a large language model that knows a vast amount of objective facts and can use existing knowledge to answer questions",
        ]
        self.best_practices = [
            # "Continuously plan trajectories and analyze your behavior based on rewards to ensure that you maximize your abilities. If received rewards is to small, repeat planning trajectories",
            # "The execution of each action comes with certain cost, and you need to complete the task with as few steps as possible",
 
            "When given the destination city, you need to complete some preparatory work (e.g., query the latitude and longitude of destination point, assume safety height is 50000.00, so set destination_pos = [latitude, longitude, height]).",
            "When given the destination position, you need to analyze the relative angle. (Notice that current position is in 'scratch', destination position is destination_pos).",
            "When given the destination position and relative angle, you need to analyze the relative height. (Notice that current position is in 'scratch', destination position is destination_pos).",
            "When given the destination position, relative angle and relative height, you need to decide which actions to take (When the absolute value of relative angle is large, left and right always tend to be choosen, otherwise climb and go down. Furthermore, relative angle < 0 means left action, otherwise right; if relative height > 0 means climb, otherwise go down). Action name in response format must be one of " + convert_list_to_str_without_id(list(self.tools_map.keys())),
            "Once know the trajectory, you need to perform the completed action.",
        ]

        self.response_prompt = """
        {
            "thought": "describe the reasons and objectives for taking the action",
            "action": {
                "name": "action name",
                "args": {
                    "args name": "args value"
                }
                "return": "return description"
            }
        }
        """

    def get_prompt_template(self, query):
        prompt_template = """
You are a pilot and need to refer to master data to plan some trajectories in order to complete specific tasks. 
You must comply with the following restrictions: {constraint}.

In order to plan trajectories, you can use several functions as as actions with determining parameter values by yourself: {tool_desc}.

Other information you can refer is {resources}.

You need to provide corresponding answers based on different questions and context. And you only need to choose one of the following situations to answer (some of them may need more than one action): {best_practices}
And you should respond in JSON format (beginning with 'json'), with the following response format:
{response_prompt}
Ensure that the response result must contain ```json any```. Comments and '//' cannot appear in JSON format.
        """

        prompt = prompt_template.format(
            constraint = convert_list_to_str(self.constraint),
            tool_desc = self.tools_desc,
            resources = convert_list_to_str(self.resources),
            best_practices = convert_list_to_str(self.best_practices),
            response_prompt = self.response_prompt
        )

        return [{"role":"system","content":prompt}] + [{"role": "user", "content": f"Your task: {query}".format(query)}]
    
    def update_scratch(self, observation):
        self.scratch.append(observation)

    def test_visual(self):
        # from matplotlib import pyplot as plt
        # ax1 = plt.axes(projection='3d')
        # # aaa = self.train_data.total_obs_withouttoken_mrad
        # for traj in self.master_data:
        #     # print(traj)
        #     ax1.plot([item[0] for item in traj], [item[1] for item in traj], [item[2] for item in traj], color = 'grey')

        # test_traj = TurnRight(self.master_data, 0.0)
        # test_traj = Down(self.master_data, 0.1)
        # for traj_item in [TurnRight(self.master_data, wei) for wei in [0.0]]:#, 0.25, 0.5, 0.75, 1.0
        #     ax1.plot([item[0] for item in traj_item], [item[1] for item in traj_item], [item[2] for item in traj_item])
        # plt.show()

        self.scratch.append(trans_from_zjenv_to_mrad([41.0, 38.0, 5000.0, 0.0, 0.0, 0.0]))
        print(trans_from_zjenv_to_mrad([40.9833, 40.5167, 5000.0]))
        print(CalRelAngle(self.scratch, [3.40101781e+06, 7.25290568e+05, 1524.0]))