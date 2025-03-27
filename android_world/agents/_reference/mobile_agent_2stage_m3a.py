# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SeeAct agent for Android."""
import time
from typing import Any

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents import mobile_agent_utils_new as mobile_agent_utils
from android_world.env import actuation_mobileagent as actuation
from android_world.env import interface
from android_world.env import json_action
from PIL import Image
import uuid
import json
import pprint
import os
from android_world.agents.seeact_qwen import SEEACT_CHOICE_PROMPT_DICT
from android_world.agents import seeact_utils

from android_world.agents.coordinate_resize import convert_point_format
from android_world.agents.coordinate_resize import update_image_size_


from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

m3a_prompt = '''Instruction
You are an agent who can operate an Android phone on behalf of a user. Based on user’s goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks ”What is my schedule for today?”.
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.
When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the action in the following list (action description followed by the JSON format) by outputing the action in the correct JSON format.
- If you think the task has been completed, finish the task by using the status action with complete as goal status: ‘{”action type”: ”status”, ”goal status”: ”complete”}
- If you think the task is not feasible (including cases like you don’t have enough informa- tion or can not perform some necessary actions), finish by using the ‘status‘ action with infeasible as goal status: ‘{”action type”: ”status”, ”goal status”: ”infeasible”}
- Answer user’s question: ‘{”action type”: ”answer”, ”text”: ”answer text”}
- Click/tap on an element on the screen. Please describe the element you want to click using natural language. ‘{”action type”: ”click”, ”target”: target element description}‘. - Long press on an element on the screen, similar with the click action above, use the semantic description to indicate the element you want to long press: ‘{”action type”: ”long press”, ”target”: target element description}.
- Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the semantic de- scription to indicate the target text field: ‘{”action type”: ”input text”, ”text”: text input, ”target”: target element description}
- Press the Enter key: ‘{”action type”: ”keyboard enter”}
- Navigate to the home screen: ‘{”action type”: ”navigate home”}
- Navigate back: ‘{”action type”: ”navigate back”}
- Scroll the screen or a scrollable UI element in one of the four directions, use the same semantic description as above if you want to scroll a specific UI element, leave it empty when scroll the whole screen: ‘{”action type”: ”scroll”, ”direction”: up, down, left, right, ”element”: optional target element description}
- Open an app (nothing will happen if the app is not installed): ‘{”action type”: ”open app”, ”app name”: name}
- Wait for the screen to update: ‘{”action type”: ”wait”}

Guidelines
Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn’t (you can see that from the history), SWITCH to other solutions.
- Sometimes you may need to navigate the phone to gather information needed to com- plete the task, for example if user asks ”what is my schedule tomorrow”, then you may want to open the calendar app (using the ‘open app‘ action), look up information there, answer user’s question (using the ‘answer‘ action) and finish (using the ‘status‘ action with complete as goal status).
- For requests that are questions (or chat messages), remember to use the ‘answer‘ action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like ”show me ...”).
- If the desired state is already achieved (e.g., enabling Wi-Fi when it’s already on), you can just complete the task.
Action Related:
- Use the ‘open app‘ action whenever you want to open an app (nothing will happen if the app is not installed), do not use the app drawer to open an app unless all other ways have failed.
- Use the ‘input text‘ action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For ‘click‘, ‘long press‘ and ‘input text‘, the target element description parameter you choose must based on a VISIBLE element in the screenshot.
- Consider exploring the screen by using the ‘scroll‘ action with different directions to reveal additional content.
- The direction parameter for the ‘scroll‘ action can be confusing sometimes as it’s op- posite to swipe, for example, to view content at the bottom, the ‘scroll‘ direction should be set to ”down”. It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.
Text Related Operations:
- Normally to select certain text on the screen: (i) Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like ‘copy‘, ‘paste‘, ‘select all‘, etc. (ii) Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the ‘select all‘ button in the bar.
- At this point, you don’t have the ability to drag something around the screen, so in general you can not select arbitrary text.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the ‘copy‘ button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a ‘paste‘ button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
'''
import base64
from datetime import datetime

def base64_encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Mobile_Agent(base_agent.EnvironmentInteractingAgent):
  """mobile agent for Android."""

  def __init__(self, env: interface.AsyncEnv, test_model_name, test_model_version, suite_family, src_format, two_stage, add_info_4o, api_key, multi_image, multi_image_add_query, two_image, name: str = "Mobile_Agent"):
    super().__init__(env, name)
    self._actions = []
    self._screenshots = []
    self._summarys = []
    self._thoughts = []
    self.output_result = {}
    self.additional_guidelines = None
    self.add_info = ''
    self.image_N = 0 # 0, 1
    self.model_name = test_model_name.value #'pre-qwen2.5vl-329-agent-v1-0112-model'
        #'pre-qwen2.5vl-329-agent-v1-0112-model'#'pre-qwen2.5vl-329-agent-zhangxi-0114-model' #'' #'pre-mobile_agent_7b_2'# 'pre-mobile_agent_7b_2' #'pre-mobile_agent_7b_sft' #'qwen-vl-max-0809' 'qwen-vl-plus-0809' 'pre-mobile_agent_7b_2'
    self.model_version = test_model_version.value
    self.now_task = suite_family.value

    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    self.image_save_path = f'screenshot_all/screenshot_{self.now_task}_{self.model_version}_{self.model_name}_{time_str}'

    self.add_thought = True
    self.secs = 2
    self._text_actions = []
    self.src_format = src_format.value #'abs_resized'#'qwen-vl'
    self.two_stage_with_4o = two_stage.value

    self.whether_add_info_4o = add_info_4o.value
    self.api_key = api_key.value

    self.multi_image = multi_image.value
    self.multi_image_add_query = multi_image_add_query.value
    self.two_image = two_image.value

    print(self.model_name, self.model_version, self.now_task, self.image_save_path, self.src_format, self.two_stage_with_4o, self.whether_add_info_4o)

    self.output_list = []
    self._response = []


  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self._actions.clear()
    self._text_actions.clear()
    self._screenshots.clear() # TODO
    self._summarys.clear()
    self._thoughts.clear()
    self._response.clear()

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def step(
      self, goal: str, verbose: bool = True
  ) -> base_agent.AgentInteractionResult:
    result = {
        "ui_elements": None,
        "screenshot": None,
        "actionable_elements": None,
        "action_gen_payload": None,
        "action_gen_response": None,
        "action_ground_payload": None,
        "action_ground_response": None,
        "seeact_action": None,
        "action": None,
        "action_description": None,
    }

    fc_add_info = ''#'Wi-Fi and Bluetooth and be set by swipe down from top of the screen to access the quick settings panel containing .'#'在输出action为answer完之后，请直接结束任务，即terminate'
    # if 'Turn brightness to the max value' in goal:
    #     fc_add_info = 'When turning, use \{\"action\": \"swipe\", \"coordinate\": [98, 212], \"coordinate2\": [1050, 212]\}'
    #
    # if 'Turn brightness to the min value' in goal:
    #     fc_add_info = 'Open menu. When turning, first click on Brightness level, then use \{\"action\": \"swipe\", \"coordinate\": [1000, 212], \"coordinate2\": [50, 212]\} to make the level 0%'


    if self.whether_add_info_4o and self.now_task == 'android_world':
        if 'open the file' in goal.lower() and 'html' in goal.lower() and 'chrome' in goal.lower():
            self.add_info = '先点击html文件的左下角来选中文件，再点击三个点的按钮来用chrome打开'

        if 'record an audio clip and save it with name' in goal.lower():
            self.add_info = '滑动来查看更多app。'
            self.add_info += '先长按键盘的删除键，直到输入框为空后，再输入新文件名'

        '''
        if 'select' in goal.lower() and 'as the date' in goal.lower():
            self.add_info = '在选择日期时，请关注月份是否正确，如果不正确，请点击月份旁边的箭头更换月份。'
            self.add_info += '在选择日期时，请根据指令要求选择日期，不要直接选择12/12/2016。'
            self.add_info += '当选择日期与指令要求的日期一致时，再点击submit'
        if 'Enter' in goal and 'as the date' in goal.lower():
            self.add_info = '如果需要修改年份，可以点击左上角的年份，然后上下滑动选择日期'
        if 'angle' in goal.lower():
            self.add_info = '构建连线时，可以点击两个实心黑点的连线的中点'
        # if 'book' in goal.lower() and 'flight from' in goal.lower():
        #     self.add_info = '可以按如下步骤操作：点击第一个输入框，输入文字，system_button enter；点击第二个输入框，输入文字，system_button enter；选择日期（可以点击月份旁边的箭头更换月份），点击提交'
            # self.add_info = '1. 点击出发地输入框 2. 输入出发地 3. 点击目的地输入框 4. 输入目的地 5. 在选择日期时，可以点击月份旁边的箭头更换月份'
        if 'Select' in goal and 'click Submit' in goal.lower():
            self.add_info = 'tap the checkbox'
        if 'pie menu below' in goal:
            self.add_info = 'after tap the pie menu, click on the corresponding element'
            self.secs = 2
        if 'Select all the shades of ' in goal:
            self.add_info = '按从上至下的顺序点击所有与指令中的颜色相接近的色块，色块有黑边就代表已经选中，然后点击submit'
        if 'Copy the text' in goal:
            self.add_info = '请按如下动作执行：点击文本，点击绿色球状按钮，点击select all，点击Copy，long_press空白输入框，点击Paste'
        if 'drag the' in goal.lower():
            self.add_info = 'use swipe, do not use click.'
        # if 'Find the email by'.lower() in goal.lower():
        #     self.add_info = '当找不到某人的email时，可以在文字部分上下滑动来查看更多'
        if 'password' in goal.lower() and 'into both text fields' in goal.lower():
            self.add_info = '可以按如下步骤操作：点击第一个输入框，输入文字，点击第二个输入框，输入文字，点击提交'
        '''
    state = self.get_post_transition_state()
    result["ui_elements"] = state.ui_elements

    result["screenshot"] = state.pixels # array
    screenshot = Image.fromarray(state.pixels)
    screenshot_uuid = str(uuid.uuid4())
    if not os.path.exists(self.image_save_path):
        os.mkdir(self.image_save_path)
    screenshot_file = f'{self.image_save_path}/screenshot_{screenshot_uuid}.png'
    screenshot.save(screenshot_file)
    self._screenshots.append(screenshot_file)

    if self.two_stage_with_4o:
        history = ''
        for i, summary in enumerate(self._text_actions):
            if summary is not None:
                history += 'Step ' + str(i + 1) + '- ' + summary
        sys_prompt = m3a_prompt + f'\nThe current user goal/request is: {goal}\n\n'
        if len(history) == 0:
            history = 'You just started, no action has been performed yet.'

        sys_prompt += f'Here is a history of what you have done so far:\n{history}\n\n'
        if len(self.add_info)>0:
            sys_prompt += 'GUIDANCE\n'
            sys_prompt += f'{self.add_info}'

        print('add info', self.add_info)
        base64_image = base64_encode_image(screenshot_file)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {
                        # "type": "image",
                        # "image": screenshot_file,
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            },
        ]


        result["action_gen_payload"] = messages

        # action generation
        action_instruction_4o = mobile_agent_utils.inference_chat_alibaba(messages)

        # print(action_instruction)
        # action_instruction = action_instruction.split('\n')[-1].split(':')[-1].replace('*', '')
        if '```json' in action_instruction_4o:
            action_instruction_4o = action_instruction_4o.split('```json')[-1].replace('```', '')
        action_instruction_4o = json.loads(action_instruction_4o)

        action_instruction = ''
        for key in action_instruction_4o.keys():
            action_instruction += action_instruction_4o[key] + ' '
        print('****4o*****', action_instruction)
        stage2_user_prompt = f'{action_instruction}.'
        stage2_history = ''
        if 'long press' in stage2_user_prompt.lower():
            fc_add_info += '; long press应该使用long_press动作'
    else:
        stage2_history = ''
        for idx, his in enumerate(self._summarys):
            if his is not None:
                stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '
        stage2_user_prompt = goal

    # grounding generation
    width, height = Image.open(screenshot_file).size
    current_image_ele = update_image_size_({'image': screenshot_file, 'width': width, 'height': height})
    resized_width = current_image_ele['resized_width']
    resized_height = current_image_ele['resized_height']

    print('fc add info', fc_add_info)
    # user_prompt_part['content'].insert(0, {"type": "image", 'image': screenshot_file} ) # TODO

    if self.multi_image:
        system_prompt_part = mobile_agent_utils.build_system_messages_only(stage2_user_prompt,
                                                                                        resized_width, resized_height,
                                                                                        fc_add_info, stage2_history)

        if self.two_image:
            user_prompt_part = mobile_agent_utils.build_user_messages_two_image(stage2_user_prompt,
                                                                                         self._screenshots,
                                                                                         self._response,
                                                                                         self.multi_image_add_query,
                                                                                         resized_width,
                                                                                         resized_height,
                                                                                         fc_add_info, stage2_history)
        else:
            user_prompt_part = mobile_agent_utils.build_user_messages_multi_image_action(stage2_user_prompt, self._screenshots,
                                                                                  self._response,
                                                                                  self.multi_image_add_query, resized_width,
                                                                                  resized_height,
                                                                                  fc_add_info, stage2_history)

        # user_prompt_part = mobile_agent_utils.build_user_messages_multi_image(stage2_user_prompt, self._screenshots, self._response, self.multi_image_add_query, resized_width, resized_height,
        #                                                                                 fc_add_info, stage2_history)

        action_response = mobile_agent_utils.call_mobile_agent_api([system_prompt_part] + user_prompt_part, model_name=self.model_name, api_key=self.api_key)
    else:
        system_prompt_part, user_prompt_part = mobile_agent_utils.build_system_messages(stage2_user_prompt,
                                                                                        resized_width, resized_height,
                                                                                        fc_add_info, stage2_history)
        user_prompt_part['content'].append({"type": "image", 'image': screenshot_file})

        action_response = mobile_agent_utils.call_mobile_agent_api([system_prompt_part, user_prompt_part], model_name=self.model_name, api_key=self.api_key)

    if self.two_stage_with_4o and 'long press' in stage2_user_prompt.lower():
        action_response = action_response.replace('click', 'long_press')
        print('后处理！')

    if self.two_stage_with_4o:
        result["output_dict"] = {'goal':goal, 'screenshot': screenshot_file, 'step_instruction': action_instruction, 'cot_output': action_response}
    else:
        result["output_dict"] = {'goal': goal, 'screenshot': screenshot_file, 'cot_output': action_response}
    result["action_response"] = action_response
    print('========== action_response ==========')
    pprint.pprint(action_response)

    # save result
    if goal in self.output_result:
        self.output_result[goal].append({'image': screenshot_file, 'pred_response': action_response})
    else:
        self.output_result[goal] = [{'image': screenshot_file, 'pred_response': action_response}]

    dummy_action = None
    thought = None
    summary = None
    try:
      if self.add_thought:
        if 'qwen-vl-' in self.model_name: # qwen-vl-max不返回<conclusion></conclusion>
          thought = action_response.split('<thinking>')[1].split('</thinking>')[0].strip('\n')
          dummy_action = action_response.split('<tool_call>')[1].split('</tool_call>')[0].strip('\n')
          summary = None
        else:
          thought = action_response.split('<thinking>\n')[1].split('\n</thinking>')[0]
          # dummy_action = action_response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
          dummy_action = '{"name": "mobile_use"' + action_response.split('{"name": "mobile_use"')[1].split('}}\n')[0] + '}}'
          summary = action_response.split('<conclusion>\n')[1].split('\n</conclusion>')[0]
      else:
        # dummy_action = action_response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
        dummy_action = '{"name": "mobile_use"' + action_response.split('{"name": "mobile_use"')[1].split('}}\n')[
            0] + '}}'
        thought = None
        summary = None

      dummy_action = json.loads(dummy_action.replace('\'', '"'))

      dummy_action['arguments']['action'] = dummy_action['arguments']['action'].replace('tap', 'click')

      # if dummy_action['arguments']['action'] == 'answer':
      #     self.env.interaction_cache = dummy_action['arguments']['text']

      # TODO --------
      if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
          dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
          self.env.interaction_cache =  self._actions[-1]['arguments']['text']


      action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action( # TODO action 解析转换
          dummy_action, current_image_ele, src_format=self.src_format, tgt_format='abs_origin'
      )


      result["dummy_action"] = dummy_action
      result["dummy_action_translated"] = dummy_action_translated
      result["action"] = action
    except seeact_utils.ParseActionError as e:
      action_description = f"No Operation with error: {e}"
      action = json_action.JSONAction(action_type=json_action.UNKNOWN)
      result["seeact_action"] = None
      result["action"] = action
    except:
        dummy_action = json.loads(action_response) # 没输出thinking, summary
        action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
            # TODO action 解析转换
            dummy_action, current_image_ele, src_format=self.src_format, tgt_format='abs_origin'
        )
        result["dummy_action"] = dummy_action
        result["dummy_action_translated"] = dummy_action_translated
        result["action"] = action
    else:
      # target_element = seeact_utils.get_referred_element(
      #     seeact_action, actionable_elements
      # )
      # action_description = seeact_utils.generate_action_description(
      #     seeact_action, target_element
      # )

      actuation.execute_adb_action(
          action,
          # [e.ui_element for e in actionable_elements],
          [],
          self.env.logical_screen_size,
          self.env.controller,
          self.secs
      )

        # TODO : 可以添加element描述 / Set of Mark
        # action_description = seeact_utils.generate_action_description(
        #     seeact_action, target_element
        # )
        # result["action_description"] = action_description

      if 'qwen-vl' in self.model_name and 'pre' not in self.model_name:
        self._text_actions.append(action_instruction)
      else:
        self._text_actions.append(summary) #(action_instruction)
      self._actions.append(dummy_action) # action_description
      self._summarys.append(summary)
      self._thoughts.append(thought)
      self._response.append(action_response)

      mobile_agent_utils.save_tra(screenshot_file, thought, dummy_action_translated, summary, goal, len(self._actions)-1, self.image_save_path)

    # if verbose:
    #   print("=" * 80)
    #   (
    #       seeact_utils.display_prompt(
    #           result["action_ground_payload"],
    #           extra_text="\n\n~~~~~~~~~ANSWER~~~~~~~~~:"
    #           + action_description
    #           + "\n\n",
    #       )
    #   )
    #   print("=" * 80)

    return base_agent.AgentInteractionResult(
        done=action.action_type == json_action.STATUS,
        data=result,
    )
