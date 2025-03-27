import base64
import dataclasses
import io
import os
import re
import string
from typing import Any
from absl import logging
from android_world.agents import infer
from android_world.env import json_action
from android_world.env import representation_utils
from IPython import display
from matplotlib.pylab import plt
import numpy as np
import PIL
import requests
# from android_world.agents.function_call_mobile_abs0_1000 import MobileUse # TODO
# print('------function_call_mobile_abs0_1000------')
# from android_world.agents.function_call_mobile import MobileUse # TODO
# print('------function_call_mobile')
from android_world.agents.function_call_mobile_answer import MobileUse # TODO
print('------function_call_mobile_with_answer-------')
# from android_world.agents.function_call_mobile_abs0_1000_answer_v13 import MobileUse # TODO
# print('------function_call_mobile_with_abs_0_1000_answer-------')
# from android_world.agents.function_call_mobile_abs0_1000_answer import MobileUse # TODO
# print('------function_call_mobile_with_abs_0_1000_answer-------')
from android_world.agents.coordinate_resize import convert_point_format

from android_world.agents.qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

import dashscope

def generate_user_prompt_single_image(instruction, history, add_info='', add_thought=True):
    user_prompt = f'''The user query: {instruction}'''
    if add_thought:
        # if len(history) > 0:
        user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
        if len(add_info) > 0:
            # user_prompt += f'\n请根据以下提示操作: {add_info}.'
            user_prompt += f'\nThe following tips can help you complete user tasks: {add_info}.'
        user_prompt += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
    return user_prompt


def generate_user_prompt_multi_image(instruction, history, add_info='', add_thought=True):
    user_prompt = f'''The user query: {instruction}'''
    if add_thought:
        # if len(history) > 0:
        # user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
        if len(add_info) > 0:
            # user_prompt += f'\n请根据以下提示操作: {add_info}.'
            user_prompt += f'\nThe following tips can help you complete user tasks: {add_info}.'
        user_prompt += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
    return user_prompt

def build_system_messages(instruction, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        # TODO
    )

    user_prompt = generate_user_prompt_single_image(instruction, history, add_info, add_thought=add_thought)

    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        ),
        Message(
            role="user",
            content=[ContentItem(text=user_prompt)],
        )
    ]

    messages = NousFnCallPrompt.preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]

    messages[0]['content'][0]['type'] = 'text'
    messages[0]['content'][1]['type'] = 'text'
    messages[1]['content'][0]['type'] = 'text'

    system_prompt_part = {'role': 'system', 'content': []} # TODO
    system_prompt_part['content'].append(
        {'text': messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']})

    user_prompt_part = {'role': 'user', 'content': []}  # user
    user_prompt_part['content'].append({'text': messages[1]['content'][0]['text']})  # 46 * 1

    return system_prompt_part, user_prompt_part

def build_system_messages_only(instruction, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        # TODO
    )

    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        )
    ]

    messages = NousFnCallPrompt.preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]

    messages[0]['content'][0]['type'] = 'text'
    messages[0]['content'][1]['type'] = 'text'

    system_prompt_part = {'role': 'system', 'content': []} # TODO
    system_prompt_part['content'].append(
        {'text': messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']})

    return system_prompt_part


def build_user_messages_multi_image_action(instruction, screenshot_file_list, response_list, multi_image_add_query, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    input_messages = []
    user_prompt_part = {'role': 'user', 'content': []}  # user

    user_prompt = generate_user_prompt_multi_image(instruction, '', add_info, add_thought=add_thought)
    user_prompt_wo_cot = user_prompt.split('\nBefore answering')[0]
    if len(response_list) > 0:
        user_prompt_part['content'].append({'type': 'text', 'text': user_prompt_wo_cot})  # 46 * 1
    else:
        user_prompt_part['content'].append({'type': 'text', 'text': user_prompt})  # 46 * 1
    user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[0]})

    input_messages.append(user_prompt_part)

    if len(response_list) > 0:
        for i in range(len(response_list)):
            assistant_prompt = {'role': 'assistant', 'content': []}
            response_temp = response_list[i]
            # TODO
            action_temp = '<tool_call>\n{"name": "mobile_use"' + response_temp.split('{"name": "mobile_use"')[1].split('}}\n')[0] + '}}\n</tool_call>'
            assistant_prompt['content'].append({'type': 'text', 'text': action_temp})
            input_messages.append(assistant_prompt)

            user_prompt_part = {'role': 'user', 'content': []}
            if multi_image_add_query:
                user_prompt_part['content'].append({'type': 'text', 'text': user_prompt_wo_cot})
            user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[i+1]})
            input_messages.append(user_prompt_part)

    input_messages[-1]['content'][0]['text'] = user_prompt

    return input_messages

def build_user_messages_two_image(instruction, screenshot_file_list, response_list, multi_image_add_query, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    input_messages = []
    user_prompt_part = {'role': 'user', 'content': []}  # user

    user_prompt = generate_user_prompt_multi_image(instruction, '', add_info, add_thought=add_thought)
    user_prompt_part['content'].append({'type': 'text', 'text': user_prompt})  # 46 * 1
    if len(screenshot_file_list) > 1:
        user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[-2]})
    else:
        user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[-1]})

    input_messages.append(user_prompt_part)

    if len(response_list) > 0:
        assistant_prompt = {'role': 'assistant', 'content': []}
        assistant_prompt['content'].append({'type': 'text', 'text': response_list[-1]})
        input_messages.append(assistant_prompt)

        user_prompt_part = {'role': 'user', 'content': []}
        if multi_image_add_query:
            user_prompt_part['content'].append({'type': 'text', 'text': user_prompt})
        user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[-1]})
        input_messages.append(user_prompt_part)

    return input_messages


def build_user_messages_multi_image(instruction, screenshot_file_list, response_list, multi_image_add_query, resized_width, resized_height, add_info='', history='', infer_mode = 'N_image_infer', add_thought=True):
    input_messages = []
    user_prompt_part = {'role': 'user', 'content': []}  # user

    user_prompt = generate_user_prompt_multi_image(instruction, '', add_info, add_thought=add_thought)
    user_prompt_part['content'].append({'type': 'text', 'text': user_prompt})  # 46 * 1
    user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[0]})

    input_messages.append(user_prompt_part)

    if len(response_list) > 0:
        for i in range(len(response_list)):
            assistant_prompt = {'role': 'assistant', 'content': []}
            assistant_prompt['content'].append({'type': 'text', 'text': response_list[i]})
            input_messages.append(assistant_prompt)

            user_prompt_part = {'role': 'user', 'content': []}
            if multi_image_add_query:
                user_prompt_part['content'].append({'type': 'text', 'text': user_prompt})
            user_prompt_part['content'].append({'type': 'image', 'image': screenshot_file_list[i+1]})
            input_messages.append(user_prompt_part)

    return input_messages




from android_world.agents.coordinate_resize import update_image_size_
from PIL import Image

def build_messages_N_image(instruction, history_images, history_actions, history_summary, history_thought, current_image, add_info='', N=1, infer_mode='N_image_infer', add_thought=True):
    # obtain base_massages

    width, height = Image.open(current_image).size
    current_image_ele = update_image_size_({'image':current_image, 'width': width, 'height':height})
    resized_width = current_image_ele['resized_width']
    resized_height = current_image_ele['resized_height']

    system_prompt_part, user_prompt_part = build_system_messages(instruction, resized_width, resized_height, add_info, infer_mode, add_thought=add_thought)
    current_step = len(history_images)

    if current_step == 0:
        user_prompt_part['content'].append({"type": "image", 'image': current_image}) # add current image
    else:
        history = ''
        for idx, his in enumerate(history_summary):
            history += 'step ' + str(idx+1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '
        user_prompt = generate_user_prompt_single_image(instruction, history, add_info, add_thought=add_thought)

        user_prompt_part['content'] = [{"type": 'text', 'text': user_prompt},{"type": "image", 'image': current_image}]


    return [system_prompt_part, user_prompt_part], current_image_ele


def call_mobile_agent_api(messages, model_name='pre-mobile_agent_7b_sft', api_key='sk-04507fb92f4249c480095126bc662828'):
    # messages = [
    #     {
    #         "role": "system",
    #         "content": [
    #             {"text": system_prompt}
    #         ]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": image},
    #             {"text": user_prompt}
    #         ]
    #     }
    # ]
    print(messages)
    vl_high_resolution_images = False
    if 'pre' in model_name:
        dashscope.base_http_api_url = 'https://poc-dashscope.aliyuncs.com/api/v1'
        dashscope.base_websocket_api_url = 'https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
        vl_high_resolution_images = True

    dashscope.api_key = api_key  #'sk-ed44a045ef5d407fb19a34071f0f2620'# 'sk-197f888ac822444d8f8be092912c123b' 'sk-04507fb92f4249c480095126bc662828'
    output = ''
    for i in range(5):
        try:
            response = dashscope.MultiModalConversation.call(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
                model=model_name,
                messages=messages,
                vl_high_resolution_images=vl_high_resolution_images,
                top_k=1,
            )
            output = response.output.choices[0].message.content[0]["text"]
        except:
            print("Network Error:")
            try:
                print(response)
            except:
                print("Request Failed")
            time.sleep(2)
        else:
            break

    return output

import time

feida_url = "https://api.claude-Plus.top/v1/chat/completions"
feida_token = "sk-59NHknvm3EcTAoFTQaDh7NSVDts9HFWfuPPUfKjcyjKcVgSy"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def inference_chat(chat, model='gpt-4o', api_url = feida_url, token = feida_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for _ in chat:
        for idx, content in enumerate(_['content']):
            if 'image' in content:
                new_content = {}
                new_content['type'] = 'image_url'
                image = content['image']
                base_64 = encode_image(image)
                new_content['image_url'] = {"url": f"data:image/jpeg;base64,{base_64}"}
                _['content'][idx] = new_content
            if 'text' in content:
                _['content'][idx]['type'] = 'text'

        data["messages"].append(_)

    while True:
        try:
            # pdb.set_trace()
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            # print(res_json)
            res_content = res_json['choices'][0]['message']['content']
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
            time.sleep(2)
        else:
            break

    return res_content

api = [
  "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjIyMDU3NCIsInBhc3N3b3JkIjoiMjIwNTc0MTIzIiwiZXhwIjoyMDI0NDcyNDU1fQ.aEnMx7bVDLS2yyfnYbeQ9HwoyVIgqzMt5ECPf1B4Jso",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJ1c2VybmFtZSI6IjM0NDMyMSIsInBhc3N3b3JkIjoiMzQ0MzIxMTIzIiwiZXhwIjoyMDQxOTE2OTcxfQ.Dk1SubEc-VnhRD1rz_G0Y7C9ltLAvVU5SYoq9rxhcxw",
  "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjM4OTUzMSIsInBhc3N3b3JkIjoiMzg5NTMxMTIzIiwiZXhwIjoyMDI2MjkzNzk4fQ.b_yUqWZ4gguMSCTrQZW3axmYcMtlhL43K50M5vAUvNQ",
  "eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjExOTYwOCIsInBhc3N3b3JkIjoiMTE5NjA4MTIzIiwiZXhwIjoyMDI3NjQ3NTc2fQ.UzU2ziq9-ZyQkmflxHZdoczocf0_ZOUfEoitNGh06Js",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJ1c2VybmFtZSI6IjQ0MzA1MyIsInBhc3N3b3JkIjoiNDQzMDUzMTIzIiwiZXhwIjoyMDQxNDkyMDEwfQ.V4Kj9pgR8XxzH_f-6qfV5GqAM-NYBOdxPMmheKVor1Q",
]



def inference_chat_alibaba(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api[-2]}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096
    }

    while True:
        try:
            # pdb.set_trace()
            response = requests.post("http://47.88.8.18:8088/api/ask", headers=headers, json=payload)
            caption_out = response.json()['data']['response']['choices'][0]['message']['content']
            # print(response.json())
            # res_content = res_json['choices'][0]['message']['content']
        except:
            print("Network Error:")
            try:
                print(response.json())
            except:
                print("Request Failed")
            time.sleep(2)
        else:
            break

    return caption_out


import json
def parse_action_output(output):
    thought = output.split('<thinking>\n')[1].split('\n</thinking>')[0]
    action = output.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
    summary = output.split('<conclusion>\n')[1].split('\n</conclusion>')[0]

    # action parse
    pred_action = json.loads(action)['arguments']
    action_type = pred_action['action']


import copy
def convert_mobile_agent_action_to_json_action(
    dummy_action, img_ele, src_format='abs_origin', tgt_format='abs_resized'
) -> json_action.JSONAction:
    """Converts a SeeActAction object to a JSONAction object.

    Args:
      action: the whole dymmay action
                  dummy_action = {
                    "name": ACTION_NAME,
                    "arguments": {
                        "action": "click",
                        "coordinate": [100, 200],
                    },
                }
      elements: UI elements.

    Returns:
      The corresponding JSONAction object.

    """
    action_type_mapping = {
      "click": json_action.CLICK,
      "terminate": json_action.STATUS,
      "answer": json_action.ANSWER, # TODO
      "long_press": json_action.LONG_PRESS,
      "type": json_action.INPUT_TEXT,
      "swipe": json_action.SWIPE,
      "wait": json_action.WAIT,
      "system_button": "system_button",
      # "OPEN APP": json_action.OPEN_APP, # TODO
    }

    x = None
    y = None
    text = None
    direction = None
    goal_status = None
    app_name = None

    result_json = {}
    arguments = dummy_action['arguments']
    try:
        action_type_org = arguments['action']
    except:
        arguments = json.loads(arguments)
        action_type_org = arguments['action']
    action_type = action_type_mapping[action_type_org]

    dummy_action_translated = copy.deepcopy({'name': 'mobile_use', 'arguments': arguments}) # dummy_action

    if action_type == json_action.INPUT_TEXT:
        text = arguments['text']

    elif action_type == json_action.SWIPE:
        start_x, start_y = arguments['coordinate']
        end_x, end_y = arguments['coordinate2']
        start_x, start_y = convert_point_format([start_x, start_y], img_ele, src_format=src_format, tgt_format=tgt_format)
        end_x, end_y = convert_point_format([end_x, end_y], img_ele, src_format=src_format, tgt_format=tgt_format)

        dummy_action_translated['arguments']['coordinate'] = [start_x, start_y]
        dummy_action_translated['arguments']['coordinate2'] = [end_x, end_y]

        direction = [start_x, start_y, end_x, end_y]
        # direction = _swipe_to_scroll(arguments['coordinate'], arguments['coordinate2'])

    elif action_type == json_action.CLICK:
        x, y = arguments['coordinate']
        x, y = convert_point_format([x, y], img_ele, src_format=src_format, tgt_format=tgt_format)
        dummy_action_translated['arguments']['coordinate'] = [x, y]

    elif action_type == json_action.LONG_PRESS:
        x, y = arguments['coordinate']
        x, y = convert_point_format([x, y], img_ele, src_format=src_format, tgt_format=tgt_format)
        dummy_action_translated['arguments']['coordinate'] = [x, y]

    # elif action_type == json_action.OPEN_APP:  # TODO
    #   app_name = action.value

    elif action_type == json_action.ANSWER: # TODO
        text = arguments['text']

    elif action_type == json_action.STATUS:
        goal_status = "task_complete"

    elif action_type == 'system_button':
        if arguments['button'] == 'Back':
            action_type = json_action.NAVIGATE_BACK
        if arguments['button'] == 'Home':
            action_type = json_action.NAVIGATE_HOME
        if arguments['button'] == 'Enter':
            action_type = json_action.KEYBOARD_ENTER

    return json_action.JSONAction(
          action_type=action_type,
          x=x,
          y=y,
          text=text,
          direction=direction,
          goal_status=goal_status,
          app_name=app_name,
      ), dummy_action_translated

def _swipe_to_scroll(start, end):
    x1, y1 = start
    x2, y2 = end

    # 计算水平和垂直的距离
    delta_x = x2 - x1
    delta_y = y2 - y1

    # 判断是水平还是垂直滑动
    if abs(delta_x) > abs(delta_y): # TODO  可能是反的
        # 水平滑动
        if delta_x > 0:
            return 'right'
        else:
            return 'left'
    else:
        # 垂直滑动
        if delta_y > 0:
            return "down"
        else:
            return "up"

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
def save_tra(screenshot, thought, action, summary, goal, step_idx, image_save_path):
    original_image = Image.open(screenshot)
    width, height = original_image.size

    # 创建一个新的图像，增加一个空白区域在顶部
    new_height = height + 500  # 增加500像素的空白区域
    new_image = Image.new('RGB', (width, new_height), (255, 255, 255))
    new_image.paste(original_image, (0, 500))

    # 准备绘图对象
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 40)
    # font = ImageFont.load_default() #  '/System/Library/Fonts/Supplemental/Arial.ttf'

    # 将文本分成每行10个单词
    text = f"Thought: {thought}" + '\n' + f"Operation: {action}" + '\n'+ f"Summary: {summary}"
    words = text.split()
    lines = []
    for i in range(0, len(words), 10):
        line = ' '.join(words[i:i + 10])
        lines.append(line)

    # 绘制文本
    text_y = 10  # 起始文本y坐标
    for line in lines:
        draw.text((40, text_y), line, font=font, fill=(0, 0, 0))
        text_y += 50  # 行高

    # 在图片顶部下方绘制一个点，代表坐标（x, y）
    action_arg = action['arguments']
    if 'coordinate' in action_arg:
        x, y = action_arg['coordinate']
        y = y + 500
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0), outline=(0, 0, 0))

    if 'coordinate2' in action_arg:
        x, y = action_arg['coordinate2']
        y = y + 500
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(0, 255, 0), outline=(0, 0, 0))

    # # 设置字体大小和路径
    # # （需要确保你的系统中存在该字体，也可以选择其他字体）
    # font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    # font_size = 40
    # font = ImageFont.truetype(font_path, font_size)

    goal = goal.replace(' ', '_').replace(':','_').replace('.','_').replace('/','_')[:250]
    try:
        folder_path = f'{image_save_path}/{goal}/'
        os.mkdir(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

    # 保存最终图像
    new_image.save(folder_path + str(step_idx) + '.jpg')

