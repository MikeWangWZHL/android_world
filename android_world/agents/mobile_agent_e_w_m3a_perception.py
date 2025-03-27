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

"""A Multimodal Autonomous Agent for Android (M3A)."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


# INIT_TIPS = (
#     'Here are some useful guidelines you need to follow:\n'
#     'General:\n'
#     '- Usually there will be multiple ways to complete a task, pick the'
#     ' easiest one. Also when something does not work as expected (due'
#     ' to various reasons), sometimes a simple retry can solve the problem,'
#     " but if it doesn't (you can see that from the history),"
#     ' SWITCH to other solutions.\n'
#     '- Sometimes you may need to navigate the phone to gather information'
#     ' needed to complete the task, for example if user asks'
#     ' "what is my schedule tomorrow", then you may want to open the calendar'
#     ' app (using the `open_app` action), look up information there, answer'
#     " user's question (using the `answer` action) and finish (using"
#     ' the `status` action with complete as goal_status).\n'
#     '- For requests that are questions (or chat messages), remember to use'
#     ' the `answer` action to reply to user explicitly before finish!'
#     ' Merely displaying the answer on the screen is NOT sufficient (unless'
#     ' the goal is something like "show me ...").\n'
#     '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
#     " it's already on), you can just complete the task.\n"
#     'Action Related:\n'
#     '- Use the `open_app` action whenever you want to open an app'
#     ' (nothing will happen if the app is not installed), do not use the'
#     ' app drawer to open an app unless all other ways have failed.\n'
#     '- Use the `input_text` action whenever you want to type'
#     ' something (including password) instead of clicking characters on the'
#     ' keyboard one by one. Sometimes there is some default text in the text'
#     ' field you want to type in, remember to delete them before typing.\n'
#     '- For `click`, `long_press` and `input_text`, the index parameter you'
#     ' pick must be VISIBLE in the screenshot and also in the UI element'
#     ' list given to you (some elements in the list may NOT be visible on'
#     ' the screen so you can not interact with them).\n'
#     '- Consider exploring the screen by using the `scroll`'
#     ' action with different directions to reveal additional content.\n'
#     '- The direction parameter for the `scroll` action can be confusing'
#     " sometimes as it's opposite to swipe, for example, to view content at the"
#     ' bottom, the `scroll` direction should be set to "down". It has been'
#     ' observed that you have difficulties in choosing the correct direction, so'
#     ' if one does not work, try the opposite as well.\n'
#     'Text Related Operations:\n'
#     '- Normally to select certain text on the screen: <i> Enter text selection'
#     ' mode by long pressing the area where the text is, then some of the words'
#     ' near the long press point will be selected (highlighted with two pointers'
#     ' indicating the range) and usually a text selection bar will also appear'
#     ' with options like `copy`, `paste`, `select all`, etc.'
#     ' <ii> Select the exact text you need. Usually the text selected from the'
#     ' previous step is NOT the one you want, you need to adjust the'
#     ' range by dragging the two pointers. If you want to select all text in'
#     ' the text field, simply click the `select all` button in the bar.\n'
#     "- At this point, you don't have the ability to drag something around the"
#     ' screen, so in general you can not select arbitrary text.\n'
#     '- To delete some text: the most traditional way is to place the cursor'
#     ' at the right place and use the backspace button in the keyboard to'
#     ' delete the characters one by one (can long press the backspace to'
#     ' accelerate if there are many to delete). Another approach is to first'
#     ' select the text you want to delete, then click the backspace button'
#     ' in the keyboard.\n'
#     '- To copy some text: first select the exact text you want to copy, which'
#     ' usually also brings up the text selection bar, then click the `copy`'
#     ' button in bar.\n'
#     '- To paste text into a text box, first long press the'
#     ' text box, then usually the text selection bar will appear with a'
#     ' `paste` button in it.\n'
#     '- When typing into a text field, sometimes an auto-complete dropdown'
#     ' list will appear. This usually indicating this is a enum field and you'
#     ' should try to select the best match by clicking the corresponding one'
#     ' in the list.\n'
# )

INIT_TIPS = (
    'General:\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Generate a description for a given UI element with important information.

  Args:
    ui_element: UI elements for the current screen.
    index: The numeric index for the UI element.

  Returns:
    The description for the UI element.
  """
  element_description = f'UI element {index}: {{"index": {index}, '
  if ui_element.text:
    element_description += f'"text": "{ui_element.text}", '
  if ui_element.content_description:
    element_description += (
        f'"content_description": "{ui_element.content_description}", '
    )
  if ui_element.hint_text:
    element_description += f'"hint_text": "{ui_element.hint_text}", '
  if ui_element.tooltip:
    element_description += f'"tooltip": "{ui_element.tooltip}", '
  element_description += (
      f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
  )
  element_description += (
      '"is_long_clickable":'
      f' {"True" if ui_element.is_long_clickable else "False"}, '
  )
  element_description += (
      f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
  )
  if ui_element.is_scrollable:
    element_description += '"is_scrollable": True, '
  if ui_element.is_focusable:
    element_description += '"is_focusable": True, '
  element_description += (
      f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
  )
  element_description += (
      f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
  )
  return element_description[:-2] + '}'

def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate concise information for a list of UIElement.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: The height and width of the screen in pixels.

  Returns:
    Concise information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += _generate_ui_element_description(ui_element, index) + '\n'
  return tree_info


import copy
import json
from android_world.agents.mobile_agent_e_w_m3a_perception_agents import (
  InfoPool, 
  Manager, 
  Executor, 
  Notetaker, 
  ActionReflector
)
from dataclasses import dataclass, field, asdict
from PIL import Image

class MobileAgentE_M3A(base_agent.EnvironmentInteractingAgent):
  """Mobile Agent E wrapper for Android World."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = 'MobileAgentE_M3A',
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes a MobileAgentE_M3A Agent.

    Args:
      env: The environment.
      llm: The multimodal LLM wrapper.
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stablize
        after executing an action
    """
    super().__init__(env, name)
    self.llm = llm
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds
    
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    
    # init info pool
    self.info_pool = InfoPool(
      additional_knowledge=copy.deepcopy(INIT_TIPS),
      err_to_manager_thresh=2
    )

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()

    # init info pool again on reset
    self.info_pool = InfoPool(
      additional_knowledge=copy.deepcopy(INIT_TIPS),
      err_to_manager_thresh=2
    )

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    ## init agents ## 
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    
    self.info_pool.instruction = goal
    step_idx = len(self.info_pool.action_history)

    print('----------step ' + str(step_idx + 1))

    ## perception ###
    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    before_ui_elements_list = _generate_ui_elements_description_list(
        before_ui_elements, logical_screen_size
    )
    raw_screenshot = state.pixels.copy()
    before_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(before_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
    before_screenshot_with_som = before_screenshot.copy()

    self.info_pool.ui_elements_list_before = before_ui_elements_list
    

    
    ###############
    ### manager ###
    ###############
    
    ## check error escalation
    self.info_pool.error_flag_plan = False
    err_to_manager_thresh = self.info_pool.err_to_manager_thresh
    if len(self.info_pool.action_outcomes) >= err_to_manager_thresh:
      # check if the last err_to_manager_thresh actions are all errors
      latest_outcomes = self.info_pool.action_outcomes[-err_to_manager_thresh:]
      count = 0
      for outcome in latest_outcomes:
          if outcome in ["B", "C"]:
              count += 1
      if count == err_to_manager_thresh:
          self.info_pool.error_flag_plan = True


    ## if previous action is invalid, skip the manager and try again first ##
    skip_manager = False
    if not self.info_pool.error_flag_plan and len(self.info_pool.action_history) > 0:
      if self.info_pool.action_history[-1]['action_type'] == 'invalid':
        skip_manager = True
      
    if not skip_manager:
      print("\n### Manager ... ###\n")
      planning_start_time = time.time()
      prompt_planning = manager.get_prompt(self.info_pool)
      output_planning, is_safe, raw_response = self.llm.predict_mm(
          prompt_planning,
          [raw_screenshot], # original screenshot      
      )
      parsed_result_planning = manager.parse_response(output_planning)
      self.info_pool.plan = parsed_result_planning['plan']
      self.info_pool.current_subgoal = parsed_result_planning['current_subgoal']
      planning_end_time = time.time()
      if not raw_response:
        raise RuntimeError('Error calling LLM in planning phase.')
      
      print('Planning prompt: ' + prompt_planning)
      print()
      print('Plan: ' + self.info_pool.plan)
      print('Current subgoal: ' + self.info_pool.current_subgoal)
      print('Planning thought: ' + parsed_result_planning['thought'], "\n")
      Image.fromarray(raw_screenshot).save("screenshots/manager_input.png")
      import pdb; pdb.set_trace()


    ## if stopping by planner ##
    if "Finished" in self.info_pool.current_subgoal.strip():
      self.info_pool.finish_thought = parsed_result_planning['thought']
      action_thought = "Finished by planner"
      action_object_str = "{\"action_type\": \"status\", \"goal_status\": \"complete\"}"
      action_description = "Finished by planner"
    
    else:

      ################
      ### Operator ###
      ################

      print("\n### Operator ... ###\n")
      action_decision_start_time = time.time()
      prompt_action = executor.get_prompt(self.info_pool)
      output_action, is_safe, raw_response = self.llm.predict_mm(
          prompt_action,
          [before_screenshot_with_som], # annotated screenshot
      )
      if not raw_response:
        raise RuntimeError('Error calling LLM in operator phase.')
      parsed_result_action = executor.parse_response(output_action)
      action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']
      action_decision_end_time = time.time()
      
      if is_safe == False:  # pylint: disable=singleton-comparison
        #  is_safe could be None
        action_thought = "Finish due to safety classifier"
        action_object_str = "{\"action_type\": \"status\", \"goal_status\": \"infeasible\"}"
        action_description = "Finish due to safety classifier"

      self.info_pool.last_action_thought = action_thought
      self.info_pool.last_summary = action_description

      # If the output is not in the right format, add it to step summary which
      # will be passed to next step and return.
      if (not action_thought) or (not action_object_str):
        print('Action prompt output is not in the correct format.')
        self.info_pool.last_action = {"action_type": "invalid"}
        self.info_pool.action_history.append({"action_type": "invalid"})
        self.info_pool.summary_history.append(action_description)
        self.info_pool.action_outcomes.append("C") # no change
        self.info_pool.error_descriptions.append("invalid action format, do nothing.")
        return base_agent.AgentInteractionResult(
            False,
            asdict(self.info_pool),
        )
      print("Prompt: " + prompt_action)
      print()
    
    print('Thought: ' + action_thought)
    print('Action: ' + action_object_str)
    print('Action description: ' + action_description)
    Image.fromarray(before_screenshot_with_som).save("screenshots/operator_input.png")
    import pdb; pdb.set_trace()

    ## parse action ##
    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action_object_str),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      self.info_pool.last_action = {"action_type": "invalid"}
      self.info_pool.action_history.append({"action_type": "invalid"})
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append("C") # no change
      self.info_pool.error_descriptions.append("invalid action format, do nothing.")
      return base_agent.AgentInteractionResult(
          False,
          asdict(self.info_pool),
      )

    ## double check action is valid ##
    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)
    if (
        converted_action.action_type
        in ['click', 'long_press', 'input_text', 'scroll']
        and action_index is not None
    ):
      if action_index >= num_ui_elements:
        print(
            f'Index out of range, prediction index is {action_index}, but the'
            f' UI element list only has {num_ui_elements} elements.'
        )
        self.info_pool.last_action = {"action_type": "invalid"}
        self.info_pool.action_history.append({"action_type": "invalid"})
        self.info_pool.summary_history.append(action_description)
        self.info_pool.action_outcomes.append("C") # no change
        self.info_pool.error_descriptions.append(f"invalid action due to UI element index out of range: got {action_index}, expected < {num_ui_elements}; do nothing.")
        return base_agent.AgentInteractionResult(
            False,
            asdict(self.info_pool),
        )

      # Add mark to the target element.
      m3a_utils.add_ui_element_mark(
          raw_screenshot,
          before_ui_elements[action_index],
          action_index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )

    if converted_action.action_type == 'status':
      outcome = "A"
      error_description = "None"
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
        outcome = "C"
        error_description = "Agent stopped since it thinks mission impossible."
      self.info_pool.last_action = json.loads(converted_action.json_str())
      self.info_pool.action_history.append(json.loads(converted_action.json_str()))
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append(outcome) # no change
      self.info_pool.error_descriptions.append(error_description)
      return base_agent.AgentInteractionResult(
          True,
          asdict(self.info_pool),
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    ## try execute action ##
    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to execute action.')
      print(str(e))
      self.info_pool.last_action = json.loads({"action_type": "invalid"})
      self.info_pool.action_history.append({"action_type": "invalid"})
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append("C") # no change
      if converted_action.action_type == "open_app":
        app_name = converted_action.app_name
        self.info_pool.error_descriptions.append(f"Failed to open the app '{app_name}'; the app name might not exist.")
      else:
        self.info_pool.error_descriptions.append(f"Failed to execute the action: {converted_action}")
      return base_agent.AgentInteractionResult(
          False,
          asdict(self.info_pool),
      )
    print("Done action execution.\n")
    self.info_pool.last_action = json.loads(converted_action.json_str())

    time.sleep(self.wait_after_action_seconds)

    
    ### Perception after execution ###

    state = self.env.get_state(wait_to_stabilize=False)
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    after_ui_elements = state.ui_elements
    after_ui_elements_list = _generate_ui_elements_description_list(
        after_ui_elements, logical_screen_size
    )
    after_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            after_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    m3a_utils.add_screenshot_label(before_screenshot_with_som, 'before')
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    after_screenshot_with_som = after_screenshot.copy()
    
    self.info_pool.ui_elements_list_after = after_ui_elements_list

    #################
    ### Reflector ###
    #################
    print("\n### Action Reflector ... ###\n")
    if converted_action.action_type != 'answer':
      action_reflection_start_time = time.time()
      prompt_action_reflect = action_reflector.get_prompt(self.info_pool)
      output_action_reflect, if_safe, raw_response = self.llm.predict_mm(
          prompt_action_reflect,
          [
            before_screenshot_with_som,
            after_screenshot_with_som,
          ],
      )
      parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
      outcome, error_description, progress_status = (
          parsed_result_action_reflect['outcome'], 
          parsed_result_action_reflect['error_description'], 
          parsed_result_action_reflect['progress_status']
      )
      action_reflection_end_time = time.time()

      if "A" in outcome: # Successful. The result of the last action meets the expectation.
          action_outcome = "A"
      elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
          action_outcome = "B"
      elif "C" in outcome: # Failed. The last action produces no changes.
          action_outcome = "C"
      else:
          raise ValueError("Invalid outcome:", outcome)
    else:
      outcome = "A"
      error_description = "None"
      progress_status = self.info_pool.progress_status + "\n" + "Answer to the request: " + converted_action.text
    
    print('Action reflection prompt: ' + prompt_action_reflect)
    print()
    print('Action reflection outcome: ' + action_outcome)
    print('Action reflection error description: ' + error_description)
    print('Action reflection progress status: ' + progress_status, "\n")
    Image.fromarray(before_screenshot_with_som).save("screenshots/action_reflection_input_before.png")
    Image.fromarray(after_screenshot_with_som).save("screenshots/action_reflection_input_after.png")
    import pdb; pdb.set_trace()

    # update action history
    self.info_pool.action_history.append(json.loads(converted_action.json_str()))
    self.info_pool.summary_history.append(action_description)
    self.info_pool.action_outcomes.append(action_outcome)
    self.info_pool.error_descriptions.append(error_description)
    self.info_pool.progress_status = progress_status
    self.info_pool.progress_status_history.append(progress_status)


    #################
    ### NoteKeeper ###
    #################
    if action_outcome == "A":
        print("\n### NoteKeeper ... ###\n")
        # if previous action is successful, record the important content
        notetaking_start_time = time.time()
        prompt_note = notetaker.get_prompt(self.info_pool)
        output_note, if_safe, raw_response = self.llm.predict_mm(
          prompt_note,
          [after_screenshot_with_som],
        )
        parsed_result_note = notetaker.parse_response(output_note)
        important_notes = parsed_result_note['important_notes']
        self.info_pool.important_notes = important_notes
        notetaking_end_time = time.time()
        
        print('Note taking prompt: ' + prompt_note)
        print()
        print('Important notes: ' + important_notes, "\n")
        Image.fromarray(after_screenshot_with_som).save("screenshots/note_taking_input.png")
        import pdb; pdb.set_trace()

    return base_agent.AgentInteractionResult(
        False,
        asdict(self.info_pool),
    )
