from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
import copy
import re
import json
import time
import os

def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)


def extract_json_object(text, json_type="dict"):
    """
    Extracts a JSON object from a text string.

    Parameters:
    - text (str): The text containing the JSON data.
    - json_type (str): The type of JSON structure to look for ("dict" or "list").

    Returns:
    - dict or list: The extracted JSON object, or None if parsing fails.
    """
    try:
        if "//" in text:
            # Remove comments starting with //
            text = re.sub(r'//.*', '', text)
        if "# " in text:
            # Remove comments starting with #
            text = re.sub(r'#.*', '', text)
        # Try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # Not a valid JSON, proceed to extract from text

    # Define patterns for extracting JSON objects or arrays
    json_pattern = r"({.*?})" if json_type == "dict" else r"(\[.*?\])"

    # Search for JSON enclosed in code blocks first
    code_block_pattern = r"```json\s*(.*?)\s*```"
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Failed to parse JSON inside code block

    # Fallback to searching the entire text
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # Try the next match

    # If all attempts fail, return None
    return None

########################


@dataclass
class InfoPool:
    """Keeping track of all information across the agents."""

    # User input / accumulated knowledge
    instruction: str = ""
    additional_knowledge: str = ""
    # shortcuts: dict = field(default_factory=dict)

    ## Perception
    # width: int = 1080
    # height: int = 2340
    # perception_infos_pre: list = field(default_factory=list) # List of clickable elements pre action
    # keyboard_pre: bool = False # keyboard status pre action
    # perception_infos_post: list = field(default_factory=list) # List of clickable elements post action
    # keyboard_post: bool = False # keyboard status post action
    
    ui_elements_list_before: str = "" # List of UI elements with index
    ui_elements_list_after: str = "" # List of UI elements with index

    # Working memory
    summary_history: list = field(default_factory=list)  # List of action descriptions
    action_history: list = field(default_factory=list)  # List of actions
    action_outcomes: list = field(default_factory=list)  # List of action outcomes
    error_descriptions: list = field(default_factory=list)

    last_summary: str = ""  # Last action description
    last_action: str = ""  # Last action
    last_action_thought: str = ""  # Last action thought
    important_notes: str = ""
    
    error_flag_plan: bool = False # if an error is not solved for multiple attempts with the executor
    error_description_plan: bool = False # explanation of the error for modifying the plan

    # Planning
    plan: str = ""
    progress_status: str = ""
    progress_status_history: list = field(default_factory=list)
    finish_thought: str = ""
    current_subgoal: str = ""
    # prev_subgoal: str = ""
    err_to_manager_thresh: int = 2

    # future tasks
    future_tasks: list = field(default_factory=list)


class BaseAgent(ABC):
    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        pass
    @abstractmethod
    def parse_response(self, response: str) -> dict:
        pass


ALL_APPS = [
"simple calendar pro: A calendar app.",
"settings: The Android system settings app for managing device settings such as Bluetooth, Wi-Fi, and brightness.",
"markor: A note-taking app for creating, editing, deleting, and managing notes and folders.",
"broccoli: A recipe management app.",
"pro expense: An expense tracking app.",
"simple sms messenger: An SMS app for sending, replying to, and resending text messages.",
"opentracks: A sport tracking app for recording and analyzing activities.",
"tasks: A task management app for tracking tasks, due dates, and priorities.",
"clock: An app with stopwatch and timer functionality.",
"joplin: A note-taking app.",
"retro music: A music player app.",
"simple gallery pro: An app for viewing images.",
"camera: An app for taking photos and videos.",
"chrome: A web browser app.",
"contacts: An app for managing contact information.",
"osmand: A maps and navigation app with support for adding location markers, favorites, and saving tracks.",
"vlc: A media player app for playing media files.",
"audio recorder: An app for recording and saving audio clips.",
"files: A file manager app for the Android filesystem, used for deleting and moving files.",
"simple draw pro: A drawing app for creating and saving drawings.",
]


class Manager(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to track progress and devise high-level plans to achieve the user's requests.\n\n"
        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### All Available Apps ###\n"
        all_apps_str = ""
        for app_str in ALL_APPS:
            all_apps_str += f"- {app_str}\n"
        prompt += f"{all_apps_str}\n"

        task_specific_note = ""
        if ".html" in info_pool.instruction:
            task_specific_note = "NOTE: The .html file may contain additional interactable elements, such as a drawing canvas or a game. Do not open other apps without completing the task in the .html file.\n"

        if info_pool.plan == "":
            # first time planning
            prompt += "---\n"
            prompt += "Make a high-level plan to achieve the user's request. If the request is complex, break it down into subgoals. The screenshot displays the starting state of the phone.\n"
            prompt += "IMPORTANT: For requests that explicitly require an answer, always add 'perform the `answer` action' as the last step to the plan!\n"
            if task_specific_note != "":
                prompt += f"{task_specific_note}\n\n"
            
            # # shortcuts
            # if info_pool.shortcuts != {}:
            #     prompt += "### Available Shortcuts from Past Experience ###\n"
            #     prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided as a reference.\n"
            #     for shortcut, value in info_pool.shortcuts.items():
            #         prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
            #     prompt += "\n"
            # prompt += "---\n"

            prompt += "Provide your output in the following format which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "A detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "1. first subgoal\n"
            prompt += "2. second subgoal\n"
            prompt += "...\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The first subgoal you should work on.\n\n"
        else:
            # continue planning
            prompt += "### Current Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
            prompt += "### Previous Subgoal ###\n"
            prompt += f"{info_pool.current_subgoal}\n\n"
            prompt += f"### Last Action ###\n"
            prompt += f"{info_pool.last_action}\n\n"
            prompt += f"### Progress Status ###\n"
            prompt += f"{info_pool.progress_status}\n\n"
            prompt += "### Important Notes ###\n"
            if info_pool.important_notes != "":
                prompt += f"{info_pool.important_notes}\n\n"
            else:
                prompt += "No important notes recorded.\n\n"
            if info_pool.error_flag_plan:
                prompt += "### Potentially Stuck! ###\n"
                prompt += "You have encountered several failed attempts. Here are some logs:\n"
                k = info_pool.err_to_manager_thresh
                recent_actions = info_pool.action_history[-k:]
                recent_summaries = info_pool.summary_history[-k:]
                recent_err_des = info_pool.error_descriptions[-k:]
                for i, (act, summ, err_des) in enumerate(zip(recent_actions, recent_summaries, recent_err_des)):
                    prompt += f"- Attempt: Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"

            prompt += "---\n"
            # prompt += "The sections above provide an overview of the previous plan you are following, the current subgoal you are working on, the overall progress made, and any important notes you have recorded. The screenshot displays the current state of the phone.\n"
            prompt += "Carefully assess the current status and the provided screenshot. Check if the current plan needs to be revised.\n Determine if the task has been fully completed. If you are confident that no further actions are required, mark the task as \"Finished\" in your output. If the task is not finished, outline the next steps. If you are stuck with errors, think step by step about whether the overall plan needs to be revised to address the error.\n"
            prompt += "NOTE: If the current situation prevents proceeding with the original plan or requires clarification from the user, make reasonable assumptions and revise the plan accordingly. Act as though you are the user in such cases.\n"
            prompt += "IMPORTANT: For requests that explicitly require an answer, always add 'perform the `answer` action' as the last step to the plan! You SHOULD NOT finish such tasks unless the last `action_type` == `answer`.\n"
            if task_specific_note != "":
              prompt += f"{task_specific_note}\n\n"

            # # shortcuts
            # if info_pool.shortcuts != {}:
            #     prompt += "### Available Shortcuts from Past Experience ###\n"
            #     prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided only as a reference.\n"
            #     for shortcut, value in info_pool.shortcuts.items():
            #         prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
            #     prompt += "\n"
            # prompt += "---\n"
            
            prompt += "Provide your output in the following format, which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "An explanation of your rationale for the updated plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "Updated high-level plan\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The next subgoal to work on. If all subgoals are completed, write \"Finished\". If the user requests an answer, always remember to check whether the last `action_type` is `answer` before finishing.\n"
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}



from android_world.env.json_action import *

ATOMIC_ACTION_SIGNITURES = {
    # STATUS: {
    #     "arguments": ["goal_status"],
    #     "description": lambda info: "If you think the task has been completed, finish the task by using the `status` action with `complete` as goal_status; If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with `infeasible` as goal_status."
    # },

    ANSWER: {
        "arguments": ["text"],
        "description": lambda info: "Answer user's question. Usage example: {\"action_type\": \"answer\", \"text\": <answer_text>}"
    },
    CLICK: {
        "arguments": ["index"],
        "description": lambda info: "Click/tap on an element on the screen. We have added marks (bounding boxes with numeric indexes on their TOP LEFT corner) to most of the UI elements in the screenshot, use the numeric index to indicate which element you want to click. Usage Example: {\"action_type\": \"click\", \"index\": <target_index>}"
    },
    LONG_PRESS: {
        "arguments": ["index"],
        "description": lambda info: "Long press on an element on the screen, similar with the click action above, use the numeric label on the bounding box to indicate which element you want to long press. Usage Example: {\"action_type\": \"long_press\", \"index\": <target_index>}"
    },
    INPUT_TEXT: {
        "arguments": ["text", "index"],
        "description": lambda info: "Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the numeric label on the bounding box to indicate the target text field. Usage Example: {\"action_type\": \"input_text\", \"text\": <text_input>, \"index\": <target_index>}"
    },
    KEYBOARD_ENTER: {
        "arguments": [],
        "description": lambda info: "Press the Enter key. Usage example: {\"action_type\": \"keyboard_enter\"}"
    },
    NAVIGATE_HOME: {
        "arguments": [],
        "description": lambda info: "Navigate to the home screen. Usage example: {\"action_type\": \"navigate_home\"}"
    },
    NAVIGATE_BACK: {
        "arguments": [],
        "description": lambda info: "Navigate back. Usage example: {\"action_type\": \"navigate_back\"}"
    },
    SCROLL: {
        "arguments": ["direction", "index"],
        "description": lambda info: "Scroll the screen or a scrollable UI element in one of the four directions (`direction` is choosen from: [up, down, left, right]). For the `index` argument, use the same numeric index if you want to scroll a specific UI element; remove the `index` field for scroll the whole screen. Usage Example: {\"action_type\": \"scroll\", \"direction\": \"down\", \"index\": <target_index>} or {\"action_type\": \"scroll\", \"direction\": \"down\"}"
    },
    OPEN_APP: {
        "arguments": ["app_name"],
        "description": lambda info: "Open an app (nothing will happen if the app is not installed); Usage example: {\"action_type\": \"open_app\", \"app_name\": <name>}"
    },
    WAIT: {
        "arguments": [],
        "description": lambda info: "Wait for the screen to update. Usage example: {\"action_type\": \"wait\"}"
    }
}


GENERAL_KNOWN_ISSUES = (
    '- When doing searching, such as in the Joplin app, sometimes the dropdown result UI elements can be shown as `is_clickable: False`. This is a bug.'
    ' YOU SHOULD still try to click that element, even if it is not clickable. Usually the correct UI element has similar/same text as the search query, and having an index larger than 2.\n'
)
KNOWN_ISSUE_DRAW = (
    '- There is currently no `draw` action. If you want to draw someting, try using the `scroll` action with the `index` argument specified to the canvas UI element, for example 10. If the canvas covers entire screen, then no need to specify `index`. Draw a simple line using `scroll` with any direction would be fine. If drawing multiple lines, it would be better to use different direction. \n'
)

class Executor(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to decide the next action to perform based on the current state of the phone and the user's request.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Screen Information ###\n"
        prompt += "The current screenshot with bounding boxes and labels added is given to you. Here is a list of detailed information for some of the UI elements (notice that some elements in this list may not be visible in the current screen and so you can not interact with it, can try to scroll the screen to reveal it first), the numeric indexes are consistent with the ones in the labeled screenshot:\n"
        prompt += info_pool.ui_elements_list_before
        prompt += "\n\n"


        if info_pool.additional_knowledge != "":
            prompt += "### Guidelines ###\n"
            # prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.additional_knowledge}\n"

        if GENERAL_KNOWN_ISSUES != "":
            prompt += "### Important Known Issues ###\n"
            prompt += f"{GENERAL_KNOWN_ISSUES}\n"

            if "draw" in info_pool.instruction:
                prompt += f"{KNOWN_ISSUE_DRAW}\n"

        prompt += "### Collected Task-Related Notes ###\n"
        if info_pool.important_notes != "":
            prompt += "Here are some potentially important content relevant to the user's request you already recorded:\n"
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "---\n"
        # prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"
        
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions.\n\n"
        
        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `action_type(arguments): description` as follows:\n"

        # if info_pool.keyboard_pre:
        #     for action, value in ATOMIC_ACTION_SIGNITURES.items():
        #         prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        # else:
        #     for action, value in ATOMIC_ACTION_SIGNITURES.items():
        #         if "Type" not in action:
        #             prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        #     prompt += "NOTE: Unable to type. The keyboard has not been activated. To type, please activate the keyboard by tapping on an input box or using a shortcut, which includes tapping on an input box first.”\n"

        for action, value in ATOMIC_ACTION_SIGNITURES.items():
            prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        
        prompt += "\n"

        # prompt += "#### Shortcuts ####\n"
        # if info_pool.shortcuts != {}:
        #     prompt += "The shortcut functions are listed in the format of `name(arguments): description | Precondition: precondition` as follows:\n"
        #     for shortcut, value in info_pool.shortcuts.items():
        #         prompt += f"- {shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        # else:
        #     prompt += "No shortcuts are available.\n"
        # prompt += "\n"

        prompt += "### Latest Action History ###\n"
        if info_pool.action_history != []:
            prompt += "Recent actions you took previously and whether they were successful:\n"
            num_actions = min(5, len(info_pool.action_history))
            latest_actions = info_pool.action_history[-num_actions:]
            latest_summary = info_pool.summary_history[-num_actions:]
            latest_outcomes = info_pool.action_outcomes[-num_actions:]
            error_descriptions = info_pool.error_descriptions[-num_actions:]
            action_log_strs = []
            for act, summ, outcome, err_des in zip(latest_actions, latest_summary, latest_outcomes, error_descriptions):
                if outcome == "A":
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
                prompt += action_log_str
                action_log_strs.append(action_log_str)
            
            prompt += "\n"
            
            # last_action_outcome = info_pool.action_outcomes[-1]
            # if last_action_outcome == "B":
            #     prompt += "NOTE: Since the last action failed and resulted in an incorrect page, I have reverted the phone state to its previous state for you.\n\n"
            # elif last_action_outcome == "C":
            #     prompt += "NOTE: Since the last action failed and did not have any effect, the state of the phone remains unchanged.\n\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        # prompt += "Provide a detailed explanation of your rationale for the chosen action. IMPORTANT: If you decide to use a shortcut, first verify that its precondition is met in the current phone state. For example, if the shortcut requires the phone to be at the Home screen, check whether the current screenshot shows the Home screen. If not, perform the appropriate atomic actions instead.\n\n"
        
        prompt += "Provide a detailed explanation of your rationale for the chosen action.\n\n"

        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options provided. IMPORTANT: Do NOT return invalid actions like null or stop. Do NOT repeat previously failed actions multiple times.\n"
        # prompt += "Use shortcuts whenever possible to expedite the process, but make sure that the precondition is met.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the `action_type` and the arguments of the action. For example, if you want to open an App, you should write {\"action_type\":\"open_app\", \"app_name\":<name>}. If an action does not require arguments, such as `navigate_home`, just include the `action_type` field, e.g., {\"action_type\": \"navigate_home\"}.\n\n"
        
        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ", " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}


class ActionReflector(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are an agent who can operate an Android phone on behalf of a user. Your goal is to verify whether the last action produced the expected behavior and to keep track of the overall progress.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "---\n"
        prompt += "The two attached images are phone screenshots taken before and after your last action. Below are two lists containing detailed information about some of the UI elements in the 'before' and 'after' screenshots.\n"
        prompt += "### UI elements before the action ###\n"
        prompt += info_pool.ui_elements_list_before
        prompt += "\n"
        prompt += "### UI elements after the action ###\n"
        prompt += info_pool.ui_elements_list_after
        prompt += "\n\n"


        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        # assert info_pool.last_action != ""
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the `scroll` action, it may take multiple attempts to display the expected content. Thus, for a `scroll` action, if the screen shows new content, it usually meets the expectation.\nPro Tip: In rare cases, the UI might not visibly change even if a click action is performed correctly — for example, when clicking on a color before drawing. In such situations, you can assume the action was successful and proceed — for example, by drawing a line.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your response as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        prompt += "### Progress Status ###\n"
        prompt += "If the action was successful or partially successful, update the progress status. If the action failed, copy the previous progress status.\n"
        prompt += "IMPORTANT: For requests that require an answer, if the last `action_type` is not `answer`, add a note that an additional `answer` action is needed.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Progress Status ###")[0].replace("\n", " ").replace("  ", " ").strip()
        progress_status = response.split("### Progress Status ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"outcome": outcome, "error_description": error_description, "progress_status": progress_status}


class Notetaker(BaseAgent):

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request.\n\n"

        prompt += "### User Request ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Progress Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"

        prompt += "### Existing Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        # prompt += "### Current Screen Information ###\n"
        # prompt += (
        #     f"The attached image is a screenshot showing the current state of the phone. "
        #     f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        # )
        # prompt += (
        #     "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons. "
        #     "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
        #     "and y represents the vertical pixel position (from top to bottom)."
        # )
        # prompt += "The extracted information is as follows:\n"

        # for clickable_info in info_pool.perception_infos_post:
        #     if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
        #         prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        # prompt += "\n"
        # prompt += (
        #     "Note that this information might not be entirely accurate. "
        #     "You should combine it with the screenshot to gain a better understanding."
        # )
        # prompt += "\n\n"

        prompt += "### Current Screen Information ###\n"
        prompt += "The current screenshot with bounding boxes and labels added is given to you. Here is a list of detailed information for some of the UI elements (notice that some elements in this list may not be visible in the current screen and so you can not interact with it, can try to scroll the screen to reveal it first), the numeric indexes are consistent with the ones in the labeled screenshot:\n"
        prompt += info_pool.ui_elements_list_after
        prompt += "\n\n"


        prompt += "---\n"
        prompt += "Carefully examine the information above to identify any important content on the current screen that needs to be recorded. IMPORTANT: Do not take notes on low-level actions; only keep track of significant textual or visual information relevant to the user's request. Do not repeat user request or progress status.\n\n"

        prompt += "Provide your output in the following format:\n"
        prompt += "### Important Notes ###\n"
        prompt += "The updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        important_notes = response.split("### Important Notes ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"important_notes": important_notes}