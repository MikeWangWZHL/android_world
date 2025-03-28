# Typically it's located in ~/Android/Sdk/emulator/emulator or
# ~/Library/Android/sdk/emulator/emulator

# load from a file
export OPENAI_API_KEY=$(cat /Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school_semafor)

# python minimal_task_runner.py --task=ContactsAddContact
# python minimal_task_runner.py --task=BrowserDraw
# python minimal_task_runner.py --task=NotesMeetingAttendeeCount
# python minimal_task_runner.py --task=SimpleCalendarAnyEventsOnDate

# ## one task testing:
# OUTPUT_DIR_NAME="testing"
# MODEL_NAME="mobile_agent_e_gpt4o"
# python run.py \
#   --suite_family=android_world \
#   --agent_name=$MODEL_NAME \
#   --tasks=BrowserDraw \
#   --checkpoint_dir="./checkpoints/$OUTPUT_DIR_NAME"



OUTPUT_DIR_NAME="full_run_3_27_2025"
# # MODEL_NAME="t3a_gpt4"
MODEL_NAME="mobile_agent_e_gpt4o"
python run.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
  --checkpoint_dir="./checkpoints/$OUTPUT_DIR_NAME"