# Typically it's located in ~/Android/Sdk/emulator/emulator or
# ~/Library/Android/sdk/emulator/emulator

# load from a file
export OPENAI_API_KEY=$(cat /Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_school_semafor)

# python minimal_task_runner.py --task=ContactsAddContact
python minimal_task_runner.py --task=BrowserDraw


# OUTPUT_DIR_NAME="testing"
# python run.py \
#   --suite_family=android_world \
#   --agent_name=t3a_gpt4 \
#   --perform_emulator_setup \
#   --tasks=ContactsAddContact \
#   --checkpoint_dir="./logs/$OUTPUT_DIR_NAME"