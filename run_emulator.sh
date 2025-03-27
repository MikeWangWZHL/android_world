# Typically it's located in ~/Android/Sdk/emulator/emulator or
# ~/Library/Android/sdk/emulator/emulator

# run simulator
EMULATOR_NAME=AndroidWorldAvd # From previous step
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -grpc 8554

# ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554

# ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -snapshot testingSetup -grpc 8554
