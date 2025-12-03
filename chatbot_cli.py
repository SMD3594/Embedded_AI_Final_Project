import datetime
from quantized_cache import run_baseline
from paged_cache import run_paged as run_baseline
# from rotating_cache import run_rotating as run_baseline


def log_message(role, message):

    # Get current time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    with open("chat_log.txt", "a") as file:
        file.write(f"[{timestamp}] {role.upper()}: {message}\n")

print("Simple Chatbot (type 'quit' to exit)")

# Storing recent messages and only keeping last 5 user+bot pairs
# history = [] 
# MAX_TURNS = 5

# Asks user for text
while True:
    user_input = input("\nYou: ")

    # Runs until user exits
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    # Add user message to history
    # history.append(("user", user_input))
    # Logging user's message
    log_message("user", user_input)

    # Ask model for a reply
    reply = run_baseline(user_input)
    print("Bot:", reply)
    
   
    # history.append(("bot", reply))
    # Logging bot's reply
    log_message("bot", reply)

    # # sliding window
    # if len(history) > MAX_TURNS * 2:
    #     history = history[-MAX_TURNS * 2:]

   


    