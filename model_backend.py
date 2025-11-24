def build_prompt(history, user_input):
    """
    Turns the conversation history into a string prompt the model will be able to understand.
    """
    prompt_lines = []

    # Add previous messages
    for role, message in history:
        if role == "user":
            prompt_lines.append(f"User: {message}")
        else:
            prompt_lines.append(f"Bot: {message}")

    # Add new user inputs
    prompt_lines.append(f"User: {user_input}")
    prompt_lines.append(f"Bot:")

    return "\n".join(prompt_lines)



def generate_reply_from_model(user_input, history):
    """
    Placeholder function.
    This is where the REAL DistilGPT-2 model will go later.
    For now, it just returns a fake reply.
    """

    prompt = build_prompt(history, user_input)

    return "This is where the real DistilGPT-2 reply will go."
