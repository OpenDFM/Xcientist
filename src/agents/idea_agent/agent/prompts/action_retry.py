ACTION_RETRY_PROMPT = """
Your previous response "{invalid_action}" is not part of the allowed action space: {action_space}.
Re-evaluate the latest step and respond with ONE valid action name exactly as spelled in the list.
Latest planner observation:
{step}
"""
