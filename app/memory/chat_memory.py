chat_sessions = {}


def get_session_history(session_id):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    return chat_sessions[session_id]















