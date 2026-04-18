import wikipedia

def get_evidence(query):
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except:
        return ""