import platform
def get_root():
    if platform.system() == "Windows":
        return "D:/tfroot/"
    elif platform.system() == "Linux":
        return "~/tfroot/"
    return ""

print(platform.system())