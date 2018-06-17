import platform
def get_root():
    if platform.system() == "Windows":
        return "D:/tfroot/"
    elif platform.system() == "":
        return ""
    return ""

print(platform.system())