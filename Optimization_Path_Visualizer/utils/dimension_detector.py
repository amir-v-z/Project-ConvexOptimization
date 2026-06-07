# تشخیص 1بعدی یا 2بعدی بودن تابع
def detect_dimension(func_str):

    if "y" in func_str:
        return 2
    else:
        return 1