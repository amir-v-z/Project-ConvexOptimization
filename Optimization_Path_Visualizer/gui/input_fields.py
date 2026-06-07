import tkinter as tk

def create_input_fields(root):

    frame = tk.Frame(root)
    frame.pack(pady=20)

    tk.Label(frame, text="f(x) or f(x,y) =").grid(row=0, column=0)
    func_entry = tk.Entry(frame, width=20)
    func_entry.grid(row=0, column=1)
    # func_entry.insert(0, "x**2")

    tk.Label(frame, text="f'(x) or f'(x,y) =").grid(row=1, column=0)
    derivative_entry = tk.Entry(frame, width=20)
    derivative_entry.grid(row=1, column=1)
    derivative_entry.insert(0, "exp-> dx ; dy")

    tk.Label(frame, text="Learning Rate =").grid(row=2, column=0)
    lr_entry = tk.Entry(frame, width=20)
    lr_entry.grid(row=2, column=1)
    lr_entry.insert(0, "0.1")

    tk.Label(frame, text="Initial x =").grid(row=3, column=0)
    x0_entry = tk.Entry(frame, width=20)
    x0_entry.grid(row=3, column=1)
    # x0_entry.insert(0, "4")

    tk.Label(frame, text="Initial y =").grid(row=4, column=0)
    y0_entry = tk.Entry(frame, width=20)
    y0_entry.grid(row=4, column=1)
    # y0_entry.insert(0, "4")

    tk.Label(frame, text="Iterations =").grid(row=5, column=0)
    iter_entry = tk.Entry(frame, width=20)
    iter_entry.grid(row=5, column=1)
    iter_entry.insert(0, "30")

    return {
        "func": func_entry,
        "derivative": derivative_entry,
        "lr": lr_entry,
        "x0": x0_entry,
        "y0": y0_entry,
        "iterations": iter_entry
    }