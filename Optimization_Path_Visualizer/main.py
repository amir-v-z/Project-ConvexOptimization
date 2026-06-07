from gui.interface import create_main_window
from gui.buttons import create_buttons
from gui.input_fields import create_input_fields

def main():

    root = create_main_window()

    inputs = create_input_fields(root)

    create_buttons(root, inputs)

    root.mainloop()

if __name__ == "__main__":
    main()