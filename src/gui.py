import taipy.gui.builder as tgb
from taipy.gui import Gui


def start():
    def my_button_pressed(state, id):
        state.first_name = "Paul"

    first_name = "John"
    with tgb.Page() as page:
        tgb.text("First name: {first_name}")
        tgb.input("{first_name}", label="First name")
        tgb.button("Press me", on_action=my_button_pressed)

    Gui(page).run(port=8080, debug=True)
