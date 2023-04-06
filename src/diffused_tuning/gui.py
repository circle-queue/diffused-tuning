import subprocess
import threading
import traceback
from pathlib import Path

import panel as pn
import param
from panel.viewable import Viewer
from PIL import Image

import diffused_tuning.util as util

pn.extension()

WIDTH = 800


def exception_handler(error):
    traceback.print_exc()
    pn.state.notifications.error(f"Error: {error!r}", duration=10_000)


pn.extension(exception_handler=exception_handler, notifications=True)


class ConfigurationPanel(param.Parameterized):
    resolution = param.Integer(default=768, bounds=(768, 1280), step=128)
    processing_time = param.Integer(default=10, bounds=(1, 100))
    prompt = param.String(default="A dog with a red ball.")
    negation_prompt = param.String(default="Cat, people, asphalt, grey, dark, lowres")
    prompt_strength = param.Number(default=7.5, bounds=(0, 100))


class InpaintPanel(Viewer, param.Parameterized):
    img = param.String(default="TODO")

    def __panel__(self):
        return pn.Column()


class GeneratePanel(Viewer, param.Parameterized):
    def __panel__(self):
        return pn.Column()


class ModelsPanel(Viewer, param.Parameterized):
    run = param.Action(lambda x: x.param.trigger("run"))
    _progress = param.Number(default=0, precedence=-1)
    _img_hex = param.String(default="", precedence=-1)

    _model_types = param.ObjectSelector(
        objects=[
            GeneratePanel(name="Generate image"),
            InpaintPanel(name="Modify image"),
        ],
        precedence=-1,
    )

    def __init__(self, config: ConfigurationPanel, **params):
        self.config = config
        self.model_tabs = pn.Tabs(*self.param._model_types.objects)

        super().__init__(**params)
        self._layout = pn.Column(self.model_tabs, self.param.run, self.progress_bar, self.image, width=800)

    @property
    def img_size(self):
        return (self.config.resolution, self.config.resolution)

    @param.depends("config.processing_time", "_progress", watch=True)
    def progress_bar(self):
        return pn.widgets.Progress(
            name="Progress", value=self._progress, max=self.config.processing_time, sizing_mode="scale_width"
        )

    @property
    def default_model_init(self) -> list[str]:
        c = self.config
        return [
            "python",
            "model.py",
            f"--prompt={c.prompt}",
            f"--negative_prompt={c.negation_prompt}",
            f"--size={c.resolution}",
            f"--num_steps={c.processing_time}",
            f"--guidance={c.prompt_strength}",
        ]

    @param.depends("run", watch=True)
    def run_model(self):
        self._progress = -1
        process = subprocess.Popen([*self.default_model_init, "--model-type=generate"], stdout=subprocess.PIPE)
        threading.Thread(target=self.update_progress, args=(process.stdout,)).start()

    def update_progress(self, pipe: subprocess.PIPE):
        while (line := pipe.readline().strip()) != b"":
            if line.startswith(b"PROGRESS="):
                self._progress = int(line.replace(b"PROGRESS=", b""))
            elif line.startswith(b"IMAGE="):
                self._img_hex = line.replace(b"IMAGE=", b"").decode("utf-8")
        self._progress = 0

    @pn.depends("_img_hex")
    def image(self):
        kwargs = {"mode": "RGB", "size": self.img_size}
        img = util.hex_to_img(self._img_hex) if self._img_hex else Image.new(**kwargs, color="WHITE")
        return pn.pane.PNG(img)

    def __panel__(self):
        return self._layout


def create_panel():
    config = ConfigurationPanel(name="Global configuration")
    generative_panel = ModelsPanel(config, name="Generate image")
    return pn.Column(config, generative_panel)


panel = create_panel()
# if __name__ == "__main__":
#     pn.serve(panel)
# else:
panel.servable()
