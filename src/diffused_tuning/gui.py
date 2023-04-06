import panel as pn
import diffused_tuning.util as util
from panel.viewable import Viewer
from PIL import Image
import threading
import subprocess
import param
import traceback

pn.extension()


def exception_handler(error):
    traceback.print_exc()
    pn.state.notifications.error(f"Error: {error!r}", duration=10_000)


pn.extension(exception_handler=exception_handler, notifications=True)


class ConfigurationPanel(param.Parameterized):
    resolution = param.Integer(default=768, bounds=(768, 2560), step=128)
    processing_time = param.Integer(default=10, bounds=(1, 100))


class GenerativePanel(Viewer, param.Parameterized):
    prompt = param.String(default="A dog with a red ball.")
    negation_prompt = param.String(default="Cat, people, asphalt, grey, dark, lowres")
    prompt_strength = param.Number(default=7.5, bounds=(0, 100))
    run = param.Action(lambda x: x.param.trigger("run"))
    progress = param.Number(default=0, precedence=-1)
    _img_hex = param.String(default="", precedence=-1)

    def __init__(self, config: ConfigurationPanel, **params):
        self.config = config
        super().__init__(**params)
        self._layout = pn.Column(
            self.param,
            self.progress_bar,
            self.image,
        )

    @property
    def img_size(self):
        return (self.config.resolution, self.config.resolution)

    @param.depends("config.processing_time", "progress", watch=True)
    def progress_bar(self):
        return pn.widgets.Progress(name="Progress", value=self.progress, max=self.config.processing_time)


    @property
    def default_model_init(self) -> list[str]:
        return [
            "python",
            "model.py",
            f"--prompt={self.prompt}",
            f"--negative_prompt={self.negation_prompt}",
            f"--size={self.config.resolution}",
            f"--num_steps={self.config.processing_time}",
            f"--guidance={self.prompt_strength}",
        ]

    @param.depends("run", watch=True)
    def generate_image(self):
        self.progress = -1
        process = subprocess.Popen(
            [*self.default_model_init, "--model-version=generate"],
            stdout=subprocess.PIPE,
        )
        threading.Thread(target=self.update_progress, args=(process.stdout,)).start()

    def update_progress(self, pipe: subprocess.PIPE):
        while (line := pipe.readline().strip()) != b"":
            if line.startswith(b"PROGRESS="):
                self.progress = int(line.replace(b"PROGRESS=", b""))
            elif line.startswith(b"IMAGE="):
                self._img_hex = line.replace(b"IMAGE=", b"").decode("utf-8")
        self.progress = 0

    @pn.depends("_img_hex")
    def image(self):
        kwargs = {"mode": "RGB", "size": self.img_size}
        img = util.parse_img_from_hex(self._img_hex) if self._img_hex else Image.new(**kwargs, color="WHITE")
        return pn.pane.PNG(img)

    def __panel__(self):
        return self._layout


def create_panel():
    config = ConfigurationPanel(name="Global configuration")
    generative_panel = GenerativePanel(config, name="Generate image")
    return pn.Column(config, generative_panel)


panel = create_panel()
if __name__ == "__main__":
    pn.serve(panel)
else:
    panel.servable()
