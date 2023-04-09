import subprocess
import numpy as np
import threading
import traceback
from pathlib import Path

import panel as pn
import param
from panel.viewable import Viewer
from PIL import Image

import diffused_tuning.util as util
import diffused_tuning.drawable_canvas

pn.extension("notifications")

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


class InpaintPanel(pn.viewable.Viewer, param.Parameterized):
    mask_b64 = param.String(default="", precedence=-1)
    img_b64 = param.String(default="", precedence=-1)
    canvas = diffused_tuning.drawable_canvas.Canvas()

    @property
    def model_args(self):
        if not self.mask_b64:
            raise ValueError("Please draw a mask to replace first.")
        mask, img = util.compressed_b64(self.mask_b64), util.compressed_b64(self.img_b64)
        return ["--model-type=inpaint", f"--inpaint-mask-b64={mask}"]

    @pn.depends("canvas.mask_uri")
    def img(self):
        if not self.canvas.mask_uri:
            return
        b64_str = util.dataurl_to_b64(self.canvas.mask_uri)
        rgba_img = util.b64_to_img(b64_str)
        black_and_white_img = Image.fromarray(np.array(rgba_img).max(axis=-1))
        arr = np.array(black_and_white_img)
        self.mask_b64 = util.img_to_b64(black_and_white_img)

    def update_drawing_background(self, img_b64: str):
        self.img_b64 = img_b64
        self.canvas.background_uri = util.b64_to_dataurl(img_b64)

    def __panel__(self):
        return pn.Column(self.canvas.layout, self.param, self.img)


class GeneratePanel(Viewer, param.Parameterized):
    @property
    def model_args(self):
        return ["--model-type=generate"]

    def __panel__(self):
        return pn.Column()


class ModelsPanel(Viewer, param.Parameterized):
    run = param.Action(lambda x: x.param.trigger("run"))
    _progress = param.Number(default=0, precedence=-1)
    _img_b64 = param.String(default=util.img_to_b64(Image.open(util.IMG_FILEPATH)), precedence=-1)

    active_model = param.ObjectSelector(
        objects=[
            GeneratePanel(name="Generate image"),
            InpaintPanel(name="Modify image"),
        ],
        precedence=-1,
    )

    def __init__(self, config: ConfigurationPanel, **params):
        self.config = config
        self.model_tabs = pn.Tabs(*self.param.active_model.objects, dynamic=True)

        super().__init__(**params)
        self.active_model = self.param.active_model.objects[self.model_tabs.active]
        self._update_inpaint_background()
        self._layout = pn.Column(self.model_tabs, self.param.run, self.progress_bar, self.image, width=800)

    @property
    def img_size(self):
        return (self.config.resolution, self.config.resolution)

    @pn.depends("model_tabs.active", watch=True)
    def _update_active_model(self):
        self.active_model = self.param.active_model.objects[self.model_tabs.active]

    @pn.depends("_img_b64", watch=True)
    def _update_inpaint_background(self):
        inpaint_pane = next(pane for pane in self.param.active_model.objects if isinstance(pane, InpaintPanel))
        inpaint_pane.update_drawing_background(self._img_b64)

    @pn.depends("config.processing_time", "_progress", watch=True)
    def progress_bar(self):
        return pn.widgets.Progress(
            name="Progress", value=self._progress, max=self.config.processing_time, sizing_mode="scale_width"
        )

    @property
    def default_model_init(self) -> list[str]:
        c = self.config
        return [
            "python",
            str(util.PKG_ROOT / "model.py"),
            f"--prompt={c.prompt}",
            f"--negative_prompt={c.negation_prompt}",
            f"--size={c.resolution}",
            f"--num_steps={c.processing_time}",
            f"--guidance={c.prompt_strength}",
        ]

    @param.depends("run", watch=True)
    def run_model(self):
        self._progress = -1
        process = subprocess.Popen([*self.default_model_init, *self.active_model.model_args], stdout=subprocess.PIPE)
        threading.Thread(target=self.update_progress, args=(process.stdout,)).start()

    def update_progress(self, pipe: subprocess.PIPE):
        while (line := pipe.readline().strip()) != b"":
            if line.startswith(b"PROGRESS="):
                self._progress = int(line.replace(b"PROGRESS=", b""))
            elif line.startswith(b"IMAGE="):
                self._img_b64 = line.replace(b"IMAGE=", b"").decode("utf-8")
        self._progress = 0

    @pn.depends("_img_b64")
    def image(self):
        kwargs = {"mode": "RGB", "size": self.img_size}
        img = util.b64_to_img(self._img_b64) if self._img_b64 else Image.new(**kwargs, color="WHITE")
        return pn.pane.PNG(img)

    def __panel__(self):
        return self._layout


def create_panel():
    config = ConfigurationPanel(name="Global configuration")
    generative_panel = ModelsPanel(config, name="Generate image")
    return pn.Column(config, generative_panel)


def serve():
    pn.serve(create_panel())
