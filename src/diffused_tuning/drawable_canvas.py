from PIL import Image
from urllib import request
import diffused_tuning.util as util

import param
import panel as pn

from panel.reactive import ReactiveHTML


class Canvas(ReactiveHTML):
    # adapted from https://panel.holoviz.org/gallery/components/CanvasDraw.html
    color = param.Color(default="#FFFFFF")

    draw_radius = param.Integer(default=0, bounds=(0, 200))
    background_uri = param.String()
    mask_uri = param.String()
    size = param.Integer(default=768)

    def __init__(self):
        super().__init__()
        instruction_text = (
            "Click and drag to select areas for re-drawing. Use the slider to draw circles instead of areas."
        )
        self.layout = pn.Column(instruction_text, self, self.param.draw_radius)

    # style="border: 1px solid; background: url(""" f'"{data_uri}"' """);"
    _template = """
    <canvas
      id="canvas"
      width="${size}"
      height="${size}"
      onmousedown="${script('start')}"
      onmousemove="${script('draw')}"
      onmouseup="${script('end')}"
    >
    </canvas>
    <button id="clear" onclick='${script("clear")}'>Clear</button>
    """

    _scripts = {
        "render": """
            state.ctx = canvas.getContext("2d")
            canvas.style = "border: 2px solid; background-image: url('" + data.background_uri + "');"
            state.ctx.fillStyle = "#FFFFFF";
        """,
        "start": """
            state.start = event
            state.ctx.beginPath();
        """,
        "draw": """
            if (state.start == null)
                return
            if (data.draw_radius != 0)
                state.ctx.beginPath();
            state.ctx.ellipse(event.offsetX, event.offsetY, data.draw_radius, data.draw_radius, 0, 0, 2 * Math.PI)
            state.ctx.fill()
        """,
        "end": """
            delete state.start
            data.mask_uri = canvas.toDataURL();
        """,
        "clear": """
            canvas.style = "border: 2px solid; background-image: url('" + data.background_uri + "');"
            state.ctx.clearRect(0, 0, canvas.width, canvas.height);
            data.mask_uri = canvas.toDataURL();
        """,
    }


if __name__ == "__main__":

    class InpaintingPanel(pn.viewable.Viewer, param.Parameterized):
        def __init__(self, canvas: Canvas):
            self.canvas = canvas
            super().__init__()

        @pn.depends("canvas.mask_uri")
        def img(self):
            if not self.canvas.mask_uri:
                return None
            return util.b64_to_img(util.dataurl_to_b64(self.canvas.mask_uri))

        def __panel__(self):
            return pn.Column(self.canvas.layout, self.param, self.img)

    img = Image.open("img.png")
    canvas = Canvas(img)
    view = pn.Column(InpaintingPanel(canvas))
    view.servable()
