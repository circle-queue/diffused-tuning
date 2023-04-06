# from https://panel.holoviz.org/gallery/components/CanvasDraw.html
from PIL import Image
from urllib import request

import param
import panel as pn

from panel.reactive import ReactiveHTML

pn.extension(template="fast")  # TODO: remove this line


class Canvas(ReactiveHTML):
    color = param.Color(default="#FFFFFF")
    line_width = param.Number(default=1, bounds=(1, 200))

    uri = param.String()
    size = param.Integer(default=768)

    _template = """
    <canvas
      id="canvas"
      style="border: 1px solid; background: url("./assets/img.png");"
      width="${size}"
      height="${size}"
      onmousedown="${script('start')}"
      onmousemove="${script('draw')}"
      onmouseup="${script('end')}"
    >
    </canvas>
    <button id="clear" onclick='${script("clear")}'>Clear</button>
    <button id="save" onclick='${script("save")}'>Save</button>
    """

    _scripts = {
        "render": """
          state.ctx = canvas.getContext("2d")
        """,
        "start": """
          state.start = event
          state.ctx.beginPath()
          state.ctx.moveTo(state.start.offsetX, state.start.offsetY)
        """,
        "draw": """
          if (state.start == null)
            return
          state.ctx.lineTo(event.offsetX, event.offsetY)
          state.ctx.stroke()
        """,
        "end": """
          delete state.start
        """,
        "clear": """
          state.ctx.clearRect(0, 0, canvas.width, canvas.height);
        """,
        "save": """
          data.uri = canvas.toDataURL();
        """,
        "line_width": """
          state.ctx.lineWidth = data.line_width;
        """,
        "color": """
          state.ctx.strokeStyle = data.color;
        """,
    }


class InpaintingPanel(pn.viewable.Viewer, param.Parameterized):
    def __init__(self, canvas: Canvas):
        self.canvas = canvas
        super().__init__()

    @pn.depends("canvas.uri")
    def img(self):
        if not self.canvas.uri:
            return None

        with request.urlopen(self.canvas.uri) as response:
            binary_str = response.read()

        return pn.pane.PNG(binary_str)

    def __panel__(self):
        return pn.Column(
            self.canvas, self.canvas.param.line_width, "", self.param, self.img
        )


canvas = Canvas()
view = pn.Column(InpaintingPanel(canvas))
view.servable()
