# from https://panel.holoviz.org/gallery/components/CanvasDraw.html
from PIL import Image
from urllib import request

import param
import panel as pn

from panel.reactive import ReactiveHTML

pn.extension(template="fast")  # TODO: remove this line


class Canvas(ReactiveHTML):
    color = param.Color(default="#FFFFFF")

    draw_radius = param.Integer(default=25, bounds=(1, 200))
    uri = param.String()
    size = param.Integer(default=768)

    def __init__(self):
        super().__init__()
        instruction_text = 'Click and drag to select areas for re-drawing'
        self.layout = pn.Column(instruction_text, self, self.param.draw_radius)

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
    """

    _scripts = {
        "render": """
          state.ctx = canvas.getContext("2d")
        """,
        "start": """
          state.start = event
          state.ctx.beginPath()
          state.ctx.fill()
        """,
        "draw": """
          if (state.start == null)
            return
          state.ctx.ellipse(event.offsetX, event.offsetY, data.draw_radius, data.draw_radius, 0, 0, 2 * Math.PI)
          state.ctx.fill()
        """,
        "end": """
          delete state.start
          data.uri = canvas.toDataURL();
        """,
        "clear": """
          state.ctx.clearRect(0, 0, canvas.width, canvas.height);
          data.uri = canvas.toDataURL();
        """,
        "draw_radius": """
          state.ctx.lineWidth = data.draw_radius;
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
            self.canvas.layout, self.param, self.img
        )


canvas = Canvas()
view = pn.Column(InpaintingPanel(canvas))
view.servable()
