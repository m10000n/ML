from typing import Literal

from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from helper.validator import validator

# color maps: https://matplotlib.org/stable/users/explain/colors/colormaps.html


##### config start #####
_QUAL_CMAP: "_VALID_QUAL_CMAPS" = "Pastel2"
_QUAL_CMAP_ORDER: list[int] | None = [1, 2, 3, 0, 4]

_SEQ_CMAP: "_VALID_SEQ_CMAPS" = "RdPu"
##### config end #####

# fmt: off
_VALID_QUAL_CMAPS = Literal[
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20','tab20b', 'tab20c'
    ]
_VALID_SEQ_CMAPS = Literal[
    "viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn", "binary",
    "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring", "summer","autumn", "winter", "cool", "Wistia", "hot",
    "afmhot", "gist_heat", "copper"
    ]
# fmt: on


class Color:
    @validator.constraints("r", ("0 <= x <= 1"))
    @validator.constraints("g", ("0 <= x <= 1"))
    @validator.constraints("b", ("0 <= x <= 1"))
    @validator.constraints("a", ("0 <= x <= 1"))
    def __init__(self, r: float, g: float, b: float, a: float = 1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __getitem__(self, index: int) -> float:
        return (self.r, self.g, self.b, self.a)[index]

    def __len__(self) -> int:
        return 4


class QualColors:
    def __init__(
        self,
        cmap_name: _VALID_QUAL_CMAPS = _QUAL_CMAP,
        order: list[int] | None = _QUAL_CMAP_ORDER,
    ):
        self.cmap_name = cmap_name
        self.cmap = colormaps[self.cmap_name]
        self.n_colors = self.cmap.N

        if order and len(order) > self.n_colors:
            raise ValueError(
                f"The length of `order` ({len(order)}) must be less than or equal to the number of colors in the colormap ({self.n_colors})."
            )

        if order is None:
            self.order = []
        else:
            self.order = order

        self.order += [x for x in list(range(self.n_colors)) if x not in self.order]

        colors = []
        for i in self.order:
            color = self.cmap(i)
            if len(color) == 3:
                r, g, b = color
                a = 1.0
            else:
                r, g, b, a = color

            colors.append(Color(r, g, b, a))
        self.colors = tuple(colors)

    def get_n(self, n: int | None = None) -> tuple["Color", ...]:
        if n is None:
            n = self.n_colors

        if self.n_colors < 1:
            raise ValueError(f"`n` ({n}) must be greater than 0.")

        colors = self.colors[:n]
        if len(colors) < n:
            colors = colors + tuple(BLACK for _ in range(n - len(colors)))

        return colors

    def __len__(self) -> int:
        return self.n_colors


class SeqColors(ListedColormap):
    def __init__(self, cmap_name: _VALID_SEQ_CMAPS = _SEQ_CMAP):
        self.cmap_name = cmap_name

        cmap = colormaps[cmap_name]

        if hasattr(cmap, "colors") and cmap.colors is not None:
            colors = cmap.colors
        else:
            colors = [cmap(i / 255) for i in range(256)]

        super().__init__(colors, name=cmap_name)


BLACK = Color(0.0, 0.0, 0.0)
WHITE = Color(1.0, 1.0, 1.0)
GREY = Color(0.5, 0.5, 0.5)
