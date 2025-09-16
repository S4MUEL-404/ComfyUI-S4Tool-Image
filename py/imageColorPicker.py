class ImageColorPicker:
    """A simple color picker node that outputs a hex color string.

    This node exposes a STRING widget for a hex color (e.g. "#FFFFFF").
    A frontend extension enhances the UX by providing a floating color
    picker overlay to update this value interactively.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex color",)
    FUNCTION = "pick"
    CATEGORY = "💀S4Tool"

    def pick(self, hex: str):
        value = (hex or "").strip()
        if not value:
            value = "#000000"
        # Normalize short hex like #FFF to #FFFFFF when possible
        if value.startswith("#") and len(value) == 4:
            try:
                r, g, b = value[1], value[2], value[3]
                value = f"#{r}{r}{g}{g}{b}{b}".upper()
            except Exception:
                pass
        return (value,)


