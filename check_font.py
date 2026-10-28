
import matplotlib
from matplotlib import font_manager

print("Matplotlib 配置目录：")
print(matplotlib.get_configdir())

print("\n检查中文字体：")

fonts = ["Microsoft YaHei", "SimHei", "SimSun"]

for font in fonts:
    try:
        path = font_manager.findfont(
            font,
            fallback_to_default=False
        )
        print(font, "已找到：", path)
    except ValueError:
        print(font, "未找到")
